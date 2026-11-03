// C-only registration: BF16 ND packed GEMM with full FP32 accumulation/output.
// The original V remains in its independently loaded, unchanged DSO.
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <sstream>

extern "C" void launch_c2_bf16_gemm(uint32_t, int32_t, int32_t, aclrtStream, uint8_t*, uint8_t*,
                                  uint8_t*, uint8_t*, uint8_t*);
namespace {
constexpr int64_t kH = 5120;
constexpr int64_t kD = 512;

void matrix(const at::Tensor& x, int64_t columns, at::ScalarType dtype = at::kBFloat16) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1 && x.scalar_type() == dtype,
                "C-only matrix dtype/device mismatch");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == columns && x.is_contiguous(), "dense ND matrix required");
}
void weight(const at::Tensor& w, int64_t layout) {
    TORCH_CHECK(layout == 0, "BF16 C supports ND only");
    matrix(w, kH);
    TORCH_CHECK(w.size(0) == 1024, "weight must be [1024,5120]");
}

std::tuple<at::Tensor, at::Tensor, int64_t, std::string> prepare(
    const at::Tensor& x, const at::Tensor& w, int64_t sm, int64_t sn,
    int64_t bm, int64_t bn, int64_t mode, int64_t layout, const std::string& soc) {
    matrix(x, kH); weight(w, layout);
    TORCH_CHECK(x.device() == w.device(), "weight must be on x device");
    TORCH_CHECK(x.size(0) > 0 && x.size(0) <= std::numeric_limits<int32_t>::max(), "invalid M");
    TORCH_CHECK(sm > 0 && sm <= 4096 && sn > 0 && sn <= 1024 && 1024 % sn == 0,
                "invalid task tile; singleN must divide 1024");
    TORCH_CHECK(bm > 0 && bm <= 4096 && bm % 16 == 0 && bn > 0 && bn <= 1024 && bn % 8 == 0,
                "invalid base tile alignment");
    TORCH_CHECK(mode == 0 || mode == 1, "mode must be NORM or MDL");
    TORCH_CHECK(layout == 0 || (sn % 16 == 0 && bn % 16 == 0), "NZ N task/base tiles require 16 alignment");
    TORCH_CHECK(soc == aclrtGetSocName(), "requested SoC differs from runtime device");
    c10::DeviceGuard guard(x.device());
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance(soc.c_str());
    TORCH_CHECK(platform != nullptr, "platform query failed");
    uint64_t l1 = 0, l0a = 0, l0b = 0, l0c = 0, ub = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0a);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0b);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    C2Bf16GemmTiling t{};
    t.rows = x.size(0); t.singleM = std::min<int64_t>(sm, t.rows); t.singleN = sn; t.mode = mode;
    t.layout = layout;
    t.tasks = ((t.rows + t.singleM - 1) / t.singleM) * ((1024 + sn - 1) / sn);
    t.blocks = std::min<int64_t>(platform->GetCoreNumAic(), t.tasks);
    TORCH_CHECK(t.blocks > 0 && l1 && l0a && l0b && l0c, "invalid platform capacities");
    matmul_tiling::MatmulApiTiling tiler(*platform);
    using namespace matmul_tiling;
    tiler.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_BF16, false);
    tiler.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_BF16, true);
    tiler.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    tiler.SetBias(false);
    tiler.SetOrgShape(t.rows, 1024, 5120, 5120);
    tiler.SetShape(t.singleM, t.singleN, 5120);
    tiler.SetMatmulConfigParams(mode);
    tiler.SetMadType(MatrixMadType::NORMAL);
    tiler.SetBufferSpace(l1, l0c, -1);
    tiler.SetFixSplit(bm, bn, -1);
    TORCH_CHECK(tiler.GetTiling(t.cube) == 0, "MatmulApiTiling rejected requested candidate");
    TORCH_CHECK(t.cube.baseM == bm && t.cube.baseN == bn && t.cube.baseK > 0,
                "tiler changed fixed baseM/baseN or produced invalid baseK");
    TORCH_CHECK(int64_t(t.cube.baseM) * t.cube.baseN * 4 <= l0c &&
                int64_t(t.cube.baseM) * t.cube.baseK * 2 <= l0a &&
                int64_t(t.cube.baseN) * t.cube.baseK * 2 <= l0b, "base tile exceeds device memory");
    auto host = at::empty({int64_t(sizeof(t))}, at::TensorOptions().dtype(at::kByte));
    std::memcpy(host.data_ptr(), &t, sizeof(t));
    auto device = host.to(x.device());
    const int64_t wsBytes = platform->GetLibApiWorkSpaceSize();
    auto ws = at::empty({std::max<int64_t>(wsBytes, 1)}, x.options().dtype(at::kByte));
    std::ostringstream info;
    info << "{\"baseM\":" << t.cube.baseM << ",\"baseN\":" << t.cube.baseN
         << ",\"baseK\":" << t.cube.baseK << ",\"singleM\":" << t.singleM
         << ",\"singleN\":" << t.singleN << ",\"tasks\":" << t.tasks
         << ",\"blocks\":" << t.blocks << ",\"workspace_bytes\":" << wsBytes
         << ",\"weight_layout_code\":" << t.layout << ",\"ub_bytes\":" << ub
         << ",\"l1_bytes\":" << l1 << ",\"l0a_bytes\":" << l0a
         << ",\"l0b_bytes\":" << l0b << ",\"l0c_bytes\":" << l0c << ",\"tcube_words\":[";
    const auto* words = reinterpret_cast<const int32_t*>(&t.cube);
    for (size_t i = 0; i < sizeof(t.cube) / 4; ++i) { if (i) info << ','; info << words[i]; }
    info << "]}";
    return {device, ws, t.blocks, info.str()};
}

at::Tensor run_out(const at::Tensor& x, const at::Tensor& w, at::Tensor pair,
                   const at::Tensor& tiling, const at::Tensor& ws,
                   int64_t blocks, int64_t mode) {
    matrix(x, kH); weight(w, 0); matrix(pair, 1024, at::kFloat);
    TORCH_CHECK(pair.size(0) == x.size(0), "pair rows must equal T");
    for (const auto& a : {w, pair, tiling, ws}) {
        TORCH_CHECK(a.device() == x.device(), "device mismatch");
    }
    TORCH_CHECK(tiling.scalar_type() == at::kByte && tiling.numel() == sizeof(C2Bf16GemmTiling) &&
                tiling.is_contiguous() && ws.scalar_type() == at::kByte && ws.is_contiguous(),
                "invalid prepared buffers");
    TORCH_CHECK(blocks > 0 && blocks <= 24 && (mode == 0 || mode == 1), "invalid launch");
    c10::DeviceGuard guard(x.device());
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto a = static_cast<uint8_t*>(x.data_ptr()), b = static_cast<uint8_t*>(w.data_ptr());
    auto p = static_cast<uint8_t*>(pair.data_ptr());
    auto t = static_cast<uint8_t*>(tiling.data_ptr());
    auto scratch = static_cast<uint8_t*>(ws.data_ptr());
    at_npu::native::OpCommand::RunOpApi("c2_bf16_c_run", [=]() -> int {
        launch_c2_bf16_gemm(static_cast<uint32_t>(blocks), static_cast<int32_t>(mode), 0,
                            stream, a, b, p, scratch, t);
        return 0;
    });
    return pair;
}
} // namespace
TORCH_LIBRARY(c2_a3_bf16_c_only, m) {
    m.def("prepare(Tensor x, Tensor w, int single_m, int single_n, int base_m, int base_n, int mode, int layout, str soc) -> (Tensor, Tensor, int, str)");
    m.def("run_out(Tensor x, Tensor w, Tensor(a!) pair, Tensor tiling, Tensor workspace, int blocks, int mode) -> Tensor(a!)");
}
TORCH_LIBRARY_IMPL(c2_a3_bf16_c_only, PrivateUse1, m) {
    m.impl("prepare", &prepare);
    m.impl("run_out", &run_out);
}
