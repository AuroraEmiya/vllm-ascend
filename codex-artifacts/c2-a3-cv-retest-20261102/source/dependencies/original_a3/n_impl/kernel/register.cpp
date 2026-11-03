// Torch registration for the cv60n operator: prepare (host tiling for the
// packed GEMM) and run (one dispatch -> GEMM kernel + fused V kernel).
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "cv60n_tiling.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <sstream>

extern "C" void launch_cv60n_gemm(uint32_t, int32_t, int32_t, aclrtStream, uint8_t*, uint8_t*,
                                  uint8_t*, uint8_t*, uint8_t*);
extern "C" void launch_cv60n_v(uint32_t, int32_t, void*, uint8_t*, uint8_t*, uint8_t*, uint8_t*,
                               uint8_t*, int64_t, int64_t, int32_t, float, int64_t);

namespace {
constexpr int64_t kH = 5120;
constexpr int64_t kD = 512;

void matrix(const at::Tensor& x, int64_t columns) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1 && x.scalar_type() == at::kFloat,
                "cv60n requires NPU FP32 tensors");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == columns && x.is_contiguous(), "dense ND matrix required");
}
void weight(const at::Tensor& w, int64_t layout) {
    TORCH_CHECK(layout == 0 || layout == 1, "weight layout must be ND=0 or NZ=1");
    TORCH_CHECK(w.device().type() == c10::DeviceType::PrivateUse1 &&
                w.scalar_type() == at::kFloat && w.is_contiguous(), "weight must be contiguous NPU FP32");
    if (layout == 0) {
        TORCH_CHECK(w.dim() == 2 && w.size(0) == 1024 && w.size(1) == 5120, "ND weight must be [1024,5120]");
    } else {
        TORCH_CHECK(w.dim() == 3 && w.size(0) == 640 && w.size(1) == 1024 && w.size(2) == 8,
                    "raw NZ weight must be [640,1024,8]");
    }
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
    Cv60nGemmTiling t{};
    t.rows = x.size(0); t.singleM = std::min<int64_t>(sm, t.rows); t.singleN = sn; t.mode = mode;
    t.layout = layout;
    t.tasks = ((t.rows + t.singleM - 1) / t.singleM) * ((1024 + sn - 1) / sn);
    t.blocks = std::min<int64_t>(platform->GetCoreNumAic(), t.tasks);
    TORCH_CHECK(t.blocks > 0 && l1 && l0a && l0b && l0c, "invalid platform capacities");
    matmul_tiling::MatmulApiTiling tiler(*platform);
    using namespace matmul_tiling;
    tiler.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT, false);
    tiler.SetBType(TPosition::GM, layout == 0 ? CubeFormat::ND : CubeFormat::NZ, DataType::DT_FLOAT, true);
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
                int64_t(t.cube.baseM) * t.cube.baseK * 4 <= l0a &&
                int64_t(t.cube.baseN) * t.cube.baseK * 4 <= l0b, "base tile exceeds device memory");
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

// V-only dispatch: the caller (route "proj":"linear") produced the packed pair
// with the fused single-call projection; this op runs just the fused V kernel.
at::Tensor run_v_out(at::Tensor pair, at::Tensor latent, at::Tensor state,
                     const at::Tensor& metadata, const at::Tensor& gamma,
                     int64_t chunk, double eps, int64_t maxq) {
    matrix(pair, 1024);
    TORCH_CHECK(latent.device().type() == c10::DeviceType::PrivateUse1 &&
                latent.scalar_type() == at::kBFloat16 && latent.is_contiguous() &&
                latent.dim() == 2 && latent.size(1) == kD,
                "latent must be contiguous [T,512] NPU BF16");
    TORCH_CHECK(latent.size(0) == pair.size(0), "latent rows must equal pair rows");
    TORCH_CHECK(state.device().type() == c10::DeviceType::PrivateUse1 &&
                state.scalar_type() == at::kFloat && state.is_contiguous() &&
                state.dim() == 3 && state.size(1) == 32 && state.size(2) == 1024,
                "state must be contiguous [P,32,1024] NPU FP32");
    TORCH_CHECK(metadata.device() == pair.device() && metadata.scalar_type() == at::kInt &&
                metadata.is_contiguous() && metadata.dim() == 2 && metadata.size(0) == 5,
                "metadata must be contiguous [5,B] NPU INT32");
    TORCH_CHECK(gamma.device() == pair.device() && gamma.scalar_type() == at::kBFloat16 &&
                gamma.is_contiguous() && gamma.dim() == 1 && gamma.size(0) == kD,
                "gamma must be contiguous [512] NPU BF16");
    for (const auto& a : {latent, state, metadata, gamma}) {
        TORCH_CHECK(a.device() == pair.device(), "device mismatch");
    }
    TORCH_CHECK(chunk == 8 || chunk == 16, "V tile chunk must be 8 or 16");
    TORCH_CHECK(eps >= 0.0 && maxq >= 0, "invalid scalars");
    const int64_t T = pair.size(0);
    const int64_t P = state.size(0);
    const int32_t B = static_cast<int32_t>(metadata.size(1));
    c10::DeviceGuard guard(pair.device());
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto p = static_cast<uint8_t*>(pair.data_ptr());
    auto s = static_cast<uint8_t*>(state.data_ptr());
    auto m = static_cast<uint8_t*>(metadata.data_ptr());
    auto g = static_cast<uint8_t*>(gamma.data_ptr());
    auto o = static_cast<uint8_t*>(latent.data_ptr());
    const float epsF = static_cast<float>(eps);
    at_npu::native::OpCommand::RunOpApi("cv60n_v_run", [=]() -> int {
        launch_cv60n_v(48, static_cast<int32_t>(chunk), stream, p, s, m, g, o,
                       T, P, B, epsF, maxq);
        return 0;
    });
    return latent;
}

at::Tensor run_out(const at::Tensor& x, const at::Tensor& w, at::Tensor pair,
                   at::Tensor latent, at::Tensor state, const at::Tensor& metadata,
                   const at::Tensor& gamma, const at::Tensor& tiling, const at::Tensor& ws,
                   int64_t blocks, int64_t mode, int64_t layout, int64_t chunk,
                   double eps, int64_t maxq) {
    matrix(x, kH); weight(w, layout); matrix(pair, 1024);
    TORCH_CHECK(pair.size(0) == x.size(0), "pair rows must equal T");
    TORCH_CHECK(latent.device().type() == c10::DeviceType::PrivateUse1 &&
                latent.scalar_type() == at::kBFloat16 && latent.is_contiguous() &&
                latent.dim() == 2 && latent.size(0) == x.size(0) && latent.size(1) == kD,
                "latent must be contiguous [T,512] NPU BF16");
    TORCH_CHECK(state.device().type() == c10::DeviceType::PrivateUse1 &&
                state.scalar_type() == at::kFloat && state.is_contiguous() &&
                state.dim() == 3 && state.size(1) == 32 && state.size(2) == 1024,
                "state must be contiguous [P,32,1024] NPU FP32");
    TORCH_CHECK(metadata.device() == x.device() && metadata.scalar_type() == at::kInt &&
                metadata.is_contiguous() && metadata.dim() == 2 && metadata.size(0) == 5,
                "metadata must be contiguous [5,B] NPU INT32");
    TORCH_CHECK(gamma.device() == x.device() && gamma.scalar_type() == at::kBFloat16 &&
                gamma.is_contiguous() && gamma.dim() == 1 && gamma.size(0) == kD,
                "gamma must be contiguous [512] NPU BF16");
    for (const auto& a : {w, pair, latent, state, metadata, gamma, tiling, ws}) {
        TORCH_CHECK(a.device() == x.device(), "device mismatch");
    }
    TORCH_CHECK(tiling.scalar_type() == at::kByte && tiling.numel() == sizeof(Cv60nGemmTiling) &&
                tiling.is_contiguous() && ws.scalar_type() == at::kByte && ws.is_contiguous(),
                "invalid prepared buffers");
    TORCH_CHECK(blocks > 0 && blocks <= 1024 && (mode == 0 || mode == 1), "invalid launch");
    TORCH_CHECK(chunk == 8 || chunk == 16, "V tile chunk must be 8 or 16");
    TORCH_CHECK(eps >= 0.0 && maxq >= 0, "invalid scalars");
    const int64_t T = x.size(0);
    const int64_t P = state.size(0);
    const int32_t B = static_cast<int32_t>(metadata.size(1));
    c10::DeviceGuard guard(x.device());
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto a = static_cast<uint8_t*>(x.data_ptr()), b = static_cast<uint8_t*>(w.data_ptr());
    auto p = static_cast<uint8_t*>(pair.data_ptr());
    auto s = static_cast<uint8_t*>(state.data_ptr());
    auto m = static_cast<uint8_t*>(metadata.data_ptr());
    auto g = static_cast<uint8_t*>(gamma.data_ptr());
    auto o = static_cast<uint8_t*>(latent.data_ptr());
    auto t = static_cast<uint8_t*>(tiling.data_ptr());
    auto scratch = static_cast<uint8_t*>(ws.data_ptr());
    const int32_t gemmBlocks = static_cast<int32_t>(blocks);
    const float epsF = static_cast<float>(eps);
    at_npu::native::OpCommand::RunOpApi("cv60n_run", [=]() -> int {
        launch_cv60n_gemm(gemmBlocks, static_cast<int32_t>(mode), static_cast<int32_t>(layout),
                          stream, a, b, p, scratch, t);
        launch_cv60n_v(48, static_cast<int32_t>(chunk), stream, p, s, m, g, o,
                       T, P, B, epsF, maxq);
        return 0;
    });
    return latent;
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("cv60n_prepare(Tensor x, Tensor w, int single_m, int single_n, int base_m, int base_n, int mode, int layout, str soc) -> (Tensor, Tensor, int, str)");
    m.def("cv60n_run(Tensor x, Tensor w, Tensor(a!) pair, Tensor(b!) latent, Tensor(c!) state, Tensor metadata, Tensor gamma, Tensor tiling, Tensor workspace, int blocks, int mode, int layout, int chunk, float eps, int maxq) -> Tensor(b!)");
    m.def("cv60n_v_run(Tensor pair, Tensor(b!) latent, Tensor(c!) state, Tensor metadata, Tensor gamma, int chunk, float eps, int maxq) -> Tensor(b!)");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("cv60n_prepare", &prepare);
    m.impl("cv60n_run", &run_out);
    m.impl("cv60n_v_run", &run_v_out);
}
