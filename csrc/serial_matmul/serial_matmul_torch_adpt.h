// SPDX-License-Identifier: Apache-2.0
// Generic CANN 9.2 L0 signatures verified against ops-nn v9.2.0,
// 814ce56c09d6bac1a99364a71a421ef7c1abf519, matmul/common/op_host/op_api.
// This adapter selects V3 explicitly; matrix arithmetic remains in CANN.
#pragma once
#include <memory>
#include <torch/extension.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>
#include <acl/acl.h>
#include "serial_matmul_opdev.h"
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>

namespace vllm_ascend {
inline std::shared_ptr<aclTensor> serial_nd_descriptor(const at::Tensor& tensor)
{
    const int64_t storageElements = tensor.storage().nbytes() / tensor.element_size();
    auto* descriptor = aclCreateTensor(tensor.sizes().data(), tensor.dim(), ACL_BF16,
        tensor.strides().data(), tensor.storage_offset(), ACL_FORMAT_ND,
        &storageElements, 1, const_cast<void*>(tensor.storage().data()));
    TORCH_CHECK(descriptor, "cannot create serial V3 ND descriptor");
    return std::shared_ptr<aclTensor>(descriptor, [](aclTensor* p) { aclDestroyTensor(p); });
}

inline void serial_matmul_v3_out(const at::Tensor& a, const at::Tensor& b,
    bool transposeB, bool batched, at::Tensor& out)
{
    TORCH_CHECK(a.device().type() == c10::DeviceType::PrivateUse1, "serial V3 requires NPU");
    const c10_npu::NPUGuard guard(a.device());
    const at::Tensor* tensors[] = {&a, &b, &out};
    for (const at::Tensor* value : tensors) {
        TORCH_CHECK(value->scalar_type() == at::kBFloat16 && value->device() == a.device()
            && value->is_contiguous() && at_npu::native::custom_ops::get_npu_format(*value) == ACL_FORMAT_ND,
            "serial V3 requires contiguous BF16 ND on one NPU");
    }
    const int64_t rank = batched ? 3 : 2;
    TORCH_CHECK(a.dim() == rank && b.dim() == rank && out.dim() == rank,
        "serial V3 requires matching matrix ranks");
    const int64_t k = b.size(transposeB ? -1 : -2);
    const int64_t n = b.size(transposeB ? -2 : -1);
    TORCH_CHECK(a.size(-1) == k && out.size(-2) == a.size(-2) && out.size(-1) == n,
        "serial V3 matrix/output shapes differ");
    TORCH_CHECK(!batched || (a.size(0) == b.size(0) && out.size(0) == a.size(0)),
        "serial V3 batch broadcasting is unsupported");
    auto da = serial_nd_descriptor(a), db = serial_nd_descriptor(b), dy = serial_nd_descriptor(out);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto status = aclnnSerialMatmulV3GetWorkspaceSize(da.get(), db.get(), transposeB, batched,
        dy.get(), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "serial V3 setup failed: ", status, ": ", aclGetRecentErrMsg());
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, a.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    // Retain descriptors and workspace until the queued host handler submits work.
    auto call = [da, db, dy, workspace, workspaceSize, executor, stream]() -> int {
        auto result = aclnnSerialMatmulV3(workspaceSize ? workspace.data_ptr() : nullptr,
            workspaceSize, executor, stream);
        TORCH_CHECK(result == ACL_SUCCESS, "serial V3 execution failed: ", result,
            ": ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand command;
    command.Name("aclnnSerialMatmulV3").SetCustomHandler(call).Run();
}
} // namespace vllm_ascend
