// SPDX-License-Identifier: Apache-2.0
// CANN opdev uses its own libstdc++ ABI; no Torch headers in this TU.
#include "serial_matmul_opdev.h"
#include <opdev/make_op_executor.h>
#include <opdev/op_dfx.h>
#include <opdev/common_types.h>
#include <aclnn_kernels/contiguous.h>

namespace l0op {
const aclTensor* ReFormat(const aclTensor*, const ge::Format&, aclOpExecutor*);
const aclTensor* MatMulV3Nd(const aclTensor*, const aclTensor*, const aclTensor*,
    bool, bool, bool, int64_t, aclOpExecutor*);
const aclTensor* BatchMatMulV3Nd(const aclTensor*, const aclTensor*, const aclTensor*,
    const aclTensor*, bool, bool, bool, bool, aclOpExecutor*);
}

extern "C" aclnnStatus aclnnSerialMatmulV3GetWorkspaceSize(const aclTensor* a, const aclTensor* b,
    bool transposeB, bool batched, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnSerialMatmulV3, DFX_IN(a, b, transposeB, batched), DFX_OUT(out));
    auto owner = CREATE_EXECUTOR();
    if (!owner.get()) return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    // The public matmul front-end reinterprets NZ as an NZ view before L0.
    // Keep its physical layout and dtype; ReFormat is descriptor-only.
    if (!batched && b->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ) {
        b = l0op::ReFormat(b, op::Format::FORMAT_FRACTAL_NZ, owner.get());
        if (!b) return ACLNN_ERR_INNER_NULLPTR;
    }
    const aclTensor* value = batched
        ? l0op::BatchMatMulV3Nd(a, b, nullptr, nullptr, false, transposeB, false, false, owner.get())
        : l0op::MatMulV3Nd(a, b, nullptr, false, transposeB, false, int64_t{1}, owner.get());
    if (!value || !l0op::ViewCopy(value, out, owner.get())) return ACLNN_ERR_INNER_NULLPTR;
    *workspaceSize = owner->GetWorkspaceSize();
    owner.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnSerialMatmulV3(void* workspace, uint64_t size, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSerialMatmulV3);
    return CommonOpExecutorRun(workspace, size, executor, stream);
}

