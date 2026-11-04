// SPDX-License-Identifier: Apache-2.0
// CANN opdev uses its own libstdc++ ABI; no Torch headers in this TU.
#include "opdev.h"
#include <cstdio>
#include <opdev/make_op_executor.h>
#include <opdev/op_dfx.h>
#include <opdev/common_types.h>
#include <aclnn_kernels/contiguous.h>

namespace l0op {
const aclTensor* MatMulV3Nd(const aclTensor*, const aclTensor*, const aclTensor*,
    bool, bool, bool, int64_t, aclOpExecutor*);
const aclTensor* BatchMatMulV3Nd(const aclTensor*, const aclTensor*, const aclTensor*,
    const aclTensor*, bool, bool, bool, bool, aclOpExecutor*);
}

namespace {
void PrintMetadata(const char* name, const aclTensor* tensor)
{
    std::fprintf(stderr, "SERIAL_V3_METADATA %s formats(original,view,storage)=%d,%d,%d", name,
        static_cast<int>(tensor->GetOriginalFormat()), static_cast<int>(tensor->GetViewFormat()),
        static_cast<int>(tensor->GetStorageFormat()));
    const op::Shape* shapes[] = {&tensor->GetOriginalShape(), &tensor->GetViewShape(), &tensor->GetStorageShape()};
    for (const auto* shape : shapes) {
        std::fprintf(stderr, " [");
        for (size_t i = 0; i < shape->GetDimNum(); ++i)
            std::fprintf(stderr, "%s%ld", i ? "," : "", static_cast<long>((*shape)[i]));
        std::fprintf(stderr, "]");
    }
    std::fprintf(stderr, "\n");
}
}

extern "C" aclnnStatus aclnnSerialMatmulV3GetWorkspaceSize(const aclTensor* a, aclTensor* b,
    bool transposeB, bool batched, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnSerialMatmulV3, DFX_IN(a, b, transposeB, batched), DFX_OUT(out));
    auto owner = CREATE_EXECUTOR();
    if (!owner.get()) return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    const aclTensor* inputB = b;
    // aclCreateTensor(NZ) initializes original shape from the physical NZ
    // storage. Our descriptor is private to this adapter: specify its logical
    // ND origin/view without changing the physical storage shape or data.
    if (!batched && b->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ) {
        b->SetOriginalShape(b->GetViewShape());
        b->SetOriginalFormat(op::Format::FORMAT_ND);
        b->SetViewFormat(op::Format::FORMAT_ND);
    }
    const aclTensor* value = batched
        ? l0op::BatchMatMulV3Nd(a, b, nullptr, nullptr, false, transposeB, false, false, owner.get())
        : l0op::MatMulV3Nd(a, b, nullptr, false, transposeB, false, int64_t{1}, owner.get());
    if (!value) {
        PrintMetadata("A", a);
        PrintMetadata("B-input", inputB);
        PrintMetadata("B-logical-ND-physical-NZ", b);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (!l0op::ViewCopy(value, out, owner.get())) return ACLNN_ERR_INNER_NULLPTR;
    *workspaceSize = owner->GetWorkspaceSize();
    owner.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnSerialMatmulV3(void* workspace, uint64_t size, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSerialMatmulV3);
    return CommonOpExecutorRun(workspace, size, executor, stream);
}

