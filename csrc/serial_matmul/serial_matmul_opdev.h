// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <acl/acl.h>
#include <aclnn/aclnn_base.h>

extern "C" aclnnStatus aclnnSerialMatmulV3GetWorkspaceSize(
    const aclTensor*, aclTensor*, bool, bool, aclTensor*, uint64_t*, aclOpExecutor**);
extern "C" aclnnStatus aclnnSerialMatmulV3(
    void*, uint64_t, aclOpExecutor*, aclrtStream);
