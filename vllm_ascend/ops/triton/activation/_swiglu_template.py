#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Compile-time Swiglu tile family, with memory and reductions owned by callers."""

from vllm.triton_utils import tl, triton


@triton.jit
def swiglu_tile(
    gate,
    up,
    SIGMOID_FP32: tl.constexpr,
    CLAMP: tl.constexpr,
    LIMIT: tl.constexpr,
):
    """Compute a loaded tile without changing its caller's dtype policy.

    Quant casts only the sigmoid input; Step passes FP32 gate/up tiles and
    enables its original clamp order. Both choices specialize at compile time.
    """
    sigmoid_input = gate
    if SIGMOID_FP32:
        sigmoid_input = gate.to(tl.float32)
    activated = gate * tl.sigmoid(sigmoid_input)
    if CLAMP:
        activated = tl.minimum(activated, LIMIT)
        up = tl.minimum(up, LIMIT)
        up = tl.maximum(-LIMIT, up)
    return activated * up
