# SPDX-License-Identifier: Apache-2.0
"""Matrix primitive shared by the two explicit static/SuperKernel pipelines.

The registered extension calls the installed CANN 9.2 L0 Matmul implementation.
This module contains no matrix arithmetic and no Graph compilation policy.
"""

import torch


def serial_matmul_v3(a, b, *, transpose_b=False):
    """BF16 ND/NZ-weight matrix or ND equal-batch matrix product via the generic CANN V3 L0."""
    n = b.shape[-2] if transpose_b else b.shape[-1]
    output = a.new_empty((*a.shape[:-1], n))
    torch.ops._C_ascend.serial_matmul_v3_out(a, b, transpose_b, a.ndim == 3, output)
    return output
