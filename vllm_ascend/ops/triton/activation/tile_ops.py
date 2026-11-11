"""Explicit tile operations used by the SwiGLU activation callers.

These helpers operate on loaded tiles within one program. Callers own tile
coordinates, dtypes, memory accesses, reductions over metadata, and launch
configuration. They do not exchange data between programs.
"""

from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.gate_tile_ops import gated_multiply_tile


@triton.jit
def silu_tile(gate, sigmoid_input):
    """Compute gate * sigmoid(sigmoid_input) in the original expression order.

    Both tiles must describe the same gate values and coordinates; the caller
    may cast sigmoid_input to the dtype required by its numerical contract.
    """
    return gated_multiply_tile(gate, sigmoid_input)


@triton.jit
def step_clamp_pair(silu_gate, up, LIMIT: tl.constexpr):
    """Clamp the SiLU result above and the up tile above then below."""
    silu_gate = tl.minimum(silu_gate, LIMIT)
    up = tl.minimum(up, LIMIT)
    up = tl.maximum(-LIMIT, up)
    return silu_gate, up


@triton.jit
def line_reduce_absmax(tile):
    """Reduce all elements of a rank-one tile, retaining its original dtype.

    This is a program-local reduction, with no rank-two or arbitrary-axis
    contract. A caller using automatic tiling must account for the reduction
    being inside a helper rather than visible in its kernel body.
    """
    return tl.max(tl.abs(tile))


@triton.jit
def quantize_segment(segment, scale, INPUT_DTYPE: tl.constexpr):
    """Normalize in FP32, cast through the input dtype, then saturate to INT8.

    INPUT_DTYPE belongs to the original activation input, which need not equal
    segment.dtype. The caller retains the original scale, including zero.
    """
    segment = (segment.to(tl.float32) / scale).to(INPUT_DTYPE)
    return segment.cast(tl.int8, overflow_mode="saturate")
