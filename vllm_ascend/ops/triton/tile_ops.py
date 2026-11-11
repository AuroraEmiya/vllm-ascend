"""Contract-bearing tile operations shared by Triton operator families.

Callers keep coordinates, loaded-value dtypes, masks, memory effects and launch
configuration. A reduction in this module is program-local; automatic tiling
analysis must account for its callee visibility separately from compilation.
"""

from vllm.triton_utils import tl, triton


@triton.jit
def square_sum_tile(values, AXIS: tl.constexpr):
    """Square a loaded tile and sum its original axis without dtype promotion.

    The caller supplies already-masked values and owns any FP32 conversion,
    division, epsilon, inverse square root and row broadcasting. In particular,
    qRMS retains its input-dtype square path, while L2/Engram pass FP32 values.
    """
    square = values * values
    return tl.sum(square, axis=AXIS)
