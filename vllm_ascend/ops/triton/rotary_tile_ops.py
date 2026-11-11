"""Raw loaded-pair rotary arithmetic shared across normalization and RoPE.

Coordinates, pair extraction/insertion, dtype conversions, output stores and
launch policy stay with callers. This helper has no product-rounding mode;
QSA's explicit BF16 rounding barriers remain in its caller.
"""

from vllm.triton_utils import triton


@triton.jit
def rotate_pair(first, second, cos, sin):
    """Rotate an already-loaded pair with the original ordered raw products."""
    return first * cos - second * sin, second * cos + first * sin
