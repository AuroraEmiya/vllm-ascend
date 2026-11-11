"""Row-quantization arithmetic without scale selection or memory effects."""

from vllm.triton_utils import tl, triton


@triton.jit
def line_absmax(values, AXIS: tl.constexpr):
    """Reduce the caller's already-converted row tile at its explicit axis."""
    return tl.max(tl.abs(values), axis=AXIS)


@triton.jit
def round_clamp_int8(normalized, QUANT_MAX: tl.constexpr):
    """Round to nearest even, clamp symmetrically, then convert to INT8.

    Callers retain div_rn and the distinct query-rounded/key-FP32 scale policy.
    This is not the activation's input-dtype saturating quantization path.
    """
    rounded = tl.extra.cann.libdevice.nearbyint(normalized)
    return tl.minimum(tl.maximum(rounded, -QUANT_MAX), QUANT_MAX).to(tl.int8)
