"""Explicit loaded-value gate operations with finite numerical policies."""

from vllm.triton_utils import tl, triton


@triton.jit
def gated_multiply_tile(values, sigmoid_input, SIGMOID_OUTPUT_DTYPE: tl.constexpr = None):
    """Multiply a sigmoid factor, optionally rounding that factor before use.

    Callers own sigmoid-input conversion. None preserves the raw SiLU factor;
    a supplied dtype preserves flash attention's intermediate sigmoid rounding.
    """
    scale = tl.sigmoid(sigmoid_input)
    if SIGMOID_OUTPUT_DTYPE is not None:
        scale = scale.to(SIGMOID_OUTPUT_DTYPE)
    return values * scale


@triton.jit
def softplus_tile(value, scaled_value, beta, threshold, LINEAR_WHEN_GT: tl.constexpr):
    """Apply the original softplus threshold policy to caller-scaled values.

    The two predicates and literal spellings retain GDN/KDA NaN and constant
    semantics. The caller keeps beta*x versus x*beta and the decay product.
    """
    if LINEAR_WHEN_GT:
        return tl.where(scaled_value > threshold, value, (1.0 / beta) * tl.log(1.0 + tl.exp(scaled_value)))
    else:
        return tl.where(scaled_value <= threshold, (1 / beta) * tl.log(1 + tl.exp(scaled_value)), value)
