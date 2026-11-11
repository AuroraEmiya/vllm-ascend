# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Finite loaded-tile families; callers retain memory, reductions and casts."""

from vllm.triton_utils import tl, triton


@triton.jit
def gate_factor(value, ACTIVATION: tl.constexpr):
    """FP32 caller value: 0=SiLU, 1=sigmoid, 2=signed-sqrt sigmoid.

    The signed-sqrt mode preserves the Engram sign-bit test, including -0.
    Masking and multiplying the returned factor stay with the caller.
    """
    sigmoid_input = value
    if ACTIVATION == 2:
        magnitude = tl.sqrt(tl.maximum(tl.abs(value), 1e-6))
        negative = (value.to(tl.int32, bitcast=True) & -2147483648) != 0
        sigmoid_input = tl.where(negative, -magnitude, magnitude)
    factor = tl.sigmoid(sigmoid_input)
    if ACTIVATION == 0:
        factor = value * factor
    return factor


@triton.jit
def normalize_tile(values, statistic, epsilon, RETURN_SCALE: tl.constexpr = False):
    """Normalize a loaded tile with the caller's expanded statistic.

    Squaring, reduction, broadcasting and any /N stay in the caller. Values
    retain their original dtype until multiplied by the computed FP32 scale.
    """
    scale = tl.rsqrt(statistic + epsilon)
    normalized = values * scale
    if RETURN_SCALE:
        return normalized, scale
    else:
        return normalized


@triton.jit
def rotate_pair_tile(first, second, cosine, sine):
    """Ordinary loaded-pair RoPE; caller owns casts, slices and stores.

    QSA's BF16-per-product rounding deliberately stays outside this family.
    """
    rotated_first = first * cosine - second * sine
    rotated_second = second * cosine + first * sine
    return rotated_first, rotated_second


@triton.jit
def decay_gate_tile(value, decay, beta, threshold, KDA: tl.constexpr):
    """Softplus decay gate with each real caller's comparison/expression policy.

    GDN uses beta*value and <=; KDA uses value*beta and >. Neither mode
    substitutes a numerically different stable-softplus implementation.
    """
    if KDA:
        scaled = value * beta
        use_linear = scaled > threshold
        softplus = tl.where(use_linear, value, (1.0 / beta) * tl.log(1.0 + tl.exp(scaled)))
    else:
        softplus = tl.where(beta * value <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * value)), value)
    return decay * softplus


@triton.jit
def quantize_row_tile(
    values, abs_max, QUANT_MAX: tl.constexpr, QUERY: tl.constexpr, MIN_SCALE: tl.constexpr = 2.0**-24
):
    """Indexer row quantization; caller retains absmax and destination layout.

    Query rounds scale through FP16 before division. Cache K divides by the
    original FP32 scale (or 1 for zero rows). These are distinct finite modes.
    """
    scale = tl.div_rn(abs_max, QUANT_MAX)
    if QUERY:
        scale = scale.to(tl.float16).to(tl.float32)
        scale = tl.maximum(scale, MIN_SCALE)
        divisor = scale
    else:
        divisor = tl.where(scale > 0, scale, 1.0)
    normalized = tl.div_rn(values, divisor[:, None])
    quantized = tl.extra.cann.libdevice.nearbyint(normalized)
    quantized = tl.minimum(tl.maximum(quantized, -QUANT_MAX), QUANT_MAX).to(tl.int8)
    return quantized, scale


@triton.jit
def cache_address_tile(physical, valid, PAGE_SIZE: tl.constexpr, MASK_INVALID: tl.constexpr):
    """Flat/page/row address tuple, with optional original invalid-slot policy.

    Slot eligibility, compression/parity and carrier masks stay in callers.
    The unmasked mode is used only for the cache writer's nonnegative leader.
    """
    if MASK_INVALID:
        safe_physical = tl.maximum(physical, 0)
        flat = tl.where(valid, physical, -1)
        page = tl.where(valid, safe_physical // PAGE_SIZE, -1)
        row = tl.where(valid, safe_physical % PAGE_SIZE, -1)
    else:
        flat = physical
        page = physical // PAGE_SIZE
        row = physical % PAGE_SIZE
    return flat, page, row


@triton.jit
def transfer_tile(source, destination, read_mask, write_mask, ZERO_OTHER: tl.constexpr):
    """Concrete GM tile transfer; no layout inference or program communication.

    Caller pointer plans own strides, logical fragments, tile ownership and
    masks. ZERO_OTHER preserves whether the original masked load specified 0.
    """
    if ZERO_OTHER:
        values = tl.load(source, mask=read_mask, other=0)
    else:
        values = tl.load(source, mask=read_mask)
    tl.store(destination, values, mask=write_mask)


@triton.jit
def packed_mask_tile(
    values, lane_bits, BLOCK_SIZE: tl.constexpr, WORD_BITS: tl.constexpr, PACK: tl.constexpr
):
    """Finite little-endian pack/unpack tile plan; caller retains reductions.

    PACK returns weighted int32 lanes before the original sum/uint8 cast.
    UNPACK returns a blocked-lane predicate before the original flatten/store.
    """
    if PACK:
        lanes = tl.trans(tl.reshape(values, (BLOCK_SIZE // WORD_BITS, WORD_BITS)))
        return lanes * lane_bits
    else:
        return (values[:, None] & lane_bits[None, :]) == 0
