"""Little-endian lane-bit operations for sampling and grammar mask words."""

from vllm.triton_utils import tl, triton


@triton.jit
def pack_bits(keep_i32, BLOCK_SIZE: tl.constexpr, WORD_BITS: tl.constexpr = 8):
    """Pack preconverted INT32 finite predicates into little-endian byte words.

    Caller supplies the original finite predicate and owns count accumulation,
    active-request mapping, padded lanes and stores. Current contract is byte8.
    """
    bits = tl.trans(tl.reshape(keep_i32, (BLOCK_SIZE // WORD_BITS, WORD_BITS)))
    bit_weights = (1 << tl.arange(0, WORD_BITS))[:, None]
    return tl.sum(bits * bit_weights, axis=0).to(tl.uint8)


@triton.jit
def unpack_bits(packed_i32, WORD_BITS: tl.constexpr = 32):
    """Return a blocked-lane matrix: zero bits block, low bit is lane zero.

    Current grammar contract is signed INT32 words, including high bit31.
    The caller retains flattening, active-position predicates and logits store.
    """
    bit_weights = tl.full((WORD_BITS,), 1, tl.int32) << tl.arange(0, WORD_BITS)
    return (packed_i32[:, None] & bit_weights[None, :]) == 0
