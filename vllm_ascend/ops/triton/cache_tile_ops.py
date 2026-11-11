"""Pure integer cache-address decomposition with caller-owned validity."""

from vllm.triton_utils import tl, triton


@triton.jit
def slot_to_page_row(nonnegative_slot, PAGE_SIZE: tl.constexpr):
    """Split a nonnegative logical slot; the caller supplies positive page size.

    Validity, compression parity, padding sentinels and physical strides stay
    with callers. This operation does not touch or synchronize cache memory.
    """
    return nonnegative_slot // PAGE_SIZE, nonnegative_slot % PAGE_SIZE
