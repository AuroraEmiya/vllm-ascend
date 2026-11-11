"""Program-local GM tile transfer with caller-owned pointer and mask plans."""

from vllm.triton_utils import tl, triton


@triton.jit
def copy_tile(source_ptrs, destination_ptrs, load_mask, store_mask, OTHER: tl.constexpr = None):
    """Copy loaded lanes, preserving omitted versus explicit masked-load other.

    This is an ordinary GM load/store pair. It provides neither layout exchange
    across programs nor synchronization; callers own all row/head coordinates.
    """
    if OTHER is None:
        values = tl.load(source_ptrs, mask=load_mask)
    else:
        values = tl.load(source_ptrs, mask=load_mask, other=OTHER)
    tl.store(destination_ptrs, values, mask=store_mask)
