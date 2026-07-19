"""
Unit tests for SparseKvGather — AscendC operator.

Interface:
  out_ctkv, out_kpe = torch.ops._C_ascend.npu_sparse_kv_gather(
      paged_ctkv   [num_blocks, 128, 1, 512]  fp16/bf16
      paged_kpe    [num_blocks, 128, 1,  64]  fp16/bf16
      block_table  [num_actual, max_blocks]    int32/int64
      topk_indices [num_actual, topk_n]        int32/int64
      cur_pos      [num_actual]                int32/int64
      block_size   int  (must be 128)
  )
  → out_ctkv [num_actual, topk_n, 512]
  → out_kpe  [num_actual, topk_n,  64]

Invalid slots (pos < 0, pos == cur_pos, logical_block >= max_blocks,
physical_block < 0 or >= num_blocks) are zeroed in-place;
valid slots are never compacted.
"""

import pytest
import torch

try:
    import torch_npu  # noqa: F401
    HAS_NPU = torch.npu.is_available()
except ImportError:
    HAS_NPU = False

BLOCK_SIZE = 128
CTKV_DIM = 512
KPE_DIM = 64


# ====================== CPU reference ======================

def _ref_gather(
    paged_ctkv: torch.Tensor,      # [num_blocks, 128, 1, 512]
    paged_kpe: torch.Tensor,       # [num_blocks, 128, 1,  64]
    block_table: torch.Tensor,     # [num_actual, max_blocks] int32/int64
    topk_indices: torch.Tensor,    # [num_actual, topk_n]     int32/int64
    cur_pos: torch.Tensor,         # [num_actual]             int32/int64
) -> tuple[torch.Tensor, torch.Tensor]:
    num_actual, topk_n = topk_indices.shape
    num_blocks = paged_ctkv.shape[0]
    max_blocks = block_table.shape[1]

    out_ctkv = torch.zeros(num_actual, topk_n, CTKV_DIM, dtype=paged_ctkv.dtype)
    out_kpe = torch.zeros(num_actual, topk_n, KPE_DIM, dtype=paged_kpe.dtype)

    for q in range(num_actual):
        cp = int(cur_pos[q].item())
        for s in range(topk_n):
            pos = int(topk_indices[q, s].item())
            if pos < 0 or pos == cp:
                continue
            logical_block = pos // BLOCK_SIZE
            if logical_block >= max_blocks:
                continue
            physical_block = int(block_table[q, logical_block].item())
            if physical_block < 0 or physical_block >= num_blocks:
                continue
            block_offset = pos % BLOCK_SIZE
            out_ctkv[q, s, :] = paged_ctkv[physical_block, block_offset, 0, :]
            out_kpe[q, s, :]  = paged_kpe[physical_block, block_offset, 0, :]

    return out_ctkv, out_kpe


# ====================== Helpers ======================

def _make_pa_tensors(num_actual=2, topk_n=4, num_blocks=16, max_blocks=8,
                     *, dtype=torch.float16, device="cpu", seed=42):
    """Create test tensors with deterministic unique data per token."""
    torch.manual_seed(seed)
    paged_ctkv = torch.randn(num_blocks, BLOCK_SIZE, 1, CTKV_DIM,
                             dtype=dtype, device=device)
    paged_kpe  = torch.randn(num_blocks, BLOCK_SIZE, 1, KPE_DIM,
                             dtype=dtype, device=device)

    # Identity mapping: logical N → physical N
    block_table = torch.zeros(num_actual, max_blocks, dtype=torch.int64, device=device)
    for q in range(num_actual):
        block_table[q] = torch.arange(q * max_blocks, (q + 1) * max_blocks,
                                      device=device)

    max_valid = max_blocks * BLOCK_SIZE
    topk_indices = torch.randint(0, max_valid, (num_actual, topk_n),
                                 dtype=torch.int64, device=device)
    if topk_n >= 2:
        topk_indices[0, -1] = -1  # last slot of first query invalid
    cur_pos = torch.full((num_actual,), -1, dtype=torch.int64, device=device)

    return paged_ctkv, paged_kpe, block_table, topk_indices, cur_pos


# ====================== CPU-only reference tests ======================

class TestReferenceImpl:
    def test_basic_shape(self):
        N, K = 1, 4
        tensors = _make_pa_tensors(N, K, device="cpu")
        ct, kp = _ref_gather(*tensors)
        assert ct.shape == (N, K, CTKV_DIM)
        assert kp.shape == (N, K, KPE_DIM)
        # -1 slot must be all-zero
        assert torch.equal(ct[0, -1], torch.zeros(CTKV_DIM, dtype=ct.dtype))

    def test_block_boundary(self):
        """Token 127→page[0].off[127]; token 128→page[1].off[0]."""
        kn = torch.randn(2, BLOCK_SIZE, 1, CTKV_DIM, dtype=torch.float16)
        kr = torch.randn(2, BLOCK_SIZE, 1, KPE_DIM,  dtype=torch.float16)
        bt = torch.tensor([[0, 1]], dtype=torch.int64)
        sp = torch.tensor([[127, 128]], dtype=torch.int64)
        cp = torch.tensor([-1], dtype=torch.int64)
        ct, _ = _ref_gather(kn, kr, bt, sp, cp)
        assert torch.equal(ct[0, 0], kn[0, 127, 0, :])
        assert torch.equal(ct[0, 1], kn[1,   0, 0, :])

    def test_cur_pos_masking(self):
        """pos == cur_pos → zero."""
        kn = torch.randn(4, BLOCK_SIZE, 1, CTKV_DIM, dtype=torch.float16)
        kr = torch.randn(4, BLOCK_SIZE, 1, KPE_DIM,  dtype=torch.float16)
        bt = torch.arange(4, dtype=torch.int64).view(1, -1)
        sp = torch.tensor([[0, 100, 200]], dtype=torch.int64)
        cp = torch.tensor([100], dtype=torch.int64)
        ct, _ = _ref_gather(kn, kr, bt, sp, cp)
        assert not torch.equal(ct[0, 0], torch.zeros(CTKV_DIM, dtype=ct.dtype))
        assert torch.equal(ct[0, 1], torch.zeros(CTKV_DIM, dtype=ct.dtype))
        assert not torch.equal(ct[0, 2], torch.zeros(CTKV_DIM, dtype=ct.dtype))

    def test_negative_index(self):
        """pos < 0 → zero."""
        kn = torch.randn(4, BLOCK_SIZE, 1, CTKV_DIM, dtype=torch.float16)
        kr = torch.randn(4, BLOCK_SIZE, 1, KPE_DIM,  dtype=torch.float16)
        bt = torch.arange(4, dtype=torch.int64).view(1, -1)
        sp = torch.tensor([[-1, 0, -1]], dtype=torch.int64)
        cp = torch.tensor([-1], dtype=torch.int64)
        ct, _ = _ref_gather(kn, kr, bt, sp, cp)
        assert torch.equal(ct[0, 0], torch.zeros(CTKV_DIM, dtype=ct.dtype))
        assert not torch.equal(ct[0, 1], torch.zeros(CTKV_DIM, dtype=ct.dtype))
        assert torch.equal(ct[0, 2], torch.zeros(CTKV_DIM, dtype=ct.dtype))

    def test_oob_logical_block(self):
        """logical_block >= max_blocks → zero."""
        kn = torch.randn(2, BLOCK_SIZE, 1, CTKV_DIM, dtype=torch.float16)
        kr = torch.randn(2, BLOCK_SIZE, 1, KPE_DIM,  dtype=torch.float16)
        bt = torch.tensor([[0, 1]], dtype=torch.int64)  # max_blocks=2
        sp = torch.tensor([[0, BLOCK_SIZE * 2]], dtype=torch.int64)  # 256 → blk 2 ≥ 2
        cp = torch.tensor([-1], dtype=torch.int64)
        ct, _ = _ref_gather(kn, kr, bt, sp, cp)
        assert not torch.equal(ct[0, 0], torch.zeros(CTKV_DIM, dtype=ct.dtype))
        assert torch.equal(ct[0, 1], torch.zeros(CTKV_DIM, dtype=ct.dtype))

    def test_oob_physical_block(self):
        """physical_block outside [0, num_blocks) → zero."""
        kn = torch.randn(2, BLOCK_SIZE, 1, CTKV_DIM, dtype=torch.float16)
        kr = torch.randn(2, BLOCK_SIZE, 1, KPE_DIM,  dtype=torch.float16)
        bt = torch.tensor([[0, 999]], dtype=torch.int64)  # illegal block
        sp = torch.tensor([[BLOCK_SIZE]], dtype=torch.int64)  # token 128 → blk 1
        cp = torch.tensor([-1], dtype=torch.int64)
        ct, _ = _ref_gather(kn, kr, bt, sp, cp)
        assert torch.equal(ct[0, 0], torch.zeros(CTKV_DIM, dtype=ct.dtype))

    def test_odd_topk_n(self):
        """Odd topk_n: last slot must go through single-slot path."""
        N, K = 1, 3
        tensors = _make_pa_tensors(N, K, num_blocks=4, max_blocks=4, device="cpu")
        ct, kp = _ref_gather(*tensors)
        assert ct.shape == (N, K, CTKV_DIM)
        assert kp.shape == (N, K, KPE_DIM)

    def test_multi_query(self):
        """Each query uses its own block_table[q] and cur_pos[q]."""
        N, K = 3, 4
        tensors = _make_pa_tensors(N, K, num_blocks=24, max_blocks=8, device="cpu")
        paged_ctkv, paged_kpe, block_table, topk_indices, cur_pos = tensors
        # Give each query a distinct cur_pos
        cur_pos[0] = topk_indices[0, 0]
        cur_pos[1] = topk_indices[1, 1]
        ct, kp = _ref_gather(paged_ctkv, paged_kpe, block_table, topk_indices, cur_pos)
        assert ct.shape == (N, K, CTKV_DIM)
        # Query 0 slot 0 must be zero (masked), slot 1 must be non-zero
        assert torch.equal(ct[0, 0], torch.zeros(CTKV_DIM, dtype=ct.dtype))
        assert not torch.equal(ct[0, 1], torch.zeros(CTKV_DIM, dtype=ct.dtype))


# ====================== NPU tests ======================

@pytest.mark.skipif(not HAS_NPU, reason="NPU not available")
class TestSparseKvGatherNPU:
    def _call_op(self, paged_ctkv, paged_kpe, block_table, topk_indices, cur_pos):
        return torch.ops._C_ascend.npu_sparse_kv_gather(
            paged_ctkv, paged_kpe, block_table,
            topk_indices, cur_pos, BLOCK_SIZE,
        )

    def test_vs_reference(self):
        """NPU output vs CPU reference — bitwise exact match."""
        tensors = _make_pa_tensors(2, 8, num_blocks=16, max_blocks=8,
                                   device="npu")
        paged_ctkv, paged_kpe, block_table, topk_indices, cur_pos = tensors

        ct_npu, kp_npu = self._call_op(
            paged_ctkv, paged_kpe, block_table, topk_indices, cur_pos)

        ct_ref, kp_ref = _ref_gather(
            paged_ctkv.cpu(), paged_kpe.cpu(),
            block_table.cpu(), topk_indices.cpu(), cur_pos.cpu(),
        )
        assert torch.equal(ct_npu.cpu(), ct_ref), "ctkv mismatch"
        assert torch.equal(kp_npu.cpu(), kp_ref), "kpe mismatch"

    def test_pair_move_forward(self):
        """Adjacent slots with physical_token1 > physical_token0 and gap ≠ 1.
        Fill every physical token with a unique magic number so wrong-slot
        reads are detectable."""
        num_actual, topk_n, max_blocks, num_blocks = 1, 2, 4, 4
        paged_ctkv = torch.zeros(num_blocks, BLOCK_SIZE, 1, CTKV_DIM,
                                 dtype=torch.float16, device="npu")
        paged_kpe  = torch.zeros(num_blocks, BLOCK_SIZE, 1, KPE_DIM,
                                 dtype=torch.float16, device="npu")
        # Unique marker: row idx in first element
        for t in range(num_blocks * BLOCK_SIZE):
            paged_ctkv[t // BLOCK_SIZE, t % BLOCK_SIZE, 0, 0] = float(t)
            paged_kpe[t // BLOCK_SIZE, t % BLOCK_SIZE, 0, 0]  = float(t)

        bt = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64, device="npu")
        sp = torch.tensor([[10, 200]], dtype=torch.int64, device="npu")  # gap=190
        cp = torch.tensor([-1], dtype=torch.int64, device="npu")

        ct_npu, kp_npu = self._call_op(paged_ctkv, paged_kpe, bt, sp, cp)
        # Slot 0 → token 10; Slot 1 → token 200 (page 1 offset 72)
        assert ct_npu[0, 0, 0].item() == 10.0, "slot 0 wrong"
        assert ct_npu[0, 1, 0].item() == 200.0, "slot 1 wrong"

    def test_pair_fallback(self):
        """physical_token1 < physical_token0 → single-slot fallback, order kept."""
        num_actual, topk_n, max_blocks, num_blocks = 1, 2, 4, 4
        paged_ctkv = torch.zeros(num_blocks, BLOCK_SIZE, 1, CTKV_DIM,
                                 dtype=torch.float16, device="npu")
        for t in range(num_blocks * BLOCK_SIZE):
            paged_ctkv[t // BLOCK_SIZE, t % BLOCK_SIZE, 0, 0] = float(t)

        paged_kpe = torch.zeros(num_blocks, BLOCK_SIZE, 1, KPE_DIM,
                                dtype=torch.float16, device="npu")
        bt = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64, device="npu")
        sp = torch.tensor([[200, 10]], dtype=torch.int64, device="npu")  # descending
        cp = torch.tensor([-1], dtype=torch.int64, device="npu")

        ct_npu, _ = self._call_op(paged_ctkv, paged_kpe, bt, sp, cp)
        # top-k order preserved despite physical order
        assert ct_npu[0, 0, 0].item() == 200.0
        assert ct_npu[0, 1, 0].item() == 10.0

    def test_invalid_cur_pos(self):
        """pos == cur_pos → zero."""
        tensors = _make_pa_tensors(1, 4, num_blocks=4, max_blocks=4, device="npu",
                                   dtype=torch.float16)
        paged_ctkv, paged_kpe, block_table, topk_indices, cur_pos = tensors
        cur_pos[0] = topk_indices[0, 1]  # mask slot 1

        ct_npu, kp_npu = self._call_op(
            paged_ctkv, paged_kpe, block_table, topk_indices, cur_pos)
        assert torch.equal(ct_npu[0, 1], torch.zeros(CTKV_DIM, dtype=ct_npu.dtype))
        assert torch.equal(kp_npu[0, 1], torch.zeros(KPE_DIM, dtype=kp_npu.dtype))
        # Unmasked slots non-zero
        assert not torch.equal(ct_npu[0, 0], torch.zeros(CTKV_DIM, dtype=ct_npu.dtype))

    def test_all_invalid(self):
        """All topk_indices == -1 → all-zero output."""
        tensors = _make_pa_tensors(1, 4, device="npu", dtype=torch.float16)
        paged_ctkv, paged_kpe, block_table, topk_indices, cur_pos = tensors
        topk_indices[:, :] = -1

        ct_npu, kp_npu = self._call_op(
            paged_ctkv, paged_kpe, block_table, topk_indices, cur_pos)
        assert torch.equal(ct_npu, torch.zeros_like(ct_npu))
        assert torch.equal(kp_npu, torch.zeros_like(kp_npu))

    def test_dtype_bf16(self):
        """Primary business dtype."""
        tensors = _make_pa_tensors(1, 4, num_blocks=4, max_blocks=4, device="npu",
                                   dtype=torch.bfloat16)
        paged_ctkv, paged_kpe, block_table, topk_indices, cur_pos = tensors
        ct_npu, kp_npu = self._call_op(
            paged_ctkv, paged_kpe, block_table, topk_indices, cur_pos)
        assert ct_npu.dtype == torch.bfloat16
        assert kp_npu.dtype == torch.bfloat16
        ct_ref, kp_ref = _ref_gather(
            paged_ctkv.cpu(), paged_kpe.cpu(),
            block_table.cpu(), topk_indices.cpu(), cur_pos.cpu(),
        )
        assert torch.equal(ct_npu.cpu(), ct_ref)

    def test_dtype_fp16(self):
        """FP16 supplemental."""
        tensors = _make_pa_tensors(1, 4, device="npu", dtype=torch.float16)
        paged_ctkv, paged_kpe, block_table, topk_indices, cur_pos = tensors
        ct_npu, _ = self._call_op(
            paged_ctkv, paged_kpe, block_table, topk_indices, cur_pos)
        assert ct_npu.dtype == torch.float16

    def test_int64_indices(self):
        """INT64 block_table and topk_indices."""
        tensors = _make_pa_tensors(1, 4, device="npu", dtype=torch.float16)
        paged_ctkv, paged_kpe, block_table, topk_indices, cur_pos = tensors
        # All already int64; verify they flow through correctly
        ct_npu, _ = self._call_op(
            paged_ctkv, paged_kpe, block_table, topk_indices, cur_pos)
        ct_ref, _ = _ref_gather(
            paged_ctkv.cpu(), paged_kpe.cpu(),
            block_table.cpu(), topk_indices.cpu(), cur_pos.cpu(),
        )
        assert torch.equal(ct_npu.cpu(), ct_ref)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
