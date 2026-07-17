"""
Unit tests for SparseKvGather — AscendC operator.

Interface (matching gather_kv_triton semantics):
  out_ctkv, out_kpe = npu_sparse_kv_gather(
      sparse_indices       [N, K]   or [B, S1, K],
      key_nope             [blocks, blockSize, 1, 512],
      key_rope             [blocks, blockSize, 1,  64],
      block_table          [N, maxBlocks],
      actual_seq_lengths_q [B]      (optional),
      actual_seq_lengths_kv [N]     (optional, per-query),
      cur_pos              [N]      (optional),
      sparse_block_size    int      (must be 1),
      layout_query         "BSND" or "TND",
      layout_kv            "PA_BSND",
  )
  → out_ctkv [N, K, 512]
  → out_kpe  [N, K,  64]
"""

import pytest
import torch
import torch_npu  # noqa: F401


# ====================== Reference impl (CPU) ======================

def _ref_gather_pa(
    sparse_indices: torch.Tensor,  # [N, K] int32
    key_nope: torch.Tensor,        # [blocks, blockSize, 1, 512]
    key_rope: torch.Tensor,        # [blocks, blockSize, 1,  64]
    block_table: torch.Tensor,     # [N, maxBlocks]  (per-query)
    cur_pos: torch.Tensor | None = None,
    act_len_kv: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    N, K = sparse_indices.shape
    block_size = key_nope.shape[1]
    max_blocks = block_table.shape[1]

    out_ctkv = torch.zeros(N, K, 512, dtype=key_nope.dtype)
    out_kpe  = torch.zeros(N, K,  64, dtype=key_rope.dtype)

    for q in range(N):
        cp = cur_pos[q].item() if cur_pos is not None else -1
        kv_len = (act_len_kv[q].item() if act_len_kv is not None
                  else max_blocks * block_size)
        for s in range(K):
            idx = sparse_indices[q, s].item()
            if idx < 0 or idx == cp:
                continue  # leave as zero
            if idx >= kv_len:
                continue

            blk_id = block_table[q, idx // block_size].item()
            blk_off = idx % block_size
            out_ctkv[q, s, :] = key_nope[blk_id, blk_off, 0, :]
            out_kpe[q, s, :]  = key_rope[blk_id, blk_off, 0, :]

    return out_ctkv, out_kpe


# ====================== Helpers ======================

def _make_pa_tensors(N=2, K=4, block_size=128, device="npu"):
    """Create PA tensors for 2D [N, K] Triton-compatible interface.

    block_table has shape [N, maxBlocks] — one row per query.
    Each query row gets its own dedicated set of physical pages.
    """
    torch.manual_seed(42)
    max_blocks = 8
    total_blocks = N * max_blocks  # each query row gets max_blocks pages

    key_nope = torch.randn(total_blocks, block_size, 1, 512,
                           dtype=torch.float16, device=device)
    key_rope = torch.randn(total_blocks, block_size, 1, 64,
                           dtype=torch.float16, device=device)

    block_table = torch.zeros(N, max_blocks, dtype=torch.int32, device=device)
    for q in range(N):
        block_table[q] = torch.arange(q * max_blocks, (q + 1) * max_blocks,
                                      device=device)

    max_valid = max_blocks * block_size
    sparse = torch.randint(0, max_valid, (N, K), dtype=torch.int32, device=device)
    sparse[0, -1] = -1  # last slot of first query invalid

    act_len_kv = torch.full((N,), max_valid, dtype=torch.int32, device=device)

    return sparse, key_nope, key_rope, block_table, act_len_kv


# ====================== CPU tests (always runnable) ======================

class TestReferenceImpl:
    def test_basic(self):
        N, K = 1, 4; bs = 128
        sparse, kn, kr, bt, al = _make_pa_tensors(N, K, bs, device="cpu")
        ct, kp = _ref_gather_pa(sparse, kn, kr, bt, act_len_kv=al)
        assert ct.shape == (N, K, 512)
        assert kp.shape == (N, K, 64)

    def test_block_boundary(self):
        """Token at boundary of two pages → correct physical address."""
        bs = 128; total = 2
        kn = torch.randn(total, bs, 1, 512, dtype=torch.float16)
        kr = torch.randn(total, bs, 1, 64,  dtype=torch.float16)
        bt = torch.tensor([[0, 1]], dtype=torch.int32)  # N=1, maxBlocks=2
        # token 127 → page 0 offset 127; token 128 → page 1 offset 0
        sp = torch.tensor([[127, 128]], dtype=torch.int32)

        ct, _ = _ref_gather_pa(sp, kn, kr, bt)
        torch.testing.assert_close(ct[0, 0], kn[0, 127, 0, :])
        torch.testing.assert_close(ct[0, 1], kn[1,   0, 0, :])

    def test_cur_pos_masking(self):
        """token at cur_pos → zero output."""
        bs = 128; total = 4
        kn = torch.randn(total, bs, 1, 512, dtype=torch.float16)
        kr = torch.randn(total, bs, 1, 64,  dtype=torch.float16)
        bt = torch.arange(total, dtype=torch.int32).view(1, -1)
        sp = torch.tensor([[0, 100, 200]], dtype=torch.int32)
        cp = torch.tensor([100], dtype=torch.int32)  # mask slot 1

        ct, _ = _ref_gather_pa(sp, kn, kr, bt, cur_pos=cp)
        assert not torch.all(ct[0, 0] == 0)  # slot 0 → valid
        assert torch.all(ct[0, 1] == 0)       # slot 1 → masked
        assert not torch.all(ct[0, 2] == 0)  # slot 2 → valid

    def test_invalid_index(self):
        """index == -1 → zero output."""
        bs = 128; total = 4
        kn = torch.randn(total, bs, 1, 512, dtype=torch.float16)
        kr = torch.randn(total, bs, 1, 64,  dtype=torch.float16)
        bt = torch.arange(total, dtype=torch.int32).view(1, -1)
        sp = torch.tensor([[-1, 0, -1]], dtype=torch.int32)

        ct, _ = _ref_gather_pa(sp, kn, kr, bt)
        assert torch.all(ct[0, 0] == 0)
        assert not torch.all(ct[0, 1] == 0)
        assert torch.all(ct[0, 2] == 0)


# ====================== NPU tests ======================

@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
class TestSparseKvGatherNPU:
    def test_basic_pa(self):
        N, K = 2, 4
        sparse, kn, kr, bt, al = _make_pa_tensors(N, K, device="npu")

        ct, kp = torch.ops._C_ascend.npu_sparse_kv_gather(
            sparse, kn, kr, bt, None, al, None,
            1, "BSND", "PA_BSND",
        )
        assert ct.shape == (N, K, 512)
        assert kp.shape == (N, K, 64)
        assert ct.dtype == kn.dtype
        assert kp.dtype == kr.dtype

    def test_vs_reference(self):
        N, K = 1, 4
        sparse, kn, kr, bt, al = _make_pa_tensors(N, K, device="npu")

        ct_npu, kp_npu = torch.ops._C_ascend.npu_sparse_kv_gather(
            sparse, kn, kr, bt, None, al, None,
            1, "BSND", "PA_BSND",
        )
        ct_ref, kp_ref = _ref_gather_pa(
            sparse.cpu(), kn.cpu(), kr.cpu(),
            bt.cpu(), act_len_kv=al.cpu(),
        )
        torch.testing.assert_close(ct_npu.cpu(), ct_ref, rtol=0, atol=0)
        torch.testing.assert_close(kp_npu.cpu(), kp_ref, rtol=0, atol=0)

    def test_cur_pos_masking_npu(self):
        N, K = 1, 4
        sparse, kn, kr, bt, al = _make_pa_tensors(N, K, device="npu")
        # Mask the token at slot 1 by setting cur_pos to its value.
        cur_pos = sparse[0, 1].clone()

        ct, kp = torch.ops._C_ascend.npu_sparse_kv_gather(
            sparse, kn, kr, bt, None, al, cur_pos,
            1, "BSND", "PA_BSND",
        )
        # Masked slot — both CTKV and KPE must be all zero.
        assert torch.all(ct[0, 1] == 0), "cur_pos-masked ctkv slot must be zero"
        assert torch.all(kp[0, 1] == 0), "cur_pos-masked kpe slot must be zero"
        # Unmasked slot 0 must have valid (non-zero) data.
        assert not torch.all(ct[0, 0] == 0), "unmasked ctkv slot must be non-zero"

    def test_all_invalid(self):
        N, K = 1, 4
        sparse, kn, kr, bt, al = _make_pa_tensors(N, K, device="npu")
        sparse[:, :] = -1

        ct, kp = torch.ops._C_ascend.npu_sparse_kv_gather(
            sparse, kn, kr, bt, None, al, None,
            1, "BSND", "PA_BSND",
        )
        assert torch.all(ct == 0)
        assert torch.all(kp == 0)

    def test_dtype_bf16(self):
        N, K = 1, 4
        sparse, kn, kr, bt, al = _make_pa_tensors(N, K, device="npu")
        kn = kn.to(torch.bfloat16)
        kr = kr.to(torch.bfloat16)

        ct, kp = torch.ops._C_ascend.npu_sparse_kv_gather(
            sparse, kn, kr, bt, None, al, None,
            1, "BSND", "PA_BSND",
        )
        assert ct.dtype == torch.bfloat16
        assert kp.dtype == torch.bfloat16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
