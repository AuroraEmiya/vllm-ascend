"""Public adapter over the frozen Ascend C kernels.

Preparation owns buffers and input views. It must run outside graph capture.
Repeated runs reuse outputs: copy them if persistent snapshots are needed.
"""
import torch
import torch_npu  # Registers the NPU backend.

from . import _projection
from .sparse_selection import runner as _selection

compute_qk_scores = _projection.run
select_sparse_indices = _selection.run


class PreparedQuantLightningIndexer:
    """Fixed-shape, fixed-storage invocation suitable for NPU graph capture.

    Update existing input storage in place before replay. Do not replace input
    tensors or change shape, layout, request metadata bounds, or device.
    One instance must not be used concurrently across streams.
    query_capacity remains the maximum chunk size (1024).
    effective_query_capacity is the allocated capacity, conservatively rounded
    to eight rows so packed storage has uniform complete groups. Captured graphs must be rebuilt for new shapes.
    """

    def __init__(self, q, k, w, q_descale, k_descale, *, cu_seqlens_q,
                 seqused_q, seqused_k, block_table,
                 candidate_topk_blocks=-1, candidate_block_size=-1,
                 query_capacity=1024):
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("Prepare outside graph capture")
        if query_capacity != 1024:
            raise ValueError("This release supports query_capacity=1024")
        if (candidate_topk_blocks, candidate_block_size) not in ((-1, -1), (2048, 8)):
            raise ValueError("Candidate configuration must be (-1,-1) or (2048,8)")
        if q.ndim != 3 or q.shape[1:] != (32, 64) or not 0 < q.shape[0] <= 2147483647:
            raise ValueError("Expected nonempty q[T,32,64] within int32 indexing")
        total = q.shape[0]
        if w.shape != (total, 32) or q_descale.shape != (total, 32, 2, 2):
            raise ValueError("w/q_descale must cover every query row")
        if block_table.ndim != 2 or block_table.shape[1] <= 0:
            raise ValueError("Expected nonempty block_table[B,pages]")
        if q.device.type != "npu":
            raise ValueError("NPU tensors required")
        tensor_specs = ((q, torch.uint8), (k, torch.uint8),
                        (q_descale, torch.uint8), (k_descale, torch.uint8),
                        (w, torch.float32), (cu_seqlens_q, torch.int32),
                        (seqused_q, torch.int32), (seqused_k, torch.int32),
                        (block_table, torch.int32))
        for tensor, dtype in tensor_specs:
            if tensor.device != q.device or tensor.dtype != dtype:
                raise ValueError("Input device/dtype mismatch")
        for tensor in (q, q_descale, w, cu_seqlens_q, seqused_q, seqused_k, block_table):
            if not tensor.is_contiguous():
                raise ValueError("Query, weight and metadata inputs must be contiguous")
        sp = ((block_table.shape[1] * 64 + 127) // 128) * 128
        if sp > 32896:
            raise ValueError("Unsupported padded K capacity")
        self.device = q.device
        self.query_capacity = query_capacity
        # Use uniform eight-row allocation groups for both raw layouts.
        # Small inputs have at most seven inactive rows; large chunks stay 1024.
        capacity = min(query_capacity, ((total + 7) // 8) * 8)
        self.effective_query_capacity = capacity
        self._calls = []
        with torch.npu.device(self.device):
            _projection.load()
            _selection.load()
            lengths = {min(query_capacity, total - start)
                       for start in range(0, total, query_capacity)}
            plans = {n: _projection.make_plan(capacity, sp, n, device=q.device)
                     for n in lengths}
            first = min(query_capacity, total)
            self.raw = _projection.allocate_raw(plans[first], device=q.device)
            self.outputs = _selection.allocate_outputs(self.raw, total, candidate_topk_blocks)
            self.workspace = _selection.allocate_workspace(self.raw, capacity,
                                                            candidate_topk_blocks)
            for start in range(0, total, query_capacity):
                n = min(query_capacity, total - start)
                end = start + n
                projection_args = (q[start:end], k, q_descale[start:end], k_descale,
                                   cu_seqlens_q, seqused_q, seqused_k, block_table,
                                   start, n, capacity)
                selection_args = (self.raw, w[start:end], cu_seqlens_q, seqused_q,
                                  seqused_k, block_table, start, n,
                                  candidate_topk_blocks, candidate_block_size)
                views = tuple(t[start:end] if t.numel() else t for t in self.outputs)
                self._calls.append((plans[n], projection_args, selection_args, views))

    def run(self):
        """Execute all chunks on the current stream; outputs are reused."""
        with torch.npu.device(self.device):
            for plan, projection_args, selection_args, outputs in self._calls:
                raw = compute_qk_scores(*projection_args, plan=plan, out=self.raw)
                if raw is not self.raw:
                    raise RuntimeError("Projection replaced the preallocated raw buffer")
                select_sparse_indices(*selection_args, plan=plan, out=outputs,
                                      workspace=self.workspace)
        return self.outputs

    __call__ = run


def prepare_quant_lightning_indexer(q, k, w, q_descale, k_descale, **kwargs):
    """Prepare fixed inputs and buffers before graph capture."""
    return PreparedQuantLightningIndexer(q, k, w, q_descale, k_descale, **kwargs)


def quant_lightning_indexer(q, k, w, q_descale, k_descale, **kwargs):
    """Convenience eager API. Use prepare_quant_lightning_indexer for graphs."""
    return prepare_quant_lightning_indexer(q, k, w, q_descale, k_descale, **kwargs).run()
