# SPDX-License-Identifier: Apache-2.0
"""Independent BF16 decode MLA Prolog: public matrix calls and Ascend C V."""

import math

import torch
import torch_npu

QUERY_RANK = 1536
KV_RANK = 512
NOPE_DIM = 128
ROPE_DIM = 64
MAX_BATCH = 128
GAMMA_CACHE_MIN_BATCH = 32
NORM_OUTPUT_OVERLAP_MIN_BATCH = 32
PACKED_ROPE_COEFFICIENT_MIN_BATCH = 32
SPLIT_DMA_OVERLAP_MIN_BATCH = 32
SPLIT_DMA_OVERLAP_MAX_BATCH = 64
FRACTAL_NZ = 29
ND_FORMAT = 2
ALLOWED_HIDDEN = (1024, 2048, 3072, 4096, 5120, 6144, 7168, 7680, 8192)
ALLOWED_HEADS = (1, 2, 4, 8, 16, 32, 64, 128)
ALLOWED_BLOCK_SIZE = (16, 128)


def validate_cache_index(index_cpu: torch.Tensor, slots: int) -> None:
    """Validate host metadata before its asynchronous copy or Graph replay.

    Device index contents must satisfy this contract for every invocation.
    This explicit setup helper performs no device synchronization in forward.
    """
    if index_cpu.device.type != "cpu" or index_cpu.dtype != torch.int64:
        raise ValueError("index validation requires a CPU int64 tensor")
    flat = index_cpu.flatten()
    if bool(((flat < 0) | (flat >= slots)).any()) or flat.unique().numel() != flat.numel():
        raise ValueError("cache indices must be in range and distinct")


def _check_tensor(tensor, shape, device, name, dtype=torch.bfloat16):
    if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
        raise ValueError(f"{name} requires shape {shape}, dtype {dtype}")
    if not tensor.is_contiguous() or tensor.device != device:
        raise ValueError(f"{name} must be contiguous and on {device}")


class MlaPrologSerial:
    """Decode implementation with explicitly prepared, address-stable NZ and transposed ND weights.

    After changing any source matrix, call refresh_weights outside Graph replay.
    The method updates existing NZ and ND buffers without changing captured addresses.
    Gamma and head-projection matrices remain direct references.
    Recreate this object and recapture when addresses, shapes or metadata change.
    Kernels support the decode subdomain documented in the accompanying README.
    """

    def __init__(self, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, gamma_q, gamma_kv, eps_q=1e-5, eps_kv=1e-5):
        if weight_dq.ndim != 2 or weight_uk.ndim != 3:
            raise ValueError("weight_dq and weight_uk require ranks two and three")
        self.hidden, self.heads = weight_dq.shape[0], weight_uk.shape[0]
        if self.hidden not in ALLOWED_HIDDEN or self.heads not in ALLOWED_HEADS:
            raise ValueError("unsupported hidden size or head count")
        if not all(math.isfinite(eps) and eps > 0 for eps in (eps_q, eps_kv)):
            raise ValueError("normalization epsilons must be finite and positive")
        self.device = weight_dq.device
        if self.device.type != "npu":
            raise ValueError("MlaPrologSerial requires NPU weights")
        for tensor, shape, name in (
            (weight_dq, (self.hidden, QUERY_RANK), "weight_dq"),
            (weight_uq_qr, (QUERY_RANK, self.heads * (NOPE_DIM + ROPE_DIM)), "weight_uq_qr"),
            (weight_uk, (self.heads, NOPE_DIM, KV_RANK), "weight_uk"),
            (weight_dkv_kr, (self.hidden, KV_RANK + ROPE_DIM), "weight_dkv_kr"),
            (gamma_q, (QUERY_RANK,), "gamma_q"),
            (gamma_kv, (KV_RANK,), "gamma_kv"),
        ):
            _check_tensor(tensor, shape, self.device, name)
        self.weight_dq, self.weight_uq_qr, self.weight_uk = weight_dq, weight_uq_qr, weight_uk
        self.weight_dkv_kr, self.gamma_q, self.gamma_kv = weight_dkv_kr, gamma_q, gamma_kv
        self.eps_q, self.eps_kv = eps_q, eps_kv
        values = self._matrix_values()
        self._matrix_nz = tuple(torch_npu.npu_format_cast(value, FRACTAL_NZ) for value in values)
        self._matrix_nd = tuple(torch_npu.npu_format_cast(value.t().contiguous(), ND_FORMAT) for value in values)
        self._check_prepared_formats()
        self.dummy = torch.empty(1, dtype=torch.bfloat16, device=self.device)

    def _check_prepared_formats(self):
        if any(torch_npu.get_npu_format(value) != FRACTAL_NZ for value in self._matrix_nz):
            raise RuntimeError("prepared NZ matrix weights require actual FRACTAL_NZ format29")
        if any(torch_npu.get_npu_format(value) != ND_FORMAT for value in self._matrix_nd):
            raise RuntimeError("prepared transposed ND matrix weights require actual format2")

    def _matrix_values(self):
        return torch.cat((self.weight_dq, self.weight_dkv_kr), dim=1), self.weight_uq_qr

    def refresh_weights(self):
        """Refresh all prepared matrices on the current stream, outside capture.

        Source shapes and storage addresses must remain unchanged. Callers order
        this stream before a Graph replay on another stream. No synchronization
        or preparation occurs inside the captured forward.
        """
        addresses = tuple(value.data_ptr() for value in self._matrix_nz + self._matrix_nd)
        for source, nz_target, nd_target in zip(self._matrix_values(), self._matrix_nz, self._matrix_nd):
            nz_target.copy_(torch_npu.npu_format_cast(source, FRACTAL_NZ))
            nd_target.copy_(torch_npu.npu_format_cast(source.t().contiguous(), ND_FORMAT))
        self._check_prepared_formats()
        if addresses != tuple(value.data_ptr() for value in self._matrix_nz + self._matrix_nd):
            raise RuntimeError("refresh changed captured weight addresses")

    def prepared_weight_info(self):
        """Return public format/address/storage metadata for setup diagnostics."""
        return {
            name: {
                "format": torch_npu.get_npu_format(value),
                "data_ptr": value.data_ptr(),
                "storage_bytes": value.untyped_storage().nbytes(),
            }
            for name, value in zip(
                ("wdq_wdkv_nz", "wuq_nz", "wdq_wdkv_t_nd", "wuq_t_nd"), self._matrix_nz + self._matrix_nd
            )
        }

    @staticmethod
    def matrix_route(rows):
        """Choose a public matrix route from static decode shape metadata."""
        return "nd_transposed_linear" if rows >= 16 and rows % 16 == 0 else "nz_mm"

    def __call__(self, token_x, rope_sin, rope_cos, cache_index, kv_cache, kr_cache):
        if token_x.ndim != 3 or token_x.shape[1] != 1 or not 1 <= token_x.shape[0] <= MAX_BATCH:
            raise ValueError("supported decode shape is [B,1,He], 1 <= B <= 128")
        rows = token_x.shape[0]
        for tensor, shape, name, dtype in (
            (token_x, (rows, 1, self.hidden), "token_x", torch.bfloat16),
            (rope_sin, (rows, 1, ROPE_DIM), "rope_sin", torch.bfloat16),
            (rope_cos, (rows, 1, ROPE_DIM), "rope_cos", torch.bfloat16),
            (cache_index, (rows, 1), "cache_index", torch.int64),
        ):
            _check_tensor(tensor, shape, self.device, name, dtype)
        if kv_cache.ndim != 4 or kv_cache.shape[1] not in ALLOWED_BLOCK_SIZE or kv_cache.shape[2:] != (1, KV_RANK):
            raise ValueError("kv_cache requires PA_BSND [blocks,16|128,1,512]")
        _check_tensor(kv_cache, kv_cache.shape, self.device, "kv_cache")
        _check_tensor(kr_cache, (*kv_cache.shape[:3], ROPE_DIM), self.device, "kr_cache")
        slots = kv_cache.numel() // KV_RANK
        if slots < rows:
            raise ValueError("cache capacity is smaller than the decode batch")
        cache_kv_flat, cache_kr_flat = kv_cache.view(slots, KV_RANK), kr_cache.view(slots, ROPE_DIM)
        sine, cosine = rope_sin.view(rows, ROPE_DIM), rope_cos.view(rows, ROPE_DIM)
        index = cache_index.view(rows)
        x = token_x.view(rows, self.hidden)

        use_linear = self.matrix_route(rows) == "nd_transposed_linear"
        down_q_kv = torch.matmul(x, self._matrix_nd[0].t()) if use_linear else torch.matmul(x, self._matrix_nz[0])
        normalized_q = torch.empty((rows, QUERY_RANK), device=self.device, dtype=torch.bfloat16)
        self._vector(
            down_q_kv,
            self.dummy,
            self.gamma_q,
            self.gamma_kv,
            sine,
            cosine,
            index,
            7 if rows >= NORM_OUTPUT_OVERLAP_MIN_BATCH else 4,
            rows,
            slots,
            normalized_q,
            self.dummy,
            cache_kv_flat,
            cache_kr_flat,
        )

        up = (
            torch.matmul(normalized_q, self._matrix_nd[1].t())
            if use_linear
            else torch.matmul(normalized_q, self._matrix_nz[1])
        )
        head_q = torch.empty((self.heads, rows, NOPE_DIM), device=self.device, dtype=torch.bfloat16)
        query_rope = torch.empty((rows, self.heads, ROPE_DIM), device=self.device, dtype=torch.bfloat16)
        self._vector(
            up,
            self.dummy,
            self.dummy,
            self.dummy,
            sine,
            cosine,
            index,
            8 if rows >= PACKED_ROPE_COEFFICIENT_MIN_BATCH else 1,
            rows,
            slots,
            head_q,
            query_rope,
            cache_kv_flat,
            cache_kr_flat,
        )

        if rows == 1:
            query = torch.bmm(head_q, self.weight_uk).view(rows, self.heads, KV_RANK)
        else:
            query = torch_npu.npu_transpose_batchmatmul(
                head_q,
                self.weight_uk,
                perm_x1=[0, 1, 2],
                perm_x2=[0, 1, 2],
                perm_y=[1, 0, 2],
                batch_split_factor=1,
            )
        return (
            query.view(rows, 1, self.heads, KV_RANK),
            query_rope.view(rows, 1, self.heads, ROPE_DIM),
            kv_cache,
            kr_cache,
        )

    def _vector(
        self, x, aux, gamma_q, gamma_kv, sine, cosine, index, mode, rows, slots, y, query_rope, kv_cache, kr_cache
    ):
        torch.ops._C_ascend.mla_prolog_vector_out(
            x,
            aux,
            gamma_q,
            gamma_kv,
            sine,
            cosine,
            index,
            mode,
            rows,
            self.heads,
            slots,
            self.eps_q,
            self.eps_kv,
            y,
            query_rope,
            kv_cache,
            kr_cache,
        )
