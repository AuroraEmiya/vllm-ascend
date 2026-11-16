# SPDX-License-Identifier: Apache-2.0
"""Independent BF16 decode MLA Prolog: public matrix calls and Ascend C V."""

import math

import torch
import torch_npu

QUERY_RANK = 1536
KV_RANK = 512
NOPE_DIM = 128
ROPE_DIM = 64
MAX_BATCH = 65536
MAX_ROWS = 65536
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
        self._matrix_backend = "v3_nz"
        values = self._matrix_values()
        self._matrix_nz = tuple(torch_npu.npu_format_cast(value, FRACTAL_NZ) for value in values)
        self._matrix_nd = tuple(torch_npu.npu_format_cast(value.t().contiguous(), ND_FORMAT) for value in values)
        self._check_prepared_formats()

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

    def compile_superkernel(self, *, optimize=True, matrix_backend="v3_nz", optimize_options=None):
        """Compile the final pipeline; warm up before outer Graph capture.

        optimize=False compiles the same children without outer grouping.
        Vector tiling is selected by each host operator from the static shape.
        """
        from mla_superkernel.compiler import CompiledSerialPipeline

        return CompiledSerialPipeline(
            self, "mla_prolog", optimize=optimize, matrix_backend=matrix_backend,
            optimize_options=optimize_options,
        )

    def _project(self, values, weight_index):
        """Shared matrix dispatch for the two 2-D projections.

        v3_nz is the final measured route. Public torch.matmul routes remain
        available for the explicitly requested SuperKernel interface study.
        """
        if self._matrix_backend in ("v3", "v3_nz"):
            from mla_superkernel.serial_matmul import serial_matmul_v3

            if self._matrix_backend == "v3_nz":
                return serial_matmul_v3(values, self._matrix_nz[weight_index])
            return serial_matmul_v3(values, self._matrix_nd[weight_index], transpose_b=True)
        if (self._matrix_backend != "torch_matmul_nz" and
                self.matrix_route(values.shape[0]) == "nd_transposed_linear"):
            return torch.matmul(values, self._matrix_nd[weight_index].t())
        return torch.matmul(values, self._matrix_nz[weight_index])

    def __call__(self, token_x, rope_sin, rope_cos, cache_index, kv_cache, kr_cache):
        if token_x.ndim != 3 or not 1 <= token_x.shape[0] <= MAX_BATCH or token_x.shape[1] < 1:
            raise ValueError("supported nonempty shape is [B,S,He], 1 <= B <= 65536, S >= 1")
        batch, sequence = token_x.shape[:2]
        rows = batch * sequence
        if rows > MAX_ROWS:
            raise ValueError("this implementation supports B*S <= 65536")
        for tensor, shape, name, dtype in (
            (token_x, (batch, sequence, self.hidden), "token_x", torch.bfloat16),
            (rope_sin, (batch, sequence, ROPE_DIM), "rope_sin", torch.bfloat16),
            (rope_cos, (batch, sequence, ROPE_DIM), "rope_cos", torch.bfloat16),
            (cache_index, (batch, sequence), "cache_index", torch.int64),
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

        # 1. Combined Q/KV down projection: [T,H] -> [T,1536+512+64].
        down_q_kv = self._project(x, 0)

        # 2. Q/KV RMSNorm, KV RoPE and in-place cache updates.
        normalized_q = torch.empty((rows, QUERY_RANK), device=self.device, dtype=torch.bfloat16)
        torch.ops._C_ascend.mla_norm_cache_out(
            down_q_kv, self.gamma_q, self.gamma_kv, sine, cosine, index,
            rows, slots, self.eps_q, self.eps_kv,
            normalized_q, cache_kv_flat, cache_kr_flat,
        )

        # 3. Q up projection: [T,1536] -> [T,N*(128+64)].
        up = self._project(normalized_q, 1)

        # 4. NoPE is written in the exact layout consumed by the head matmul.
        head_q = torch.empty((self.heads, rows, NOPE_DIM), device=self.device, dtype=torch.bfloat16)
        query_rope = torch.empty((rows, self.heads, ROPE_DIM), device=self.device, dtype=torch.bfloat16)
        torch.ops._C_ascend.mla_split_rope_out(
            up, sine, cosine, rows, self.heads, head_q, query_rope,
        )

        # 5. Per-head projection: [N,T,128] @ [N,128,512].
        if self._matrix_backend in ("v3", "v3_nz"):
            from mla_superkernel.serial_matmul import serial_matmul_v3

            head_product = serial_matmul_v3(head_q, self.weight_uk)
        else:
            head_product = torch.matmul(head_q, self.weight_uk)

        # 6. Vector addressing maps [N,T,512] to [T,N,512]. For T=1 the
        #    addresses already match, so only tensor metadata is needed.
        if rows == 1:
            query = head_product.view(rows, self.heads, KV_RANK)
        else:
            query = head_product.new_empty((rows, self.heads, KV_RANK))
            torch.ops._C_ascend.mla_layout_out(
                head_product.view(self.heads * rows, KV_RANK), rows, self.heads, query,
            )
        return (
            query.view(batch, sequence, self.heads, KV_RANK),
            query_rope.view(batch, sequence, self.heads, ROPE_DIM),
            kv_cache,
            kr_cache,
        )
