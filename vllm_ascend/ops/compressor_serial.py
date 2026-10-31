# SPDX-License-Identifier: Apache-2.0
"""A3 Compressor using public Torch matrix multiplication and two Vector ops."""

from __future__ import annotations

import math

import torch


def prepare_compressor_serial_weights(wkv: torch.Tensor, wgate: torch.Tensor) -> torch.Tensor:
    """Concatenate projection weights once after loading; retain the result explicitly.

    Rebuild this tensor whenever either weight changes. Preparation is outside
    the steady-state Graph timing boundary; there is no implicit weight cache.
    """
    if wkv.ndim != 2 or wgate.shape != wkv.shape or wgate.dtype != wkv.dtype:
        raise ValueError("wkv and wgate must have the same two-dimensional shape and dtype")
    if wgate.device != wkv.device or wkv.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("projection weights must share a device and use BF16 or FP16")
    return torch.cat((wkv, wgate), dim=0)


def _validate_inputs(
    x,
    wkv,
    wgate,
    state_cache,
    ape,
    norm_weight,
    rope_sin,
    rope_cos,
    state_block_table,
    cu_seqlens,
    seqused,
    start_pos,
    rope_head_dim,
    cmp_ratio,
    coff,
    norm_eps,
    rotary_mode,
    cache_mode,
    packed_weight,
):
    if x.ndim != 2 or cu_seqlens is None or start_pos is None:
        raise ValueError("A3 serial Compressor requires TND x, cu_seqlens and explicit start_pos")
    if rope_head_dim != 64 or rotary_mode != 2 or cache_mode != 1:
        raise ValueError("A3 serial Compressor requires interleaved RoPE64 and contiguous cache mode1")
    if norm_weight.ndim != 1:
        raise ValueError("norm_weight must have shape [D]")
    dim = norm_weight.numel()
    if (cmp_ratio, coff, dim) not in ((4, 2, 128), (4, 2, 512), (128, 1, 512)):
        raise ValueError("supported Compressor families are C4Li, C4A and C128A")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("x and projection weights must use BF16 or FP16")
    if x.shape[1] < 1024 or x.shape[1] > 10240 or x.shape[1] % 512:
        raise ValueError("hidden width must be 1024..10240 and divisible by512")
    if wkv.shape != (coff * dim, x.shape[1]) or wgate.shape != wkv.shape:
        raise ValueError("projection weights must have shape [coff*D,H]")
    if wkv.dtype != x.dtype or wgate.dtype != x.dtype:
        raise ValueError("x and projection weights must have the same dtype")
    if state_cache.ndim != 3 or state_cache.shape[-1] != 2 * coff * dim:
        raise ValueError("state_cache must have shape [blocks,block_size,2*coff*D]")
    if state_cache.shape[0] < 1 or not 1 <= state_cache.shape[1] <= 1024:
        raise ValueError("cache requires at least one block and block_size1..1024")
    if ape.shape != (cmp_ratio, coff * dim):
        raise ValueError("ape must have shape [cmp_ratio,coff*D]")
    if rope_sin.ndim != 2 or rope_sin.shape[1] != 64 or rope_cos.shape != rope_sin.shape:
        raise ValueError("rope_sin and rope_cos must have the same shape [capacity,64]")
    if rope_sin.shape[0] > x.shape[0]:
        raise ValueError("output capacity cannot exceed T")
    if state_block_table.ndim != 2:
        raise ValueError("state_block_table must have shape [B,table_columns]")
    batches = state_block_table.shape[0]
    if not 0 <= batches <= 256 or (x.shape[0] and batches == 0):
        raise ValueError("batch size must be0..256; nonempty x requires a nonempty batch")
    if cu_seqlens.shape != (batches + 1,) or start_pos.shape != (batches,):
        raise ValueError("cu_seqlens/start_pos must have shapes [B+1]/[B]")
    if seqused is not None and seqused.shape != (batches,):
        raise ValueError("seqused must have shape [B]")
    if not math.isfinite(norm_eps) or norm_eps <= 0:
        raise ValueError("norm_eps must be finite and positive")
    float_tensors = (state_cache, ape, norm_weight, rope_sin, rope_cos)
    metadata = (state_block_table, cu_seqlens, start_pos) + (() if seqused is None else (seqused,))
    if any(t.dtype != torch.float32 for t in float_tensors):
        raise ValueError("cache, APE, norm and RoPE tensors must use FP32")
    if any(t.dtype != torch.int32 for t in metadata):
        raise ValueError("request metadata must use INT32")
    tensors = (x, wkv, wgate) + float_tensors + metadata
    if packed_weight is not None:
        if packed_weight.shape != (2 * coff * dim, x.shape[1]) or packed_weight.dtype != x.dtype:
            raise ValueError("prepared weights must have shape [2*coff*D,H] and match x dtype")
        tensors += (packed_weight,)
    if any(t.device != x.device or not t.is_contiguous() for t in tensors):
        raise ValueError("all inputs must be contiguous and share the x device")


def compressor_serial(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    state_cache: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    state_block_table: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seqused: torch.Tensor | None,
    start_pos: torch.Tensor,
    rope_head_dim: int,
    cmp_ratio: int,
    coff: int,
    norm_eps: float,
    rotary_mode: int,
    cache_mode: int,
    *,
    packed_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return compressed KV and update FP32 state in place for A3 TND inputs.

    Valid metadata values and page ownership are caller responsibilities, as in
    the fused ABI. Validation reads only tensor properties, avoiding device-to-
    host synchronization in capture/eager calls. Padded output rows are zero.
    BSH, omitted start_pos, ring cache and other families are explicitly outside
    this implementation's scope. The existing fused dispatch is unchanged.
    """
    _validate_inputs(
        x,
        wkv,
        wgate,
        state_cache,
        ape,
        norm_weight,
        rope_sin,
        rope_cos,
        state_block_table,
        cu_seqlens,
        seqused,
        start_pos,
        rope_head_dim,
        cmp_ratio,
        coff,
        norm_eps,
        rotary_mode,
        cache_mode,
        packed_weight,
    )
    if x.shape[0] == 0:
        return x.new_empty((rope_sin.shape[0], norm_weight.numel()))
    if packed_weight is None:
        kv = torch.mm(x, wkv.T, out_dtype=torch.float32)
        gate = torch.mm(x, wgate.T, out_dtype=torch.float32)
    else:
        kv = torch.mm(x, packed_weight.T, out_dtype=torch.float32)
        gate = kv
    has_used = seqused is not None
    used = cu_seqlens if seqused is None else seqused
    pool = torch.ops._C_ascend.compressor_serial_pool(
        kv,
        gate,
        state_cache,
        ape,
        cu_seqlens,
        start_pos,
        used,
        state_block_table,
        cmp_ratio,
        coff,
        rope_sin.shape[0],
        has_used,
    )
    return torch.ops._C_ascend.compressor_serial_finalize(
        pool,
        kv,
        gate,
        ape,
        norm_weight,
        rope_sin,
        rope_cos,
        state_block_table,
        cu_seqlens,
        start_pos,
        used,
        x,
        state_cache,
        cmp_ratio,
        coff,
        has_used,
        norm_eps,
    )
