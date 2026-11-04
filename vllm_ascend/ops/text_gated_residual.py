# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Serial TextGatedResidual using public torch.mm and native Ascend C V stages.

The selected forward domain is BF16, channels=4, branch hidden=2560 and
low rank=320, with 1 <= B*S <= 2048. The dimensions follow the public
Qwen3.8-Flash-Next configuration; token counts are synthetic test cases.
"""

import math

import torch

CHANNELS = 4
BRANCH_HIDDEN = 2560
WIDTH = CHANNELS * BRANCH_HIDDEN
LOW_RANK = 320
PACKED_WIDTH = 336
MAX_TOKENS = 2048


class TextGatedResidualSerial:
    """Caller-owned prepared weights; all input-dependent work stays in forward.

    The five outputs follow the low-level CANN schema. Absent optional outputs
    are empty tensors. ``need_backward`` requests forward auxiliary tensors;
    this class does not implement autograd backward.

    Down/gate weights are prepared together when gate is present. Call
    ``refresh_weights_`` after changing their contents and before Graph replay;
    this updates the prepared buffers at their existing addresses. NPU up
    weights also have an explicit BF16 NZ copy for the up projection; source views
    remain available for the ND route. Gate-free down weights remain views.
    Enable torch.npu.config.allow_internal_format during NPU setup.
    Replace the object and recapture when
    replacing a source weight tensor. Preparation never depends on input x.
    """

    def __init__(self, w_down, w_up, w_gate=None, channels=CHANNELS, eps=1e-6):
        if channels != CHANNELS or not math.isfinite(eps) or eps <= 0:
            raise ValueError("requires channels=4 and finite positive eps")
        for name, weight, shape in (
            ("w_down", w_down, (LOW_RANK, WIDTH)),
            ("w_up", w_up, (WIDTH, LOW_RANK)),
            ("w_gate", w_gate, (CHANNELS, WIDTH)),
        ):
            if name == "w_gate" and weight is None:
                continue
            if (
                not isinstance(weight, torch.Tensor)
                or tuple(weight.shape) != shape
                or weight.dtype != torch.bfloat16
                or not weight.is_contiguous()
            ):
                raise ValueError(f"{name} must be contiguous BF16 with shape {shape}")
            if weight.device != w_down.device:
                raise ValueError("all weights must be on the same device")
        self.w_down = w_down.t()
        self.w_up = w_up.t()
        self.w_gate = None if w_gate is None else w_gate.t()
        self.source_down = w_down
        self.source_gate = w_gate
        self.device = w_down.device
        self.eps = eps
        self._matrix_backend = "torch"
        self._vector_backend = "legacy"
        self._sk_packed_weight = None
        self._sk_pad_down = False
        self._sk_activation_mode = 10
        self._mix_mode = 2
        self._norm_mode = 4
        self._activation_mode = 5
        self._use_nz_up = True
        self._nz_up_max_tokens = MAX_TOKENS
        self.up_nz = None
        self.dummy = torch.empty(1, device=self.device, dtype=torch.bfloat16)
        self.dummy_gamma = torch.empty(1, device=self.device, dtype=torch.float32)
        self.packed_weight = (
            None if w_gate is None else torch.zeros((PACKED_WIDTH, WIDTH), device=self.device, dtype=torch.bfloat16)
        )
        self.refresh_weights_()

    def refresh_weights_(self):
        """Repack changed weight contents during load/update, outside forward.

        The caller orders this operation before replay on the same stream, or
        uses a stream dependency. The prepared address and captured Graph stay
        valid. All per-forward input work remains inside the Graph.
        """
        if self.packed_weight is not None:
            self.packed_weight[:LOW_RANK].copy_(self.source_down)
            self.packed_weight[LOW_RANK : LOW_RANK + CHANNELS].copy_(self.source_gate)
        if self._sk_packed_weight is not None:
            self._sk_packed_weight[:PACKED_WIDTH].copy_(self.packed_weight)
        if self.device.type == "npu":
            import torch_npu

            prepared = torch_npu.npu_format_cast(self.w_up.t(), 29)
            if prepared.dtype != torch.bfloat16 or torch_npu.get_npu_format(prepared) != 29:
                raise RuntimeError("up preparation requires actual BF16 FRACTAL_NZ format29")
            if self.up_nz is None:
                self.up_nz = prepared
            else:
                address = self.up_nz.data_ptr()
                self.up_nz.copy_(prepared)
                if self.up_nz.data_ptr() != address:
                    raise RuntimeError("refresh changed the captured up weight address")
        return self

    def prepare_superkernel_packing(self, width=640):
        """Pad only constant down/gate columns, outside capture and replay."""
        if width not in (0, 640):
            raise ValueError("SuperKernel down padding must be 0 or 640")
        self._sk_pad_down = width != 0 and self.packed_weight is not None
        if self._sk_pad_down and self._sk_packed_weight is None:
            self._sk_packed_weight = self.packed_weight.new_zeros((width, WIDTH))
            self._sk_packed_weight[:PACKED_WIDTH].copy_(self.packed_weight)
        return self

    def compile_superkernel(
        self,
        *,
        optimize=True,
        matrix_backend="v3_nz",
        optimize_options=None,
        pad_down=640,
        batch_activation=True,
        vector_backend="staged",
    ):
        """Compile the full pipeline; warm up before outer Graph capture.

        ``optimize=False`` selects the static-kernel control. Weight refresh
        remains explicit on this prepared owner, outside replay.
        """
        from vllm_ascend.compilation.serial_superkernel import CompiledSerialPipeline

        if vector_backend not in ("legacy", "staged"):
            raise ValueError("vector_backend must be legacy or staged")
        self._vector_backend = vector_backend
        self._sk_activation_mode = 11 if batch_activation else 10
        self.prepare_superkernel_packing(pad_down if matrix_backend == "v3_nz" else 0)
        return CompiledSerialPipeline(
            self,
            "text_gated_residual",
            optimize=optimize,
            matrix_backend=matrix_backend,
            optimize_options=optimize_options,
        )

    def __call__(self, x, gamma=None, need_backward=False):
        if (
            x.ndim != 3
            or x.shape[-1] != WIDTH
            or x.shape[0] < 1
            or x.shape[1] < 1
            or x.shape[0] * x.shape[1] > MAX_TOKENS
        ):
            raise ValueError("x must have shape [B,S,10240], with B,S>0 and B*S<=2048")
        if x.dtype != torch.bfloat16 or x.device != self.device or not x.is_contiguous():
            raise ValueError("x must be contiguous BF16 on the weight device")
        if gamma is not None and (
            gamma.dtype != torch.float32
            or tuple(gamma.shape) != (WIDTH,)
            or gamma.device != x.device
            or not gamma.is_contiguous()
        ):
            raise ValueError("gamma must be contiguous FP32[10240] on the input device")
        if not isinstance(need_backward, bool):
            raise ValueError("need_backward must be bool")

        batch, sequence, _ = x.shape
        tokens = batch * sequence
        flat = x.view(tokens, WIDTH)
        empty = x.new_empty(0)
        empty_fp32 = x.new_empty(0, dtype=torch.float32)
        norm = torch.empty_like(flat)
        inv_rms = x.new_empty((tokens, CHANNELS), dtype=torch.float32) if need_backward else empty_fp32
        (
            torch.ops._C_ascend.npu_tgr_norm_out
            if self._vector_backend == "staged"
            else torch.ops._C_ascend.npu_tgr_vector_out
        )(
            flat,
            self.dummy_gamma if gamma is None else gamma,
            self.dummy,
            self.dummy,
            0 if need_backward else self._norm_mode,
            CHANNELS,
            self.eps,
            norm,
            inv_rms,
            empty,
        )
        padded_down = self._matrix_backend == "v3_nz" and self._sk_pad_down
        if self._matrix_backend in ("v3", "v3_nz"):
            from vllm_ascend.ops.serial_matmul import serial_matmul_v3

            down_weight = self.source_down if self.packed_weight is None else self.packed_weight
            projected = serial_matmul_v3(norm, self._sk_packed_weight if padded_down else down_weight, transpose_b=True)
        else:
            projected = torch.matmul(norm, self.w_down if self.packed_weight is None else self.packed_weight.t())
        down = projected if self.packed_weight is None else projected[:, :LOW_RANK]
        activation = x.new_empty((tokens, LOW_RANK))
        inject = empty if self.w_gate is None else x.new_empty((tokens, 16))
        (
            torch.ops._C_ascend.npu_tgr_activation_out
            if self._vector_backend == "staged"
            else torch.ops._C_ascend.npu_tgr_vector_out
        )(
            projected,
            self.dummy_gamma,
            self.dummy,
            self.dummy,
            1 if self.packed_weight is None else (self._sk_activation_mode if padded_down else self._activation_mode),
            CHANNELS,
            self.eps,
            activation,
            empty_fp32,
            inject,
        )
        up_weight = (
            self.up_nz.t()
            if self._use_nz_up and tokens <= self._nz_up_max_tokens and self.up_nz is not None
            else self.w_up
        )
        up = (
            serial_matmul_v3(
                activation, self.up_nz if self._matrix_backend == "v3_nz" else self.w_up.t(), transpose_b=True
            )
            if self._matrix_backend in ("v3", "v3_nz")
            else torch.matmul(activation, up_weight)
        )
        mixed = x.new_empty((tokens, BRANCH_HIDDEN))
        (
            torch.ops._C_ascend.npu_tgr_mix_out
            if self._vector_backend == "staged"
            else torch.ops._C_ascend.npu_tgr_vector_out
        )(norm, self.dummy_gamma, up, self.dummy, self._mix_mode, CHANNELS, self.eps, mixed, empty_fp32, empty)
        return (
            mixed.view(batch, sequence, BRANCH_HIDDEN),
            inject if self.w_gate is None else inject[:, :CHANNELS].view(batch, sequence, CHANNELS),
            inv_rms.view(batch, sequence, CHANNELS) if need_backward else empty_fp32,
            down.view(batch, sequence, LOW_RANK) if need_backward else empty,
            up.view(batch, sequence, WIDTH) if need_backward else empty,
        )


def text_gated_residual(x, w_down, w_up, w_gate=None, channels=CHANNELS, gamma=None, eps=1e-6, need_backward=False):
    """Convenience API matching the public package's optional return convention."""
    outputs = TextGatedResidualSerial(w_down, w_up, w_gate, channels, eps)(x, gamma, need_backward)
    if need_backward:
        return outputs if w_gate is not None else (outputs[0], outputs[2], outputs[3], outputs[4])
    return outputs[:2] if w_gate is not None else outputs[0]
