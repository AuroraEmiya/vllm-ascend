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
    this updates the prepared buffer at its existing address. Up weights and
    gate-free down weights remain views. Replace the object and recapture when
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
        return self

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
        torch.ops._C_ascend.npu_tgr_vector_out(
            flat,
            self.dummy_gamma if gamma is None else gamma,
            self.dummy,
            self.dummy,
            0 if need_backward else 4,
            CHANNELS,
            self.eps,
            norm,
            inv_rms,
            empty,
        )
        projected = torch.matmul(norm, self.w_down if self.packed_weight is None else self.packed_weight.t())
        down = projected if self.packed_weight is None else projected[:, :LOW_RANK]
        activation = x.new_empty((tokens, LOW_RANK))
        inject = empty if self.w_gate is None else x.new_empty((tokens, 16))
        torch.ops._C_ascend.npu_tgr_vector_out(
            projected,
            self.dummy_gamma,
            self.dummy,
            self.dummy,
            1 if self.packed_weight is None else 5,
            CHANNELS,
            self.eps,
            activation,
            empty_fp32,
            inject,
        )
        up = torch.matmul(activation, self.w_up)
        mixed = x.new_empty((tokens, BRANCH_HIDDEN))
        torch.ops._C_ascend.npu_tgr_vector_out(
            norm, self.dummy_gamma, up, self.dummy, 2, CHANNELS, self.eps, mixed, empty_fp32, empty
        )
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
