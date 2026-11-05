# SPDX-License-Identifier: Apache-2.0
"""Contact's same three vector stages with isolated public matrix API choices.

All projections are bias-free and BF16. Transpose views are prepared once;
there is no explicit device dtype cast or format conversion inside replay.
The unchanged installed Contact wrapper remains the independent control.
"""
import torch
import torch.nn.functional as F


class ContactMatrixVariant:
    def __init__(self, data, nz, route):
        self.data = data
        self.route = route
        self.backend = 'linear' if 'linear' in route else 'mm' if '_mm_' in route else 'matmul'
        self.head_backend = 'bmm' if '_bmm_' in route else 'matmul'
        self.layout = 'ND' if route.endswith('_nd') else 'FRACTAL_NZ'
        source = data if self.layout == 'ND' else nz
        self.weights = {key: source[key] for key in ('wdq', 'wdkv', 'wuq')}
        self.linear_weights = {key: value.transpose(-1, -2) for key, value in self.weights.items()}

    def dense(self, x, key):
        if self.backend == 'linear':
            return F.linear(x, self.linear_weights[key], bias=None)
        w = self.weights[key]
        if self.backend == 'mm':
            return torch.mm(x.reshape(-1, x.shape[-1]), w).view(*x.shape[:-1], w.shape[-1])
        return torch.matmul(x, w)

    def info(self):
        return dict(dense_api=self.backend, head_api=self.head_backend, layout=self.layout,
            bias=None, explicit_device_dtype_cast=False, replay_format_cast=False,
            weights={k: dict(shape=list(v.shape), stride=list(v.stride()), dtype=str(v.dtype),
                            data_ptr=v.data_ptr(), linear_transpose_ptr=self.linear_weights[k].data_ptr())
                     for k,v in self.weights.items()})

    def __call__(self, kv_cache, kr_cache):
        d = self.data
        mm_dq = self.dense(d['x'], 'wdq')
        mm_dkv = self.dense(d['x'], 'wdkv')
        cq = torch.ops.cann_ops_transformer.mla_prolog_norm_rope_contact(
            mm_dq, mm_dkv, d['gq'], d['gkv'], d['sin'], d['cos'], d['index'], kv_cache, kr_cache,
            epsilon_cq=1e-5, epsilon_ckv=1e-5, qc_qr_scale=1.0, kc_scale=1.0,
            cache_mode=0, do_rope=True)
        mm_uq = self.dense(cq, 'wuq')
        heads, dim_head, _ = d['wuk'].shape
        qc, qr = torch.ops.cann_ops_transformer.mla_prolog_qc_rope_contact(
            mm_uq, d['sin'], d['cos'], heads, dim_head, d['sin'].shape[-1], True)
        query_nt = (torch.bmm(qc, d['wuk']) if self.head_backend == 'bmm'
                    else torch.matmul(qc, d['wuk']))
        query, _ = torch.ops.cann_ops_transformer.mla_prolog_query_contact(
            query_nt, False, d['x'].shape[0], d['x'].shape[1], 3)
        return query, qr, kv_cache, kr_cache
