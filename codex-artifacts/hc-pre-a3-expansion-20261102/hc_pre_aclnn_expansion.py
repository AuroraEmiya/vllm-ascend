"""Parameterised task-local copy of hc_pre_aclnn.py; no NPU work on import.

The baseline vendor is selected explicitly by PERF_BASELINE_VENDOR. Both scalar
epsilons are forwarded unchanged; this file does not edit the baseline source.
"""

import ctypes as C
import hashlib
import os
from pathlib import Path

import torch

VENDOR = Path(os.environ["PERF_BASELINE_VENDOR"]).resolve()
BINARY = VENDOR / "op_api/lib/libcust_opapi.so"
P, I64, U64 = C.c_void_p, C.c_int64, C.c_uint64


class HcPreAclnn:
    def __init__(self, inputs, *, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps):
        self.inputs = inputs
        # Actual installed aclnn_hc_pre.h: hcEps precedes normEps. Keep the
        # public Python keyword names separate from the C positional ABI.
        self.named_parameters = dict(
            hc_mult=hc_mult, hc_sinkhorn_iters=hc_sinkhorn_iters, norm_eps=norm_eps, hc_eps=hc_eps
        )
        self.parameters = (hc_mult, hc_sinkhorn_iters, hc_eps, norm_eps)
        self.api = C.CDLL(str(BINARY), mode=C.RTLD_LOCAL)
        self.base = C.CDLL("libopapi.so", mode=C.RTLD_LOCAL)
        self.base.aclCreateTensor.argtypes = [
            C.POINTER(I64),
            U64,
            C.c_int,
            C.POINTER(I64),
            I64,
            C.c_int,
            C.POINTER(I64),
            U64,
            P,
        ]
        self.base.aclCreateTensor.restype = P
        self.api.aclnnHcPreGetWorkspaceSize.argtypes = (
            [P] * 5 + [I64, I64, C.c_double, C.c_double] + [P] * 4 + [C.POINTER(U64), C.POINTER(P)]
        )
        self.api.aclnnHcPreGetWorkspaceSize.restype = C.c_int
        self.api.aclnnHcPre.argtypes = [P, U64, P, P]
        self.api.aclnnHcPre.restype = C.c_int
        self.keepalive = []

    def tensor(self, tensor):
        assert tensor.is_contiguous() and tensor.storage_offset() == 0
        dtype = {torch.bfloat16: 27, torch.float32: 0}[tensor.dtype]
        shape = (I64 * tensor.ndim)(*tensor.shape)
        stride = (I64 * tensor.ndim)(*tensor.stride())
        handle = self.base.aclCreateTensor(
            shape, tensor.ndim, dtype, stride, 0, 2, shape, tensor.ndim, tensor.data_ptr()
        )
        assert handle, "aclCreateTensor returned null"
        self.keepalive.append((tensor, shape, stride, handle))
        return handle

    def __call__(self):
        x = self.inputs[0]
        t, hc, hidden = x.shape
        output = (
            torch.empty((t, hidden), dtype=x.dtype, device=x.device),
            torch.empty((t, hc), dtype=torch.float32, device=x.device),
            torch.empty((t, hc, hc), dtype=torch.float32, device=x.device),
            torch.empty((t, hc), dtype=torch.float32, device=x.device),
        )
        handles = [self.tensor(value) for value in (*self.inputs, *output)]
        size, executor = U64(), P()
        status = self.api.aclnnHcPreGetWorkspaceSize(
            *handles[:5], *self.parameters, *handles[5:], C.byref(size), C.byref(executor)
        )
        assert status == 0, f"aclnnHcPreGetWorkspaceSize: {status}"
        workspace = torch.empty((max(1, size.value),), dtype=torch.uint8, device=x.device)
        self.keepalive.append(workspace)
        status = self.api.aclnnHcPre(workspace.data_ptr(), size.value, executor, torch.npu.current_stream().npu_stream)
        assert status == 0, f"aclnnHcPre: {status}"
        return output

    def identity(self):
        maps = Path("/proc/self/maps").read_text()
        return {
            "binding": "direct aclnnHcPre; torch tensor storage; current NPU stream",
            "vendor": str(VENDOR),
            "parameters": self.named_parameters,
            "aclnn_argument_order": ["hc_mult", "hc_sinkhorn_iters", "hc_eps", "norm_eps"],
            "aclnn_positional_parameters": self.parameters,
            "mapped_custom_libraries": sorted({line.split()[-1] for line in maps.splitlines() if "libcust_op" in line}),
            "vendor_hashes": {
                str(path.relative_to(VENDOR)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in VENDOR.rglob("*")
                if path.is_file() and path.suffix in (".so", ".o", ".json")
            },
        }
