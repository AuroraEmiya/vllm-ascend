# SPDX-License-Identifier: Apache-2.0
from .build_extension import load_extension

load_extension()

import torch

def _no_output_meta(*args, **kwargs):
    return None

for _name in ("mla_prolog_vector_out", "mla_norm_cache_out", "mla_split_rope_out", "mla_layout_out", "serial_matmul_v3_out"):
    if not torch._C._dispatch_has_kernel_for_dispatch_key("_C_ascend::" + _name,"Meta"):
        torch.library.register_fake("_C_ascend::" + _name)(_no_output_meta)

from .pipeline import MlaPrologSerial

__all__ = ["MlaPrologSerial"]
