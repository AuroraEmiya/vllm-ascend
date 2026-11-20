"""BF16 ND C, then the unchanged original A3 V from its existing DSO."""
import importlib.util
import json
from pathlib import Path
import torch

_THIS = Path(__file__).resolve().parent
_TASK = _THIS.parents[1]
_ORIGINAL = _TASK / "dependencies/original_a3/n_impl"
_loaded = False


def load_libraries():
    global _loaded
    if not _loaded:
        torch.ops.load_library(str(_ORIGINAL / "build/libcv60n.so"))
        torch.ops.load_library(str(_THIS / "build/libc2_a3_bf16_c_only.so"))
        _loaded = True


def route_for(rows):
    routes = json.loads((_ORIGINAL / "configs/cv60n_routes.json").read_text())["routes"]
    cfg = routes.get(str(int(rows)))
    if cfg is None:
        keys = sorted(routes, key=lambda key: abs(int(key) - int(rows)))
        cfg = {k: v for k, v in routes[keys[0]].items() if k != "proj"}
    return dict(cfg)


class Prepared:
    def __init__(self, x, wkv, wgate, state_cache, metadata, norm_weight,
                 epsilon, max_query_len, *, soc):
        load_libraries()
        if any(t.dtype != torch.bfloat16 for t in (x, wkv, wgate)):
            raise TypeError("x, wkv and wgate must be BF16")
        if x.ndim != 2 or x.shape[1] != 5120:
            raise ValueError("x must have shape [T,5120]")
        if tuple(wkv.shape) != (512, 5120) or tuple(wgate.shape) != (512, 5120):
            raise ValueError("weights must each have shape [512,5120]")
        self.x, self.wkv, self.wgate = x, wkv, wgate
        self.state, self.metadata, self.gamma = state_cache, metadata, norm_weight
        self.epsilon, self.maxq = float(epsilon), int(max_query_len)
        if self.x.shape[0] == 0:
            self._prepare_empty()
            return
        self.route = route_for(x.shape[0])
        cfg = self.route
        self.mode = 0 if cfg["template"] == "NORM" else 1
        self.w = torch.empty((1024, 5120), dtype=torch.bfloat16, device=x.device)
        self.pair = torch.empty((x.shape[0], 1024), dtype=torch.float32, device=x.device)
        self.latent = torch.empty((x.shape[0], 512), dtype=torch.bfloat16, device=x.device)
        self.tiling, self.ws, self.blocks, info = torch.ops.c2_a3_bf16_c_only.prepare(
            x, self.w, cfg["single_m"], cfg["single_n"], cfg["base_m"], cfg["base_n"],
            self.mode, 0, soc)
        self.info = json.loads(info)
        self.chunk = int(cfg.get("chunk", 0))
        if self.chunk == 0:
            self.chunk = 16 if self.info.get("ub_bytes", 0) >= 208 * 1024 else 8
        self.last_pair = self.pair[:, :512], self.pair[:, 512:]

    def _prepare_empty(self):
        x = self.x
        tensors = (x, self.wkv, self.wgate, self.state, self.metadata, self.gamma)
        if x.device.type != "npu" or any(t.device != x.device or not t.is_contiguous()
                                         for t in tensors):
            raise ValueError("inputs must be contiguous tensors on the same NPU")
        if (self.state.dtype != torch.float32 or self.state.ndim != 3
                or tuple(self.state.shape[1:]) != (32, 1024)):
            raise ValueError("state must be FP32 [P,32,1024]")
        if (self.metadata.dtype != torch.int32 or self.metadata.ndim != 2
                or self.metadata.shape[0] != 5):
            raise ValueError("metadata must be INT32 [5,B]")
        if self.gamma.dtype != torch.bfloat16 or tuple(self.gamma.shape) != (512,):
            raise ValueError("gamma must be BF16 [512]")
        if not (self.epsilon >= 0 and self.maxq >= 0):
            raise ValueError("invalid scalars")
        self.pair = x.new_empty((0, 1024), dtype=torch.float32)
        self.latent = x.new_empty((0, 512))
        self.last_pair = self.pair[:, :512], self.pair[:, 512:]
        self.info = {"rows": 0}

    def c_only(self):
        if self.x.shape[0] == 0:
            return self.pair
        torch.cat((self.wkv, self.wgate), dim=0, out=self.w)
        return torch.ops.c2_a3_bf16_c_only.run_out(
            self.x, self.w, self.pair, self.tiling, self.ws, self.blocks, self.mode)

    def __call__(self):
        if self.x.shape[0] == 0:
            return self.latent
        self.c_only()
        torch.ops.npu.cv60n_v_run(self.pair, self.latent, self.state, self.metadata,
                                 self.gamma, self.chunk, self.epsilon, self.maxq)
        return self.latent


def prepare(x, wkv, wgate, state_cache, metadata, norm_weight, epsilon, max_query_len, *, soc):
    return Prepared(x, wkv, wgate, state_cache, metadata, norm_weight, epsilon,
                    max_query_len, soc=soc)
