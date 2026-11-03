import glob as _glob
import os as _os
from pathlib import Path as _Path
import torch as _torch
_here = _Path(__file__).resolve().parent
for _so in _glob.glob(str(_here / "hc_pre_ext*.so")):
    _torch.ops.load_library(_so)
