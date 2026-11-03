#!/usr/bin/env python3
"""Wheel packaging for the hc_pre AscendC kernel extension.

The built .so uses TORCH_LIBRARY registration (no PyInit_), so it is wrapped as
a Python package: hc_pre/__init__.py calls torch.ops.load_library() on the
vendored _kernel.<plat>.so. NpuExtension only keeps ext_modules -> platform
specific wheel tag; a missing .so triggers cmake + make via build_lib.
"""

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

KERNEL_DIR = Path(__file__).resolve().parent
BUILD_DIR = KERNEL_DIR / "build"
PKG_NAME = "hc_pre"
SOC_VERSION = os.environ.get("SOC_VERSION", "ascend910_9391")
ASCEND_HOME = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.1.0")


class BuildLib(build_py):
    """Trigger cmake + make when the .so is not present yet."""

    def run(self):
        so_files = glob.glob(str(BUILD_DIR / "hc_pre_ext*.so"))
        if not so_files:
            env = os.environ.copy()
            env["SOC_VERSION"] = SOC_VERSION
            env["ASCEND_HOME_PATH"] = ASCEND_HOME
            env["ASCEND_HOME"] = ASCEND_HOME
            subprocess.check_call(
                ["cmake", str(KERNEL_DIR), "-B", str(BUILD_DIR),
                 f"-DSOC_VERSION={SOC_VERSION}",
                 f"-DASCEND_CANN_PACKAGE_PATH={ASCEND_HOME}",
                 "-DCMAKE_BUILD_TYPE=Release"],
                env=env, cwd=str(KERNEL_DIR))
            subprocess.check_call(["cmake", "--build", str(BUILD_DIR), "-j"], env=env,
                                  cwd=str(KERNEL_DIR))
        super().run()


def _package_data():
    data = []
    for so in glob.glob(str(BUILD_DIR / "hc_pre_ext*.so")):
        dst = KERNEL_DIR / PKG_NAME / Path(so).name
        (KERNEL_DIR / PKG_NAME).mkdir(parents=True, exist_ok=True)
        shutil.copy2(so, dst)
        data.append(Path(so).name)
    init_py = KERNEL_DIR / PKG_NAME / "__init__.py"
    init_py.write_text(
        'import glob as _glob\n'
        'import os as _os\n'
        'from pathlib import Path as _Path\n'
        'import torch as _torch\n'
        '_here = _Path(__file__).resolve().parent\n'
        'for _so in _glob.glob(str(_here / "hc_pre_ext*.so")):\n'
        '    _torch.ops.load_library(_so)\n',
        encoding="utf-8")
    return data


setup(
    name=PKG_NAME,
    version="0.1.0",
    packages=[PKG_NAME],
    package_data={PKG_NAME: _package_data()},
    cmdclass={"build_py": BuildLib},
)
