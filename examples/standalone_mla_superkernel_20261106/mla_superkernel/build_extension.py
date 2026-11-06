# SPDX-License-Identifier: Apache-2.0
"""Standalone Torch binding. The generic CANN L0 implementation is an external library."""
import os
from pathlib import Path
import subprocess

def load_extension():
    import torch
    import torch_npu
    from torch.utils.cpp_extension import load
    existing=os.environ.get('MLASK_EXISTING_BINDING')
    if existing:
        # Study the matrix interface against an already-built copy of our own R5 V kernels.
        # No automatic fallback: the explicit path is recorded by the runner's binary hashes.
        torch.ops.load_library(str(Path(existing).resolve(strict=True)))
        return
    root=Path(__file__).resolve().parents[1]
    cann=Path(os.environ['ASCEND_HOME_PATH'])
    build=root/'build'/'binding'
    build.mkdir(parents=True,exist_ok=True)
    opdev=build/'serial_matmul_opdev.o'
    source=root/'csrc/aclnn_torch_adapter/serial_matmul_v3/opdev.cpp'
    # CANN 9.2 ABI=0; this object only exposes C/ACL types to the ABI=1 Torch TU.
    command=[os.environ.get('CXX','c++'),'-std=c++17','-O2','-fPIC','-D_GLIBCXX_USE_CXX11_ABI=0',
             '-I'+str(cann/'include'),'-I'+str(cann/'include/aclnn'),'-c',str(source),'-o',str(opdev)]
    if not opdev.exists() or source.stat().st_mtime > opdev.stat().st_mtime:
        subprocess.run(command,check=True)
    npu=Path(torch_npu.__file__).resolve().parent
    os.environ.setdefault('MAX_JOBS','4')
    load(name='mla_superkernel_native',
         sources=[str(root/'csrc/binding.cpp'),str(root/'csrc/aclnn_torch_adapter/NPUBridge.cpp'),str(root/'csrc/aclnn_torch_adapter/NPUStorageImpl.cpp')],
         extra_include_paths=[str(root/'csrc'),str(cann/'include'),str(cann/'include/aclnn'),str(npu/'include')],
         extra_cflags=['-O2','-std=c++17'],
         extra_ldflags=[str(opdev),'-L'+str(cann/'lib64'),'-lopapi','-lnnopbase','-lascendcl','-L'+str(npu/'lib'),'-ltorch_npu'],
         build_directory=str(build),verbose=True,is_python_module=False)
