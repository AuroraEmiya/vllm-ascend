import glob
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension
import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension

root=os.path.dirname(__file__)
inc=os.path.join(os.path.dirname(torch_npu.__file__),'include/third_party/acl/inc')
setup(name='tgr-serial-ops',version='0.1.0',packages=find_packages(),
      ext_modules=[NpuExtension('tgr_ops._C',glob.glob(os.path.join(root,'tgr_ops/csrc/*.cpp')),
                                extra_compile_args=['-O2','-I'+inc])],
      cmdclass={'build_ext':BuildExtension})
