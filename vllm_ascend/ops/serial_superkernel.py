# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports for callers using the original experiment module.

New code imports Graph compilation from vllm_ascend.compilation.serial_superkernel
and the matrix primitive from vllm_ascend.ops.serial_matmul.
"""

from vllm_ascend.compilation.serial_superkernel import CompiledSerialPipeline
from vllm_ascend.ops.serial_matmul import serial_matmul_v3

__all__ = ["CompiledSerialPipeline", "serial_matmul_v3"]
