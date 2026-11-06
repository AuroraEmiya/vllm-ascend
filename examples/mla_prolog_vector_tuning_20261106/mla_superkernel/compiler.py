# SPDX-License-Identifier: Apache-2.0
"""Compile a complete prepared serial operator with npugraph_ex SuperKernel.

Compilation is lazy: warm up and synchronize before outer Graph capture.
Then call optimize_graph(graph) before replay to perform binary grouping.
The caller retains input, weight and cache ownership. Refresh prepared weights
on the original object outside replay; recapture after address/metadata changes.
"""

import copy
import os
import time

import torch


class CompiledSerialPipeline:
    """Opt-in complete-pipeline compilation, with an explicit static-only control.

    No eager fallback is installed. Successful compilation is not evidence that
    every child kernel was grouped; inspect the device trace for that claim.
    """

    def __init__(self, pipeline, scope, *, optimize=True, matrix_backend="v3_nz", optimize_options=None):
        if matrix_backend not in ("torch", "torch_matmul", "torch_matmul_nz", "v3", "v3_nz"):
            raise ValueError("matrix_backend must be torch, torch_matmul, torch_matmul_nz, v3 or v3_nz")
        if matrix_backend in ("v3", "v3_nz") and not hasattr(torch.ops._C_ascend, "serial_matmul_v3_out"):
            raise RuntimeError("explicit V3 adapter requires the CANN opdev build")
        execution = copy.copy(pipeline)
        execution._matrix_backend = matrix_backend
        self.matrix_backend = matrix_backend
        if not hasattr(torch.npu.NPUGraph, "super_kernel_optimize"):
            raise RuntimeError("this Torch-NPU build does not provide SuperKernel")
        if "LOCAL_WORLD_SIZE" not in os.environ:
            os.environ["LOCAL_WORLD_SIZE"] = "1"
        self.optimize = optimize
        self.optimize_options = copy.deepcopy(
            {"constant_codegen": 1, "preload_code": 5} if optimize_options is None else optimize_options
        )
        self.pipeline = pipeline
        self.scope = scope
        self.preparation = {"first_call_wall_us_compile_install_execute": None, "outer_graph_optimize_wall_us": []}
        self.options = {
            "force_eager": True,
            "inplace_pass": False,
            "clone_input": False,
            "clone_output": False,
            "static_kernel_compile": True,
            "super_kernel_optimize": True,
        }
        # Both controls compile the same SK-ready children; only the outer
        # Graph grouping is disabled for optimize=False.
        # Default CANN child cache handling; whole-cache flush is opt-in.
        self.options["super_kernel_optimize_options"] = self.optimize_options

        def forward(*args, **kwargs):
            torch.npu.super_kernel_scope_begin(scope)
            outputs = execution(*args, **kwargs)
            torch.npu.super_kernel_scope_end(scope)
            return outputs

        self.compiled = torch.compile(
            forward, backend="npugraph_ex", options=self.options, dynamic=False, fullgraph=True
        )

    def __call__(self, *args, **kwargs):
        if self.preparation["first_call_wall_us_compile_install_execute"] is None:
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            result = self.compiled(*args, **kwargs)
            torch.npu.synchronize()
            self.preparation["first_call_wall_us_compile_install_execute"] = (time.perf_counter_ns() - start) / 1000
            return result
        return self.compiled(*args, **kwargs)

    def optimize_graph(self, graph):
        """Group the completed outer Graph after capture and before replay.

        force_eager compiles static children while the caller owns capture.
        As in vLLM-Ascend, this explicit step performs the binary grouping.
        """
        if self.optimize:
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            graph.super_kernel_optimize(optimize_options=self.optimize_options)
            torch.npu.synchronize()
            self.preparation["outer_graph_optimize_wall_us"].append((time.perf_counter_ns() - start) / 1000)
        return graph

    def __getattr__(self, name):
        # Preparation and refresh methods keep operating on the original owner.
        return getattr(self.pipeline, name)
