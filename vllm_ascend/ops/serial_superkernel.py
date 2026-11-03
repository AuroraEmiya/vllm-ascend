# SPDX-License-Identifier: Apache-2.0
"""Compile a complete prepared serial operator with npugraph_ex SuperKernel.

Compilation is lazy: warm up and synchronize before outer Graph capture.
Then call optimize_graph(graph) before replay to perform binary grouping.
The caller retains input, weight and cache ownership. Refresh prepared weights
on the original object outside replay; recapture after address/metadata changes.
"""

import os

import torch


class CompiledSerialPipeline:
    """Opt-in complete-pipeline compilation, with an explicit static-only control.

    No eager fallback is installed. Successful compilation is not evidence that
    every child kernel was grouped; inspect the device trace for that claim.
    """

    def __init__(self, pipeline, scope, *, optimize=True):
        if not hasattr(torch.npu.NPUGraph, "super_kernel_optimize"):
            raise RuntimeError("this Torch-NPU build does not provide SuperKernel")
        if "LOCAL_WORLD_SIZE" not in os.environ:
            os.environ["LOCAL_WORLD_SIZE"] = "1"
        self.optimize = optimize
        self.pipeline = pipeline
        self.scope = scope
        self.options = {
            "force_eager": True,
            "inplace_pass": False,
            "clone_input": False,
            "clone_output": False,
            "static_kernel_compile": True,
            "super_kernel_optimize": optimize,
        }
        if optimize:
            self.options["super_kernel_optimize_options"] = {"dcci_after_kernel_end": [".*"]}

        def forward(*args, **kwargs):
            if optimize:
                torch.npu.super_kernel_scope_begin(scope)
            outputs = pipeline(*args, **kwargs)
            if optimize:
                torch.npu.super_kernel_scope_end(scope)
            return outputs

        self.compiled = torch.compile(
            forward, backend="npugraph_ex", options=self.options, dynamic=False, fullgraph=True
        )

    def __call__(self, *args, **kwargs):
        return self.compiled(*args, **kwargs)

    def optimize_graph(self, graph):
        """Group the completed outer Graph after capture and before replay.

        force_eager compiles static children while the caller owns capture.
        As in vLLM-Ascend, this explicit step performs the binary grouping.
        """
        if self.optimize:
            graph.super_kernel_optimize(optimize_options={"dcci_after_kernel_end": [".*"]})
        return graph

    def __getattr__(self, name):
        # Preparation and refresh methods keep operating on the original owner.
        return getattr(self.pipeline, name)
