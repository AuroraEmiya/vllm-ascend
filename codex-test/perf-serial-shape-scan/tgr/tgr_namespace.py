"""Root-created test orchestration: load fixed serial modules without vLLM init.

No operator implementation is copied here. Caller must import torch/torch_npu,
freeze this harness identity and the returned paths/hashes, and run the actual
adapter checks. Installing this namespace does not prove NPU support.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys
import types


MODULES = {
    "vllm_ascend.ops.serial_matmul": (
        "vllm_ascend/ops/serial_matmul.py",
        "e666f2b17fd932638c549e9e2862f510b67ab993d28d2c3a2477be7b1a6423d7",
    ),
    "scan_tgr_vector_r5": (
        "vllm_ascend/ops/text_gated_residual.py",
        "86d1dd234283f19f8fbc3e73b202d28fe7020ff0762c07c7e73d64cb1de17d30",
    ),
}


def install_serial_namespace(source_root):
    """Preload only the shared wrapper; adapter loads the checked TGR wrapper.

    Run in a new test process, before anything imports vllm_ascend. Existing
    package modules are rejected instead of being replaced or patched.
    """
    root = Path(source_root).resolve(strict=True)
    occupied = [name for name in sys.modules
                if name == "vllm_ascend" or name.startswith("vllm_ascend.")]
    if occupied:
        raise RuntimeError("test namespace must precede package imports: " + ",".join(sorted(occupied)))
    evidence = []
    for name, (relative, expected) in MODULES.items():
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root):
            raise RuntimeError("serial source escapes fixed checkout")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise RuntimeError("fixed serial module hash mismatch: " + relative)
        evidence.append({"module": name, "path": str(path), "sha256": observed})
    inserted = []
    try:
        for name, relative in (("vllm_ascend", "vllm_ascend"),
                               ("vllm_ascend.ops", "vllm_ascend/ops")):
            module = types.ModuleType(name)
            module.__package__ = name
            module.__path__ = [str(root / relative)]
            sys.modules[name] = module
            inserted.append(name)
        sys.modules["vllm_ascend"].ops = sys.modules["vllm_ascend.ops"]
        name = "vllm_ascend.ops.serial_matmul"
        spec = importlib.util.spec_from_file_location(name, evidence[0]["path"])
        if spec is None or spec.loader is None:
            raise RuntimeError("serial module loader unavailable")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        inserted.append(name)
        spec.loader.exec_module(module)
        sys.modules["vllm_ascend.ops"].serial_matmul = module
        return {"method": "test_process_fixed_serial_namespace_v1", "modules": evidence,
                "preloaded_module": name, "package_initializers_executed": False}
    except BaseException:
        for name in reversed(inserted):
            sys.modules.pop(name, None)
        raise
