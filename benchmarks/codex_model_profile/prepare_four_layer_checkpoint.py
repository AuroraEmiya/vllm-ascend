"""Build an independent four-layer Kimi K3.1 checkpoint without editing its source.

Only selected tensors are materialized. The manifest records selected tensor
checksums and source stat/header fingerprints; full source payloads are not hashed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
import struct
import sys
from pathlib import Path, PurePosixPath

LAYER = re.compile(
    r"^(?:(?:model\.)?language_model\.(?:model\.)?|model\.)?layers\.(\d+)\."
)
DTYPE_BITS = {
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "I16": 16,
    "U16": 16,
    "I32": 32,
    "U32": 32,
    "I64": 64,
    "U64": 64,
    "F16": 16,
    "BF16": 16,
    "F32": 32,
    "F64": 64,
    "F8_E4M3": 8,
    "F8_E5M2": 8,
    "F8_E8M0": 8,
}
CANONICAL_INDEX = "model.safetensors.index.json"
MANIFEST = "four_layer_manifest.json"


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=unique_object)


def write_json(path, obj):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def layer_id(name):
    match = LAYER.match(name)
    return int(match.group(1)) if match else None


def selected(name):
    index = layer_id(name)
    return index is None or index < 4


def safe_relative(name):
    if not isinstance(name, str) or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe checkpoint-relative filename: {name!r}")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or not path.parts
        or any(x in (".", "..") for x in path.parts)
    ):
        raise ValueError(f"Unsafe checkpoint-relative filename: {name!r}")
    if path.suffix != ".safetensors":
        raise ValueError(f"Index references a non-safetensors file: {name}")
    return path


def header(path):
    """Read and validate metadata only, without loading any tensor payload."""
    stat = path.stat()
    with path.open("rb") as stream:
        size_bytes = stream.read(8)
        if len(size_bytes) != 8:
            raise ValueError(f"Truncated safetensors file: {path}")
        size = struct.unpack("<Q", size_bytes)[0]
        if size < 2 or size > 64 << 20 or 8 + size > stat.st_size:
            raise ValueError(f"Invalid safetensors header length: {path}")
        raw = stream.read(size)
    info = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(info, dict):
        raise ValueError(f"Header must be an object: {path}")
    tensors = {key: value for key, value in info.items() if key != "__metadata__"}
    spans = []
    payload_size = stat.st_size - 8 - size
    for name, value in tensors.items():
        if not isinstance(value, dict):
            raise ValueError(f"Malformed tensor header: {path}:{name}")
        shape, dtype, offsets = (
            value.get("shape"),
            value.get("dtype"),
            value.get("data_offsets"),
        )
        if (
            not isinstance(shape, list)
            or any(type(x) is not int or x < 0 for x in shape)
            or dtype not in DTYPE_BITS
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(x) is not int for x in offsets)
        ):
            raise ValueError(
                f"Unsupported/invalid tensor metadata: {path}:{name}: {value}"
            )
        start, end = offsets
        expected_bytes = (math.prod(shape) * DTYPE_BITS[dtype] + 7) // 8
        if (
            start < 0
            or end < start
            or end > payload_size
            or end - start != expected_bytes
        ):
            raise ValueError(f"Invalid tensor byte span: {path}:{name}")
        spans.append((start, end))
    position = 0
    for start, end in sorted(spans):
        if start != position:
            raise ValueError(
                f"Overlapping or non-contiguous safetensors payload: {path}"
            )
        position = end
    if position != payload_size:
        raise ValueError(f"Unaccounted safetensors payload bytes: {path}")
    return (
        tensors,
        8 + size,
        {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "header_sha256": hashlib.sha256(size_bytes + raw).hexdigest(),
        },
    )


def clipped_config(config):
    result = copy.deepcopy(config)
    text = result.get("text_config", result)
    if not isinstance(text, dict) or type(text.get("num_hidden_layers")) is not int:
        raise ValueError("Missing text num_hidden_layers")
    original_layers = text["num_hidden_layers"]
    if original_layers < 4 or text.get("attn_res_block_size") != 12:
        raise ValueError(
            "Expected at least four layers and unchanged attn_res_block_size=12"
        )
    linear = text.get("linear_attn_config")
    dsa = text.get("dsa_config")
    if not isinstance(linear, dict) or not isinstance(dsa, dict):
        raise ValueError("Expected Kimi K3.1 linear_attn_config and dsa_config")
    for owner, key in (
        (linear, "kda_layers"),
        (linear, "full_attn_layers"),
        (dsa, "dsa_layers"),
    ):
        values = owner.get(key)
        if (
            not isinstance(values, list)
            or any(type(x) is not int or x < 1 or x > original_layers for x in values)
            or len(values) != len(set(values))
        ):
            raise ValueError(f"Invalid one-based source layer list: {key}")
        owner[key] = [x for x in values if x <= 4]
    if (
        linear["kda_layers"] != [1, 2, 3]
        or linear["full_attn_layers"] != [4]
        or dsa["dsa_layers"] != [4]
    ):
        raise ValueError(
            "Source prefix must already be KDA[1,2,3], full/DSA[4]; refusing to change layer types"
        )
    text["num_hidden_layers"] = 4
    return result, original_layers


def discover_index(source, index_name=None):
    if index_name:
        if Path(index_name).name != index_name:
            raise ValueError("--index must be a basename inside the source directory")
        paths = [source / index_name]
    else:
        paths = sorted(source.glob("*.safetensors.index.json"))
    if not paths:
        raise ValueError(
            "No *.safetensors.index.json found; specify --index if necessary"
        )
    indexes = [(path, read_json(path)) for path in paths]
    if any(
        not isinstance(value.get("weight_map"), dict) or not value["weight_map"]
        for _, value in indexes
    ):
        raise ValueError("Checkpoint index must contain a non-empty weight_map")
    # Some checkpoints provide the same index under canonical and ModelSlim names.
    if any(
        value["weight_map"] != indexes[0][1]["weight_map"] for _, value in indexes[1:]
    ):
        raise ValueError("Multiple different indexes found; choose one with --index")
    return indexes[0]


def copy_resources(source, output, max_resource_bytes):
    """Copy small auxiliary files; never follow directory symlinks or copy weights."""
    copied = []
    skip_names = {
        "config.json",
        "quant_model_description.json",
        "PREPARING.json",
        MANIFEST,
    }
    for path in sorted(source.iterdir()):
        if (
            not path.is_file()
            or path.name in skip_names
            or path.name.endswith(".safetensors.index.json")
        ):
            continue
        if path.suffix.lower() in {
            ".safetensors",
            ".bin",
            ".pt",
            ".pth",
            ".ckpt",
            ".onnx",
            ".zip",
            ".tar",
        }:
            continue
        if path.stat().st_size > max_resource_bytes:
            continue
        shutil.copyfile(path, output / path.name)
        copied.append(
            {
                "name": path.name,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return copied


def build_checkpoint(
    source,
    output,
    *,
    index_name=None,
    shard_bytes=512 << 20,
    tensor_framework="pt",
    max_resource_bytes=64 << 20,
):
    from safetensors import safe_open

    if tensor_framework == "pt":
        import torch
        from safetensors.torch import save_file

        def content_sha(tensor):
            view = tensor.reshape(-1).view(torch.uint8).numpy()
            return hashlib.sha256(memoryview(view).cast("B")).hexdigest()
    elif tensor_framework == "np":
        from safetensors.numpy import save_file

        def content_sha(tensor):
            return hashlib.sha256(memoryview(tensor).cast("B")).hexdigest()
    else:
        raise ValueError("tensor_framework must be pt or np")

    source = Path(source).resolve(strict=True)
    output = Path(output).resolve()
    if not source.is_dir() or output.exists():
        raise ValueError("Source must be a directory and output must not exist")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Output must be independent of the source directory")
    if shard_bytes <= 0 or max_resource_bytes < 0:
        raise ValueError("Invalid shard/resource size limit")
    config_path = source / "config.json"
    quant_path = source / "quant_model_description.json"
    config, original_layers = clipped_config(read_json(config_path))
    quant = read_json(quant_path)
    if not isinstance(quant, dict):
        raise ValueError("quant_model_description.json must be an object")
    index_path, source_index = discover_index(source, index_name)
    source_small = {
        path.name: sha256_file(path) for path in (config_path, quant_path, index_path)
    }
    weight_map = source_index["weight_map"]
    files = {}
    for name, filename in weight_map.items():
        if not isinstance(name, str):
            raise ValueError("Weight names must be strings")
        safe_relative(filename)
        files.setdefault(filename, set()).add(name)
    headers, source_fingerprints, selected_records, plan = {}, {}, {}, []
    source_tensor_bytes = 0
    source_layer_ids = set()
    for filename, names in sorted(files.items()):
        tensors, _, fingerprint = header(source / filename)
        if set(tensors) != names:
            raise ValueError(f"Index/header tensor mismatch for {filename}")
        headers[filename], source_fingerprints[filename] = tensors, fingerprint
        group, group_bytes = [], 0
        for name, info in tensors.items():
            size = info["data_offsets"][1] - info["data_offsets"][0]
            source_tensor_bytes += size
            index = layer_id(name)
            if index is not None:
                source_layer_ids.add(index)
            if not selected(name):
                continue
            if group and group_bytes + size > shard_bytes:
                plan.append((filename, group))
                group, group_bytes = [], 0
            group.append(name)
            group_bytes += size
            selected_records[name] = {
                "source_shard": filename,
                "shape": info["shape"],
                "dtype": info["dtype"],
                "bytes": size,
                "layer": index,
            }
        if group:
            plan.append((filename, group))
    if not set(range(4)).issubset(source_layer_ids):
        raise ValueError("Source checkpoint does not contain all decoder layers 0..3")
    selected_layer_ids = {
        record["layer"]
        for record in selected_records.values()
        if record["layer"] is not None
    }
    if selected_layer_ids != set(range(4)):
        raise ValueError("Selected layer proof is not exactly 0..3")
    declared_size = source_index.get("metadata", {}).get("total_size")
    if declared_size is not None and declared_size != source_tensor_bytes:
        raise ValueError(
            f"Source index total_size {declared_size} != header tensor bytes {source_tensor_bytes}"
        )
    selected_size = sum(record["bytes"] for record in selected_records.values())
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "PREPARING.json", {"source": str(source), "status": "incomplete"}
    )
    output_map, output_files = {}, {}
    for number, (filename, names) in enumerate(plan, start=1):
        output_name = f"four-layer-{number:05d}-of-{len(plan):05d}.safetensors"
        with safe_open(
            str(source / filename), framework=tensor_framework, device="cpu"
        ) as reader:
            tensors = {}
            for name in names:
                tensor = reader.get_tensor(name)
                tensors[name] = tensor
                selected_records[name]["sha256"] = content_sha(tensor)
                selected_records[name]["output_shard"] = output_name
                output_map[name] = output_name
            save_file(tensors, str(output / output_name), metadata={"format": "pt"})
            del tensor, tensors
        # Verify every output tensor's exact bytes without re-reading source payloads.
        written, data_start, _ = header(output / output_name)
        if set(written) != set(names):
            raise ValueError(f"Output tensor set mismatch: {output_name}")
        with (output / output_name).open("rb") as stream:
            for name in names:
                info, expected = written[name], selected_records[name]
                if (
                    info["shape"] != expected["shape"]
                    or info["dtype"] != expected["dtype"]
                ):
                    raise ValueError(f"Output tensor shape/dtype mismatch: {name}")
                start, end = info["data_offsets"]
                stream.seek(data_start + start)
                digest, remaining = hashlib.sha256(), end - start
                while remaining:
                    block = stream.read(min(8 << 20, remaining))
                    if not block:
                        raise ValueError(f"Truncated output tensor: {name}")
                    digest.update(block)
                    remaining -= len(block)
                if digest.hexdigest() != expected["sha256"]:
                    raise ValueError(f"Output content mismatch: {name}")
        output_files[output_name] = {
            "size": (output / output_name).stat().st_size,
            "sha256": sha256_file(output / output_name),
        }
        print(f"Saved {number}/{len(plan)}: {len(names)} selected tensors", flush=True)
    output_index = {"metadata": {"total_size": selected_size}, "weight_map": output_map}
    write_json(output / index_path.name, output_index)
    if index_path.name != CANONICAL_INDEX:
        write_json(output / CANONICAL_INDEX, output_index)
    write_json(output / "config.json", config)
    filtered_quant = {key: value for key, value in quant.items() if selected(key)}
    write_json(output / "quant_model_description.json", filtered_quant)
    resources = copy_resources(source, output, max_resource_bytes)
    for filename, fingerprint in source_fingerprints.items():
        if header(source / filename)[2] != fingerprint:
            raise ValueError(f"Source shard changed during preparation: {filename}")
    for filename, fingerprint in source_small.items():
        if sha256_file(source / filename) != fingerprint:
            raise ValueError(f"Source metadata changed during preparation: {filename}")
    manifest = {
        "status": "complete",
        "purpose": "four-layer operator-share diagnostic, not full-model performance",
        "source": str(source),
        "output": str(output),
        "source_index": index_path.name,
        "vllm_index": CANONICAL_INDEX,
        "tensor_framework": tensor_framework,
        "source_fingerprint_scope": "all shard stat+header hashes; config/index/quant SHA256; no full source payload hash",
        "source_metadata_sha256": source_small,
        "source_shards": source_fingerprints,
        "source_layer_count": original_layers,
        "selected_layer_ids_zero_based": [0, 1, 2, 3],
        "kda_layers_one_based": [1, 2, 3],
        "dsa_layers_one_based": [4],
        "attn_res_block_size": 12,
        "source_tensor_count": len(weight_map),
        "source_tensor_bytes": source_tensor_bytes,
        "selected_tensor_count": len(selected_records),
        "selected_tensor_bytes": selected_size,
        "dropped_tensor_count": len(weight_map) - len(selected_records),
        "retained_non_decoder_tensors": sum(
            record["layer"] is None for record in selected_records.values()
        ),
        "shard_target_bytes": shard_bytes,
        "largest_selected_tensor_bytes": max(
            record["bytes"] for record in selected_records.values()
        ),
        "all_selected_tensor_content_verified": True,
        "source_stat_and_headers_unchanged": True,
        "output_shards": output_files,
        "output_config_sha256": sha256_file(output / "config.json"),
        "output_quant_sha256": sha256_file(output / "quant_model_description.json"),
        "copied_resources": resources,
        "selected_tensors": selected_records,
    }
    write_json(output / MANIFEST, manifest)
    (output / "PREPARING.json").unlink()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--index", help="Source index basename; auto-discovers *.safetensors.index.json"
    )
    parser.add_argument("--shard-mib", type=int, default=512)
    parser.add_argument(
        "--tensor-framework",
        choices=("pt", "np"),
        default="pt",
        help="pt for the real BF16 checkpoint; np only for CPU toy fixtures",
    )
    args = parser.parse_args()
    try:
        manifest = build_checkpoint(
            args.source,
            args.output,
            index_name=args.index,
            shard_bytes=args.shard_mib << 20,
            tensor_framework=args.tensor_framework,
        )
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "status",
                    "source_tensor_count",
                    "selected_tensor_count",
                    "selected_tensor_bytes",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
