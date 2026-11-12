"""CPU toy fixtures only; these are not model/NPU validation results."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from prepare_four_layer_checkpoint import build_checkpoint, clipped_config, header


class FourLayerCheckpointTests(unittest.TestCase):
    def make_fixture(self, source):
        source.mkdir()
        config = {
            "model_type": "kimi_k31",
            "text_config": {
                "num_hidden_layers": 6,
                "num_experts": 32,
                "attn_res_block_size": 12,
                "hidden_size": 8,
                "untouched": {"keep": [7, 9]},
                "linear_attn_config": {
                    "kda_layers": [1, 2, 3, 5, 6],
                    "full_attn_layers": [4],
                    "use_full_rank_gate": True,
                },
                "dsa_config": {"dsa_layers": [4], "index_topk": 2048},
            },
            "vision_config": {"vt_num_hidden_layers": 27},
        }
        tensors = {
            f"model.language_model.layers.{i}.self_attn.weight": np.arange(
                8, dtype=np.float32
            ).reshape(2, 4)
            + i
            for i in range(6)
        }
        tensors.update(
            {
                "model.language_model.embed_tokens.weight": np.arange(8, dtype=np.int8),
                "model.language_model.norm.weight": np.ones(8, dtype=np.float32),
                "model.language_model.output_attn_res_proj.weight": np.ones(
                    (1, 8), dtype=np.float32
                ),
                "lm_head.weight": np.ones((2, 8), dtype=np.float32),
                # Vision layer number must not be treated as a decoder layer.
                "vision_tower.layers.20.weight": np.ones(2, dtype=np.float32),
            }
        )
        # Both source shards deliberately mix retained and excluded decoder layers.
        groups = [list(tensors)[::2], list(tensors)[1::2]]
        weight_map = {}
        for i, keys in enumerate(groups):
            name = f"quant-{i}.safetensors"
            save_file({key: tensors[key] for key in keys}, str(source / name))
            weight_map.update(dict.fromkeys(keys, name))
        index = {
            "metadata": {"total_size": sum(x.nbytes for x in tensors.values())},
            "weight_map": weight_map,
        }
        quant = {key: "W4A8" for key in tensors}
        quant["optional"] = {"unchanged": True}
        quant["fa_quant_type"] = "C8"
        for name, value in (
            ("config.json", config),
            ("quant_model_weights.safetensors.index.json", index),
            ("quant_model_description.json", quant),
        ):
            (source / name).write_text(json.dumps(value), encoding="utf-8")
        (source / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        # This is intentionally invalid: the reducer must never deserialize/copy it.
        (source / "unused.pt").write_bytes(b"not a pickle")
        return config, tensors

    @staticmethod
    def hashes(source):
        return {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()
        }

    def test_mixed_shards_content_config_index_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "source", Path(temp) / "output"
            config, tensors = self.make_fixture(source)
            before = self.hashes(source)
            manifest = build_checkpoint(
                source, output, shard_bytes=48, tensor_framework="np"
            )
            self.assertEqual(before, self.hashes(source))
            self.assertEqual(manifest["selected_layer_ids_zero_based"], [0, 1, 2, 3])
            self.assertEqual(manifest["dropped_tensor_count"], 2)
            self.assertTrue(manifest["all_selected_tensor_content_verified"])
            self.assertFalse((output / "PREPARING.json").exists())
            self.assertFalse((output / "unused.pt").exists())
            self.assertTrue((output / "tokenizer_config.json").exists())
            expected_config = copy.deepcopy(config)
            text = expected_config["text_config"]
            text["num_hidden_layers"] = 4
            text["linear_attn_config"]["kda_layers"] = [1, 2, 3]
            self.assertEqual(
                json.loads((output / "config.json").read_text()), expected_config
            )
            index = json.loads((output / "model.safetensors.index.json").read_text())
            alias = json.loads(
                (output / "quant_model_weights.safetensors.index.json").read_text()
            )
            self.assertEqual(index, alias)
            expected = {
                key: value
                for key, value in tensors.items()
                if ".layers.4." not in key and ".layers.5." not in key
            }
            self.assertEqual(set(index["weight_map"]), set(expected))
            self.assertEqual(
                index["metadata"]["total_size"],
                sum(x.nbytes for x in expected.values()),
            )
            found = {}
            for path in output.glob("*.safetensors"):
                info, _, _ = header(path)
                with safe_open(str(path), framework="np") as reader:
                    self.assertEqual(set(reader.keys()), set(info))
                    found.update({key: reader.get_tensor(key) for key in reader.keys()})
            self.assertEqual(set(found), set(expected))
            for key, value in expected.items():
                np.testing.assert_array_equal(found[key], value)
            quant = json.loads((output / "quant_model_description.json").read_text())
            self.assertNotIn("model.language_model.layers.4.self_attn.weight", quant)
            self.assertEqual(quant["optional"], {"unchanged": True})
            self.assertIn("vision_tower.layers.20.weight", quant)

    def test_no_overwrite_and_output_inside_source_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            self.make_fixture(source)
            with self.assertRaises(ValueError):
                build_checkpoint(source, source / "mini", tensor_framework="np")
            output = Path(temp) / "existing"
            output.mkdir()
            with self.assertRaises(ValueError):
                build_checkpoint(source, output, tensor_framework="np")
            self.assertEqual(list(output.iterdir()), [])

    def test_bad_header_index_and_prefix_fail_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "source", Path(temp) / "output"
            config, _ = self.make_fixture(source)
            bad = copy.deepcopy(config)
            bad["text_config"]["linear_attn_config"]["kda_layers"] = [1, 2, 4, 5, 6]
            with self.assertRaises(ValueError):
                clipped_config(bad)
            index_path = source / "quant_model_weights.safetensors.index.json"
            index = json.loads(index_path.read_text())
            index["weight_map"].pop(next(iter(index["weight_map"])))
            index_path.write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Index/header tensor mismatch"):
                build_checkpoint(source, output, tensor_framework="np")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
