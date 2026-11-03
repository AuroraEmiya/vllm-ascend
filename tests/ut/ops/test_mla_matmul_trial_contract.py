# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks that the trial cannot silently change its pinned workload."""

import importlib.util
import json
import random
import struct
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HARNESS = ROOT / "benchmarks/kernels/mla_prolog/matmul_trial.py"
SPEC = importlib.util.spec_from_file_location("mla_matmul_contract", HARNESS)
TRIAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRIAL)


class TestMlaMatmulTrialContract(unittest.TestCase):
    def test_original_wrapper_and_manifest_are_pinned(self):
        pinned = TRIAL.pinned_inputs()
        self.assertEqual(pinned["baseline_commit"], "af0d582472abb826d3ea6cbe37c75e7f49f9481c")
        cases = pinned["manifest"]["cases"]
        self.assertEqual(len(cases), 15)
        self.assertEqual(sum(c["role"] == "acceptance" for c in cases), 8)
        self.assertEqual(sum(c["role"] != "acceptance" for c in cases), 7)

    def test_up_nd_plan_uses_exact_retained_round7_control(self):
        result = subprocess.run(
            [sys.executable, str(HARNESS), "--plan", "--comparison", "up-nd-row-major", "--mode", "paired"],
            check=True,
            capture_output=True,
            text=True,
        )
        plan = json.loads(result.stdout)
        self.assertEqual(plan["protocol"]["new_budget_round"], 9)
        self.assertEqual(plan["pinned_baseline"]["baseline_commit"], TRIAL.NORM_OVERLAP_BASE_COMMIT)
        self.assertEqual(plan["pinned_baseline"]["baseline_canonical_sha256"], TRIAL.NORM_OVERLAP_BASE_SHA256)
        self.assertEqual(len(plan["selected_cases"]), 15)
        self.assertEqual(plan["protocol"]["calls_per_graph"], 10)
        self.assertEqual(plan["protocol"]["paired_ab_ba_samples"], 20)
        self.assertIn("mode8 is never selected", plan["protocol"]["native_comparison"])
        self.assertIn("no native rebuild", plan["protocol"]["native_comparison"])

    def test_subset_keeps_frozen_manifest_order(self):
        manifest = TRIAL.pinned_inputs()["manifest"]
        result = TRIAL.select_cases(manifest, ["decode-b128", "decode-b1", "decode-b1"])
        self.assertEqual([c["id"] for c in result], ["decode-b1", "decode-b128"])

    def test_v3_case_id_is_rejected(self):
        manifest = TRIAL.pinned_inputs()["manifest"]
        with self.assertRaisesRegex(ValueError, "unknown frozen case IDs"):
            TRIAL.select_cases(manifest, ["glm-b1"])

    def test_cli_plan_needs_no_torch_or_npu(self):
        command = [sys.executable, str(HARNESS), "--plan", "--mode", "paired", "--case-id", "guard-tail-b31"]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        plan = json.loads(result.stdout)
        self.assertEqual([c["id"] for c in plan["selected_cases"]], ["guard-tail-b31"])
        self.assertEqual(plan["protocol"]["calls_per_graph"], 10)
        self.assertEqual(plan["protocol"]["paired_ab_ba_samples"], 20)
        self.assertTrue(plan["protocol"]["no_new_V3_domain"])
        self.assertIsNone(plan["identity"])

    def test_profile_requires_one_case(self):
        result = subprocess.run(
            [sys.executable, str(HARNESS), "--plan", "--mode", "profile"], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("exactly one frozen --case-id", result.stderr)

    def test_head_comparison_has_its_own_pinned_control(self):
        pinned = TRIAL.pinned_inputs("head-layout")
        self.assertEqual(pinned["baseline_commit"], "592d3963e3ef59fc147ef88ab0f114f35b731776")
        self.assertEqual(pinned["manifest_canonical_sha256"], TRIAL.pinned_inputs()["manifest_canonical_sha256"])
        result = subprocess.run(
            [
                sys.executable,
                str(HARNESS),
                "--plan",
                "--comparison",
                "head-layout",
                "--mode",
                "head-micro",
                "--case-id",
                "decode-b128",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        plan = json.loads(result.stdout)
        self.assertEqual(plan["protocol"]["new_budget_round"], 2)
        self.assertEqual(plan["protocol"]["calls_per_graph"], 10)
        self.assertEqual(plan["protocol"]["paired_ab_ba_samples"], 20)
        self.assertEqual([c["id"] for c in plan["selected_cases"]], ["decode-b128"])

    def test_micro_modes_cannot_silently_select_the_other_comparison(self):
        result = subprocess.run(
            [sys.executable, str(HARNESS), "--plan", "--mode", "head-micro"], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("head-micro requires head-layout", result.stderr)

    def test_gmm_plan_pins_round1_instead_of_the_rejected_round2(self):
        result = subprocess.run(
            [
                sys.executable,
                str(HARNESS),
                "--plan",
                "--comparison",
                "gmm-down",
                "--mode",
                "micro",
                "--case-id",
                "decode-b16",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        plan = json.loads(result.stdout)
        self.assertEqual(plan["pinned_baseline"]["baseline_commit"], "592d3963e3ef59fc147ef88ab0f114f35b731776")
        self.assertEqual(plan["protocol"]["new_budget_round"], 3)
        self.assertEqual(plan["protocol"]["calls_per_graph"], 10)
        self.assertEqual(plan["protocol"]["paired_ab_ba_samples"], 20)
        self.assertTrue(plan["protocol"]["no_new_V3_domain"])

    def test_norm_plan_keeps_distinct_same_binary_control_algorithm(self):
        result = subprocess.run(
            [
                sys.executable,
                str(HARNESS),
                "--plan",
                "--comparison",
                "norm-broadcast",
                "--mode",
                "paired",
                "--case-id",
                "decode-b128",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        plan = json.loads(result.stdout)
        self.assertEqual(plan["protocol"]["new_budget_round"], 4)
        self.assertEqual(plan["pinned_baseline"]["baseline_commit"], "592d3963e3ef59fc147ef88ab0f114f35b731776")
        self.assertIn("mode3", plan["protocol"]["native_comparison"])
        self.assertIn("mode4", plan["protocol"]["native_comparison"])
        self.assertEqual(plan["protocol"]["calls_per_graph"], 10)
        self.assertEqual(plan["protocol"]["paired_ab_ba_samples"], 20)

    def test_inverse_broadcast_addressing_reads_only_the_first_block(self):
        # Brcb<float> expands each of eight source scalars to a 32-byte block.
        # Mul mask64, src1BlkStride=src1RepStride=0 must read only block0,
        # including the final repeat of both supported RMS widths.
        source = [0.125] + [1000 + lane for lane in range(1, 8)]
        broadcast = [value for value in source for _ in range(8)]
        for width in (512, 1536):
            destinations, coefficients = [], []
            for repeat in range(width // 64):
                for block in range(8):
                    for lane in range(8):
                        destinations.append(repeat * 64 + block * 8 + lane)
                        coefficients.append(broadcast[0 * repeat + 0 * block + lane])
            self.assertEqual(destinations, list(range(width)))
            self.assertEqual(coefficients, [source[0]] * width)

    def test_gamma_cache_plan_pins_round4_control(self):
        result = subprocess.run(
            [
                sys.executable,
                str(HARNESS),
                "--plan",
                "--comparison",
                "norm-gamma-cache",
                "--mode",
                "paired",
                "--case-id",
                "decode-b128",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        plan = json.loads(result.stdout)
        self.assertEqual(plan["protocol"]["new_budget_round"], 5)
        self.assertEqual(plan["pinned_baseline"]["baseline_commit"], "8b0d4da8d4897b3099736689c63ae7baea1dff3e")
        self.assertEqual(
            plan["pinned_baseline"]["baseline_canonical_sha256"],
            "cc2cacc7d07beb06138643d1aa868f0ef0865b5166a574b94be7d95df7e734fc",
        )
        self.assertIn("B>=32", plan["protocol"]["changed"])
        self.assertIn("mode4", plan["protocol"]["native_comparison"])
        self.assertIn("mode5", plan["protocol"]["native_comparison"])

    def test_cached_gamma_region_survives_rope_and_width_changes(self):
        # f1 has2048 FP32 elements. The first64 are transient RoPE sine;
        # cached gamma starts at64 and is tagged by width. Include odd core
        # counts, where a core can change query/KV type, and a second launch
        # with updated gamma values. No tag may persist across launches.
        def launch(rows, cores, generation):
            loads = 0
            blocks = min(2 * rows, cores)
            for core in range(blocks):
                cached_width = 0
                buffer = [None] * 2048
                for unit in range(core, 2 * rows, blocks):
                    width = 1536 if unit % 2 == 0 else 512
                    gamma = [generation * 10000 + width + lane for lane in range(width)]
                    if cached_width != width:
                        buffer[64 : 64 + width] = gamma
                        cached_width = width
                        loads += width * 2
                    self.assertEqual(buffer[64 : 64 + width], gamma)
                    self.assertEqual(len(buffer), 2048)
                    if unit % 2:
                        buffer[:64] = [-unit] * 64
                        self.assertEqual(buffer[64 : 64 + width], gamma)
            return loads

        for rows in (32, 33, 64, 128):
            for generation in (0, 1):
                self.assertEqual(launch(rows, 40, generation), 81920)
                self.assertLess(launch(rows, 40, generation), rows * 4096)
                for cores in (1, 39):
                    self.assertEqual(launch(rows, cores, generation), rows * 4096)

    def test_split_overlap_plan_pins_round5_control(self):
        result = subprocess.run(
            [
                sys.executable,
                str(HARNESS),
                "--plan",
                "--comparison",
                "split-dma-overlap",
                "--mode",
                "paired",
                "--case-id",
                "guard-tail-b33",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        plan = json.loads(result.stdout)
        self.assertEqual(plan["protocol"]["new_budget_round"], 6)
        self.assertEqual(plan["pinned_baseline"]["baseline_commit"], "5e35db3be11162a6edcb06e131d8dbaae7a2948e")
        self.assertEqual(
            plan["pinned_baseline"]["baseline_canonical_sha256"],
            "ff46e1fa24ed70794a191aaee14bdb68fa5512bffab0f09eedbe46ea9b254b42",
        )
        self.assertEqual([case["id"] for case in plan["selected_cases"]], ["guard-tail-b33"])
        self.assertIn("B>=32", plan["protocol"]["changed"])
        self.assertIn("mode1", plan["protocol"]["native_comparison"])
        self.assertIn("mode6", plan["protocol"]["native_comparison"])

    def test_coefficient_pack_plan_pins_exact_round7_control(self):
        pinned = TRIAL.pinned_inputs("split-coefficient-pack")
        self.assertEqual(pinned["baseline_commit"], "7ada452c33c5748fe141a786250c2b30345b637f")
        self.assertEqual(
            pinned["baseline_canonical_sha256"],
            "1030da1293222b46d337840a496acc87d0bae152c532007121368da507096d56",
        )
        self.assertNotIn("baseline_derivation", pinned)
        result = subprocess.run(
            [sys.executable, str(HARNESS), "--plan", "--comparison", "split-coefficient-pack", "--mode", "paired"],
            check=True,
            capture_output=True,
            text=True,
        )
        plan = json.loads(result.stdout)
        self.assertEqual(plan["protocol"]["new_budget_round"], 8)
        self.assertEqual(len(plan["selected_cases"]), 15)
        self.assertEqual(plan["protocol"]["calls_per_graph"], 10)
        self.assertEqual(plan["protocol"]["paired_ab_ba_samples"], 20)

    def test_packed_coefficients_cover_legal_head_tiles_and_keep_fp32_operation_order(self):
        def fp32(value):
            return struct.unpack("<f", struct.pack("<f", value))[0]

        generator = random.Random(718032)
        sine = [fp32(generator.uniform(-1, 1)) for _ in range(64)]
        cosine = [fp32(generator.uniform(-1, 1)) for _ in range(64)]
        packed = [777.0] * 2048
        halves = (cosine[:32], cosine[32:], sine[:32], sine[32:])
        for index, half in enumerate(halves):
            packed[index * 64 : (index + 1) * 64] = [fp32(value + 0.0) for value in half] * 2
        self.assertEqual(packed[256:], [777.0] * (2048 - 256))

        def product(values, offset, heads):
            if heads not in (1, 2, 4, 8, 16, 32):
                raise ValueError("outside the fixed power-of-two head contract")
            mask, repeats = (32, 1) if heads == 1 else (64, heads // 2)
            result = [None] * (heads * 32)
            for repeat in range(repeats):
                for lane in range(mask):
                    destination = repeat * 64 + lane
                    result[destination] = fp32(values[destination] * packed[offset + lane])
            self.assertNotIn(None, result)
            return result

        for heads in (1, 2, 4, 8, 16, 32):
            original = [[fp32(generator.uniform(-4, 4)) for _ in range(64)] for _ in range(heads)]
            even = [value for head in original for value in head[::2]]
            odd = [value for head in original for value in head[1::2]]
            cos_even, cos_odd = product(even, 0, heads), product(odd, 64, heads)
            sin_odd, sin_even = product(odd, 128, heads), product(even, 192, heads)
            actual = [fp32(c - s) for c, s in zip(cos_even, sin_odd)]
            actual += [fp32(c + s) for c, s in zip(cos_odd, sin_even)]
            expected_even, expected_odd = [], []
            for head in original:
                for lane in range(32):
                    expected_even.append(
                        fp32(fp32(head[2 * lane] * cosine[lane]) - fp32(head[2 * lane + 1] * sine[lane]))
                    )
                    expected_odd.append(
                        fp32(fp32(head[2 * lane + 1] * cosine[32 + lane]) + fp32(head[2 * lane] * sine[32 + lane]))
                    )
            self.assertEqual(
                b"".join(struct.pack("<f", value) for value in actual),
                b"".join(struct.pack("<f", value) for value in expected_even + expected_odd),
            )
        # An odd legal tail is one head. Other odd head counts are rejected
        # by the existing contract; the trial does not extend that domain.
        for heads in (3, 5, 31):
            with self.assertRaisesRegex(ValueError, "fixed power-of-two"):
                product([0.0] * (heads * 32), 0, heads)

    def test_packed_rope_barriers_order_gather_reuse_and_all_true_vector_dependencies(self):
        def hazards(missing=None):
            coefficients = ("CE", "CO", "SO", "SE")
            reads_f0 = {"f0-" + name: "R" for name in coefficients}
            phases = [
                [("gather-E", {**reads_f0, "f3-E": "W"}), ("gather-O", {**reads_f0, "f3-O": "W"})],
                [("pack-" + name, {"f0-" + name: "W"}) for name in coefficients],
                [
                    ("cos-E", {"f3-E": "R", "f0-CE": "R", "f4-E": "W"}),
                    ("cos-O", {"f3-O": "R", "f0-CO": "R", "f4-O": "W"}),
                ],
                [("sin-O", {"f3-O": "W", "f0-SO": "R"}), ("sin-E", {"f3-E": "W", "f0-SE": "R"})],
                [("sub-E", {"f4-E": "W", "f3-O": "R"}), ("add-O", {"f4-O": "W", "f3-E": "R"})],
                [("output-cast", {"f4-E": "R", "f4-O": "R", "b3": "W"})],
            ]
            operations, before = {}, {}
            earlier_fence = set()
            for index, phase in enumerate(phases):
                dependencies = set(earlier_fence)
                if index and index != missing:
                    dependencies.update(name for name, _ in phases[index - 1])
                for name, accesses in phase:
                    operations[name] = accesses
                    before[name] = set(dependencies)
                earlier_fence = dependencies
            for name in before:
                pending = list(before[name])
                while pending:
                    predecessor = pending.pop()
                    for ancestor in before[predecessor] - before[name]:
                        before[name].add(ancestor)
                        pending.append(ancestor)
            conflicts = []
            names = list(operations)
            for index, left in enumerate(names):
                for right in names[index + 1 :]:
                    shared = operations[left].keys() & operations[right].keys()
                    if any("W" in (operations[left][buffer], operations[right][buffer]) for buffer in shared):
                        if left not in before[right] and right not in before[left]:
                            conflicts.append((left, right))
            return conflicts, before

        conflicts, before = hazards()
        self.assertEqual(conflicts, [])
        self.assertNotIn("cos-E", before["cos-O"])
        self.assertNotIn("sin-E", before["sin-O"])
        self.assertNotIn("sub-E", before["add-O"])
        for barrier in range(1, 6):
            self.assertTrue(hazards(missing=barrier)[0], f"missing barrier{barrier} must expose a hazard")

    def test_norm_output_overlap_pins_the_retained_round6_control(self):
        result = subprocess.run(
            [
                sys.executable,
                str(HARNESS),
                "--plan",
                "--comparison",
                "norm-output-overlap",
                "--mode",
                "paired",
                "--case-id",
                "decode-b128",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        plan = json.loads(result.stdout)
        pinned = plan["pinned_baseline"]
        self.assertEqual(plan["protocol"]["new_budget_round"], 7)
        self.assertEqual(pinned["baseline_commit"], "f1f0dba2d1202579b8bea9b2d03f480d7fb003ed")
        self.assertEqual(
            pinned["baseline_canonical_sha256"],
            "a2c2e333f94e9e62f3f13bdaa644fc1658cc9fffbf3777a71061d27f23d1fc44",
        )
        derivation = pinned["baseline_derivation"]
        self.assertEqual(
            derivation["source_wrapper_sha256"],
            "21279de14dfd20af8078b15c680982104de9b29aaee67fbee2cd9cabe8dd6508",
        )
        self.assertIn("32<=B<=64", derivation["change"])
        self.assertTrue(derivation["same_change_in_candidate"])
        self.assertTrue(derivation["rollback_is_not_new_gain"])
        self.assertEqual(plan["protocol"]["calls_per_graph"], 10)
        self.assertEqual(plan["protocol"]["paired_ab_ba_samples"], 20)

    def test_norm_overlap_orders_real_buffer_reuse_without_blocking_disjoint_input(self):
        # Model a core's norm / KV-to-KR / next-unit phases. Each FIFO engine
        # has its own order; only explicit events create cross-engine edges.
        # Removing either reuse event must expose the corresponding hazard.
        def hazards(input_reuse=True, output_reuse=True):
            operations, before = {}, {}

            def add(name, accesses, dependencies):
                operations[name] = accesses
                before[name] = set(dependencies)

            previous_input = previous_vector = previous_output = None
            for phase in range(4):
                prefix = str(phase)
                dependencies = [previous_input] if previous_input else []
                if input_reuse and previous_vector:
                    dependencies.append(previous_vector)
                add(prefix + "-in", {"b0": "W"}, dependencies)
                add(
                    prefix + "-cast",
                    {"b0": "R", "f0": "W"},
                    [prefix + "-in"] + ([previous_vector] if previous_vector else []),
                )
                add(prefix + "-math", {"f0": "W"}, [prefix + "-cast"])
                dependencies = [prefix + "-math"]
                if output_reuse and previous_output:
                    dependencies.append(previous_output)
                add(prefix + "-out-cast", {"f0": "R", "b3": "W"}, dependencies)
                add(
                    prefix + "-out",
                    {"b3": "R"},
                    [prefix + "-out-cast"] + ([previous_output] if previous_output else []),
                )
                previous_input, previous_vector, previous_output = (
                    prefix + "-in",
                    prefix + "-out-cast",
                    prefix + "-out",
                )
            add("final-drain", {}, [previous_output])
            for name in before:
                pending = list(before[name])
                while pending:
                    predecessor = pending.pop()
                    for ancestor in before[predecessor] - before[name]:
                        before[name].add(ancestor)
                        pending.append(ancestor)
            conflicts = []
            names = list(operations)
            for index, left in enumerate(names):
                for right in names[index + 1 :]:
                    shared = operations[left].keys() & operations[right].keys()
                    writes = [b for b in shared if "W" in (operations[left][b], operations[right][b])]
                    if writes and left not in before[right] and right not in before[left]:
                        conflicts.append((left, right, writes))
            return conflicts, before

        conflicts, before = hazards()
        self.assertEqual(conflicts, [])
        self.assertNotIn("0-out", before["1-in"])
        self.assertIn("0-out-cast", before["1-in"])
        self.assertIn("0-out", before["1-out-cast"])
        self.assertIn("3-out", before["final-drain"])
        self.assertTrue(any("b0" in item[2] for item in hazards(input_reuse=False)[0]))
        self.assertTrue(any("b3" in item[2] for item in hazards(output_reuse=False)[0]))

    def test_split_pipeline_buffer_dependencies_include_reuse_and_last_output(self):
        # Abstract completed operations on two consecutive units of one core.
        # Every conflicting access must be ordered, even when DMA latency is
        # adversarial. This checks the lifetime model, not compiled device code.
        def hazards(rope_buffer="b3", reuse_fence=True):
            operations, before = {}, {}

            def add(name, buffer, write, dependencies):
                operations[name] = (buffer, write)
                before[name] = set(dependencies)

            for unit in range(2):
                prefix = str(unit)
                previous = ["0-q-out", "0-r-out", "0-r-cast"] if unit and reuse_fence else []
                add(prefix + "-q-in", "b0", True, previous)
                add(prefix + "-q-out", "b0", False, [prefix + "-q-in"])
                add(prefix + "-r-in", rope_buffer, True, [prefix + "-q-in"])
                add(prefix + "-r-read", rope_buffer, False, [prefix + "-r-in"])
                add(prefix + "-r-cast", "b3", True, [prefix + "-r-read"])
                add(prefix + "-r-out", "b3", False, [prefix + "-q-out", prefix + "-r-cast"])
            add("last-drain", None, False, ["1-q-out", "1-r-out"])
            changed = True
            while changed:
                changed = False
                for predecessors in before.values():
                    closure = set(predecessors)
                    for predecessor in predecessors:
                        closure.update(before[predecessor])
                    if closure != predecessors:
                        predecessors.update(closure)
                        changed = True
            conflicts = []
            for left, (left_buffer, left_write) in operations.items():
                for right, (right_buffer, right_write) in operations.items():
                    if left >= right or left_buffer is None or left_buffer != right_buffer:
                        continue
                    if left_write or right_write:
                        if left not in before[right] and right not in before[left]:
                            conflicts.append((left, right))
            return conflicts, before

        conflicts, before = hazards()
        self.assertEqual(conflicts, [])
        # The intended overlap has no artificial Q-out -> RoPE-in dependency.
        self.assertNotIn("0-q-out", before["0-r-in"])
        self.assertIn("0-r-out", before["1-q-in"])
        self.assertIn("1-r-out", before["last-drain"])
        # Negative controls must detect both the old alias and lost reuse fence.
        self.assertTrue(hazards(rope_buffer="b0")[0])
        self.assertTrue(hazards(reuse_fence=False)[0])


if __name__ == "__main__":
    unittest.main()
