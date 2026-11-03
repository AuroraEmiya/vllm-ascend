# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks that the trial cannot silently change its pinned workload."""

import importlib.util
import json
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


if __name__ == "__main__":
    unittest.main()
