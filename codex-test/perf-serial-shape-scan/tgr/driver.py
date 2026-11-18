import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import time

import adapter as a

HERE = Path(__file__).resolve().parent
TGR = HERE.parent
ROOT = TGR.parent
SOURCE = TGR / "checkout"
SEED = 20261118
BASE = TGR / "baseline/vendors/custom_transformer/op_api/lib/libcust_opapi.so"
LIB = SOURCE / "vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so"
CAND = SOURCE / "vllm_ascend/_cann_ops_custom/vendors/serial_transformer/op_api/lib/libcust_opapi.so"
MATRIX = SOURCE / "vllm_ascend/libvllm_ascend_kernels.so"
EXPECTED = {
    BASE: "d8a1fd2917ef4a001b099b0625662859457e58de704e12253624a572cbf21e1b",
    LIB: "5848fe21daf465a505b7d9c5103092807707f6d5b94db95ac4183987b14af89d",
    CAND: "b5f09a5971516a962d9487e51a7ee1e45ca739ace391adcf487302f073b4aadc",
    MATRIX: "516c06ddc41793617ad930bde86d39bc5a69d1b74f54f954f78704434621e75a",
}
PROTOCOL = "tgr_direct_staged_eager_2p5_split_routes_v1"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def identity(torch, route):
    paths = {str(Path(line.split()[-1]).resolve()) for line in Path("/proc/self/maps").read_text().splitlines()
             if line.split()[-1].startswith("/")}
    selected = [BASE] if route == "baseline" else [LIB, CAND, MATRIX]
    actual = [{"path": str(p), "sha256": a.sha_file(p)} for p in selected]
    assert all(item["sha256"] == EXPECTED[p] and str(p.resolve()) in paths for p, item in zip(selected, actual))
    if route == "baseline":
        jit = [Path(p) for p in paths if Path(p).name == "text_gated_residual.so"]
        assert len(jit) == 1 and jit[0].is_relative_to(TGR / "torch-extensions")
        actual.append({"path": str(jit[0]), "sha256": a.sha_file(jit[0])})
    return actual


class Route:
    def __init__(self, torch, case, route):
        self.torch = torch
        self.case = case
        self.route = route
        self.scalar = a.normalize(case)
        for p in ([BASE] if route == "baseline" else [LIB, CAND, MATRIX]):
            assert a.sha_file(p) == EXPECTED[p]
        if route == "baseline":
            module = importlib.import_module(a.CONTRACT["baseline_module"])
            module.text_gated_residual_op_builder.load()
            expected = load_json(TGR / "runtime-schema.json")["schema"]
            assert str(torch.ops.cann_ops_transformer.text_gated_residual.default._schema) == expected
        else:
            import tgr_namespace
            tgr_namespace.install_serial_namespace(SOURCE)
            torch.ops.load_library(str(LIB))
            spec = importlib.util.spec_from_file_location("scan_tgr_vector_r5", SOURCE / a.CONTRACT["candidate_module"])
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        self.host = a.make_inputs(torch, case, SEED)
        self.d = {key: v.to("npu:0") for key, v in self.host.items()}
        if route == "candidate":
            self.owner = a.candidate_settings(module.TextGatedResidualSerial(
                self.d["w_down"], self.d["w_up"], self.d.get("w_gate"), self.scalar["channels"], self.scalar["eps"]))
        torch.npu.synchronize()
        self.digest = a.inputs_digest(torch, self.host, case)
        assert self.observation()["inputs_sha256"] == self.digest

    def call(self):
        if self.route == "baseline":
            return a.baseline_call(self.torch, self.d, self.scalar)
        return self.owner(self.d["x"], self.d.get("gamma"), self.scalar["need_backward"])

    def observation(self):
        return {"inputs_sha256": a.inputs_digest(self.torch, self.d, self.case),
                "aliases": {k: x.untyped_storage().data_ptr() for k, x in self.d.items()}}

    def change(self):
        self.host = a.make_inputs(self.torch, self.case, SEED + 1)
        for k, v in self.host.items():
            self.d[k].copy_(v)
        if self.route == "candidate":
            self.owner.refresh_weights_()
        self.torch.npu.synchronize()

    def snapshot(self, values):
        assert isinstance(values, (tuple, list)) and len(values) == 5
        ptrs = set(self.observation()["aliases"].values())
        return {"outputs": [x.detach().cpu().clone() for x in values],
                "inputs_disjoint": [x.numel() == 0 or x.untyped_storage().data_ptr() not in ptrs for x in values],
                "input_digest_after_call": a.inputs_digest(self.torch, self.d, self.case)}


def precision(session, out):
    phases = []
    for phase in a.PHASES:
        if phase == "changed_values_same_eager":
            session.change()
        print("TGR_BEGIN", session.case["case_id"], session.route, phase, flush=True)
        value = session.call()
        session.torch.npu.synchronize()
        snapshot = session.snapshot(value)
        phases.append({"phase": phase, "input": a.inputs_digest(session.torch, session.host, session.case),
                       "snapshot": snapshot})
    data = {"case": session.case, "route": session.route, "protocol": PROTOCOL, "phases": phases,
            "actual_libraries": identity(session.torch, session.route)}
    session.torch.save(data, out / "precision.pt")
    a.save(out / "route.json", {**{k: v for k, v in data.items() if k != "phases"},
        "phases": [{"phase": q["phase"], "input": q["input"]} for q in phases],
        "raw_sha256": a.sha_file(out / "precision.pt")})


def compare(torch, case, folder):
    result = {}
    for route in ("baseline", "candidate"):
        p = folder / route / "precision.pt"
        status = load_json(p.with_name("route.json"))
        assert p.is_file() and status["raw_sha256"] == a.sha_file(p)
        result[route] = torch.load(p, weights_only=True)
    assert all(x["case"] == case and x["protocol"] == PROTOCOL for x in result.values())
    phases = []
    for x, y in zip(result["baseline"]["phases"], result["candidate"]["phases"], strict=True):
        assert x["input"] == y["input"] and x["phase"] == y["phase"]
        checks = a.output_checks(torch, case, {"baseline": x["snapshot"], "candidate": y["snapshot"]}, x["input"])
        phases.append({"phase": x["phase"], "inputs_sha256": x["input"], "checks": checks})
    assert [x["phase"] for x in phases] == list(a.PHASES)
    assert phases[0]["inputs_sha256"] == phases[1]["inputs_sha256"] != phases[2]["inputs_sha256"]
    passed = all(x["passed"] for q in phases for x in q["checks"])
    record = {"case": case, "status": "PASS" if passed else "FAIL", "precision_rules": a.precision_rules(case),
              "phases": phases, "inputs_sha256": phases[0]["inputs_sha256"], "execution_mode": "eager",
              "route_libraries": {k: v["actual_libraries"] for k, v in result.items()}, "protocol": PROTOCOL}
    a.save(folder / "pair.json", record)
    print("TGR_PAIRED_PRECISION", case["case_id"], record["status"], flush=True)
    return 0 if passed else 4


def profile(session, folder, precision_path, started, physical):
    checked = load_json(precision_path)
    assert (checked["status"] == "PASS" and checked["case"] == session.case
            and checked["inputs_sha256"] == session.digest and checked["protocol"] == PROTOCOL)
    torch = session.torch
    warm = torch.zeros((128, 1024, 1024), dtype=torch.float16, device="npu:0")
    torch.npu.synchronize()
    original = session.observation()
    completed = []
    for i in range(7):
        start = time.time_ns()
        assert session.observation() == original
        torch.npu.synchronize()
        end = time.time_ns()
        w0 = time.time_ns()
        torch.argmax(warm)
        torch.npu.synchronize()
        w1 = time.time_ns()
        t0 = time.time_ns()
        output = session.call()
        torch.npu.synchronize()
        t1 = time.time_ns()
        completed.append({"ordinal": i, "phase": "prewarm" if i < 2 else "measurement",
            "phase_index": i if i < 2 else i-2, "observe_start_ns": start, "observe_end_ns": end,
            "warmup_start_ns": w0, "warmup_end_ns": w1, "target_start_ns": t0, "target_end_ns": t1})
    obs_start = time.time_ns()
    assert session.observation() == original
    session.torch.save(session.snapshot(output), folder / "final-outputs.pt")
    observed = identity(torch, session.route)
    assert observed == checked["route_libraries"][session.route]
    a.save(folder / "runtime-receipt.json", {"case_id": session.case["case_id"], "case": session.case,
        "route": session.route, "pid": os.getpid(), "physical_device": physical, "runtime_device_id": "0",
        "execution_mode": "eager", "graph_used": False, "protocol_id": PROTOCOL, "groups": completed,
        "controls": [{"phase": "preparation", "start_ns": started, "end_ns": completed[0]["observe_start_ns"]},
                     {"phase": "final_output_and_identity_observation", "start_ns": obs_start, "end_ns": time.time_ns()}],
        "inputs_sha256": session.digest, "libraries": observed,
        "precision_evidence_sha256": a.sha_file(precision_path),
        "final_outputs_sha256": a.sha_file(folder / "final-outputs.pt")})


def main():
    started = time.time_ns()
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["plan", "precision", "compare", "profile"])
    parser.add_argument("case", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--route", choices=["baseline", "candidate"])
    parser.add_argument("--physical", type=int)
    parser.add_argument("--precision", type=Path)
    args = parser.parse_args()
    case = load_json(args.case)
    assert case["scenario"] == "prefill" and case["allowed_execution_modes"] == ["eager"]
    a.normalize(case)
    if args.action == "plan":
        print(json.dumps(a.plan(case)))
        return 0
    import torch
    if args.action == "compare":
        return compare(torch, case, args.out)
    import torch_npu
    assert args.route and args.physical == 10
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.npu.matmul.allow_hf32 = False
    assert torch.npu.get_device_name(0) == "Ascend910_9362"
    args.out.mkdir(parents=True, exist_ok=False)
    session = Route(torch, case, args.route)
    if args.action == "precision":
        precision(session, args.out)
    else:
        profile(session, args.out, args.precision, started, args.physical)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
