"""Development run of staged decomposition with recorded inner and outer loops.

Uses supplied snapshots and never reads gold during inference. A completed run
means valid execution, not proven source correctness. Every failed case remains
in the scheduled denominator. Existing run directories cannot be overwritten.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import getpass
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import re
import shutil
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from loop_compare import load_inputs
from model_io import Budget, BudgetClient, write
from newsverify import provenance as p
from newsverify.decisions import present_decision, round_decisions
from staged_semantic import StagedDecomposer, StagedVerifier
from extended_semantic import PLAN_SCHEMA_VERSION, TargetPlanner, project_plan


class DevelopmentProvider:
    """Repeat eligible source versions to exercise the complete update contract."""
    def __init__(self, materials):
        self.materials = tuple(materials)
        self.history = []

    def search(self, target, tasks, round_number, limit):
        chosen = self.materials[:limit]
        self.history.append({"round": round_number,
            "kind": "fixed_corpus_development_reanalysis",
            "tasks": [asdict(t) for t in tasks],
            "returned": [m.version_id for m in chosen]})
        return iter(chosen)


def run(args):
    data = load_inputs(args.inputs)
    if args.cases:
        selected = set(args.cases.split(","))
        known = {c["target"]["id"] for c in data["cases"]}
        if not selected <= known:
            raise ValueError("Unknown selected development case")
        data = {**data, "cases": [c for c in data["cases"] if c["target"]["id"] in selected]}
    if not data["cases"]:
        raise ValueError("No development cases selected")
    if any(not isinstance(c["target"]["id"], str) or
           not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", c["target"]["id"])
           for c in data["cases"]):
        raise ValueError("Case IDs must be safe, bounded artifact filenames")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    budget = Budget(calls=args.max_calls, output_tokens=args.output_tokens,
                    per_call_output_tokens=args.per_call_tokens, seconds=args.seconds)
    trace_config = p.TraceConfig(max_rounds=args.rounds, max_documents=24,
        max_decomposition_calls=24, experimental_force_rounds=True)
    used_experiments = ["staged_run.py", "staged_semantic.py", "extended_semantic.py", "model_io.py",
                        "loop_compare.py", "semantic_adapter.py"]
    source_files = list((ROOT / "newsverify").glob("*.py")) + [ROOT / "experiments" / n for n in used_experiments]
    extension_enabled = bool(getattr(args, "target_extension", False))
    config = {"experiment": ("target_extended_psi_development_v2" if extension_enabled
                              else "staged_psi_development_v1"), "model": args.model,
        "reasoning_effort": args.reasoning_effort, "budget_per_case": asdict(budget),
        "trace_config": asdict(trace_config), "max_inner_repairs": args.max_repairs,
        "target_extension": extension_enabled,
        "target_plan_schema": PLAN_SCHEMA_VERSION if extension_enabled else None,
        "target_structure_repairs": args.max_repairs if extension_enabled else None,
        "target_extension_repairs": args.max_repairs if extension_enabled else None,
        "target_extension_output_repairs": args.max_repairs if extension_enabled else None,
        "material_stage_repairs": args.max_repairs,
        "judgement_repairs": args.max_repairs if extension_enabled else None,
        "probe_result_structure_repairs": args.max_repairs if extension_enabled else None,
        "target_extension_contract": ("one immutable target contract and decision-probe extension per case; "
            "one grounded result per routed probe; Python aggregates target logic; bounded target, "
            "result-structure and judgement repair; plan never counts as evidence" if extension_enabled else None),
        "workers": args.workers, "ordering_seed": 20260906,
        "case_ids": [c["target"]["id"] for c in data["cases"]],
        "scheduled_cases": len(data["cases"]), "gold_read_during_inference": False,
        "new_response_only": True, "automatic_transport_retries": 0,
        "dataset_status": "previously_seen_development_cases_not_hidden_benchmark",
        "loop_contract": "every material return uses atoms, lineage and critic; targeted bounded inner repair; forced outer rounds",
        "provider": "fixed eligible snapshots, no open-web collection",
        "input_sha256": hashlib.sha256(Path(args.inputs).read_bytes()).hexdigest(),
        "source_sha256": {f.relative_to(ROOT).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
                          for f in sorted(source_files)}}
    write(output / "config.json", config)
    write(output / "inputs.json", data)
    for source in source_files:
        dest = output / "executed-code" / source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    if not os.environ.get("OPENAI_API_KEY"):
        write(output / "status.json", {"status": "blocked_missing_auth", "new_api_attempts": 0})
        return 2

    def one(case):
        identifier = case["target"]["id"]

        class RecordedClient(BudgetClient):
            def __init__(self):
                super().__init__(args.model, budget, reasoning_effort=args.reasoning_effort)
                self.recording_lock = threading.Lock()

            def call(self, *a, **kw):
                with self.recording_lock:
                    try:
                        result = super().call(*a, **kw)
                        record = self.records[-1]
                        if record.get("actual_model") != args.model:
                            record.update(error_type="UnexpectedModel", error_code="unexpected_model")
                            raise ValueError("Returned model differs from configured model")
                        if record["usage"]["output_tokens"] > record["max_completion_tokens"]:
                            record.update(error_type="OutputCapExceeded", error_code="per_call_cap_exceeded")
                            raise ValueError("Observed completion cap exceeded")
                        return result
                    finally:
                        write(output / f"{identifier}-calls.json", self.records)
                        print(json.dumps({"case": identifier, "calls": self.calls}), flush=True)

        client = planner = plan = psi = verifier = provider = None
        row = {"id": identifier, "assessment_mode": case["target"]["assessment_mode"],
            "status": "error", "prediction": None, "checkpoints": []}
        hashes = []

        def capture(checkpoint):
            text = p.canonical_checkpoint_json(checkpoint)
            round_number = checkpoint.state["usage"]["rounds"]
            (output / f"{identifier}-checkpoint-{round_number}.json").write_text(text + "\n")
            hashes.append({"round": round_number, "sha256": p.checkpoint_sha256(checkpoint)})

        try:
            client = RecordedClient()
            target = p.Target(**{**case["target"], "evidence_scope": tuple(case["target"]["evidence_scope"])})
            materials = [p.MaterialVersion(**m) for m in case["materials"]]
            cutoff = p._time(target.as_of, "as_of")
            eligible = [m for m in materials if not p._material_eligibility(m, cutoff)]
            target_plan = None
            if extension_enabled:
                planner = TargetPlanner(
                    client,
                    max_repairs=args.max_repairs,
                    max_structure_repairs=args.max_repairs,
                    max_output_repairs=args.max_repairs,
                )
                plan = planner.prepare(target)
                target_plan = {stage: project_plan(plan, stage).to_payload()
                               for stage in ("atoms", "lineage", "critic", "evidence", "world")}
                write(output / f"{identifier}-target-plan.json",
                      {**plan.to_payload(), "sha256": plan.sha256, "projections": target_plan})
                row["target_plan_sha256"] = plan.sha256
            psi = StagedDecomposer(client, max_repairs=args.max_repairs, target_plan=target_plan)
            verifier = StagedVerifier(
                client,
                target_plan=target_plan,
                max_repairs=args.max_repairs,
                max_structure_repairs=args.max_repairs,
            )
            provider = DevelopmentProvider(eligible)
            report = p.run_provenance(target, provider, psi, verifier, trace_config,
                                      checkpoint_callback=capture)
            write(output / f"{identifier}-report.json", report)
            row["checkpoints"] = round_decisions(report)
            row["native_prediction"] = present_decision(report)
            row["engine_usage"] = report["usage"]
            if not row["native_prediction"]["assessment_valid"]:
                raise ValueError("Engine did not accept a complete valid assessment")
            if report["usage"]["verification_calls"] != args.rounds:
                raise ValueError("Required actual outer verification rounds were not completed")
            if any(r.get("error_type") for r in client.records):
                raise ValueError("A failed model call cannot be ignored")
            row["prediction"] = row["native_prediction"]["decision"]
            row["status"] = "completed"
        except Exception as exc:
            row["error_type"] = type(exc).__name__
            if getattr(exc, "stage", None):
                row["error_stage"] = exc.stage
        finally:
            row["usage"] = client.usage() if client else {k: 0 for k in
                ("model_calls", "input_tokens", "output_tokens", "seconds")}
            row["checkpoint_hashes"] = hashes
            records = client.records if client else []
            row["new_api_attempts"] = len(records)
            row["returned_responses"] = sum(bool(r.get("response_id")) for r in records)
            write(output / f"{identifier}-psi-history.json", getattr(psi, "history", []))
            write(output / f"{identifier}-verification-history.json", getattr(verifier, "history", []))
            write(output / f"{identifier}-target-planner-history.json", getattr(planner, "history", []))
            write(output / f"{identifier}-retrieval.json", getattr(provider, "history", []))
            write(output / f"{identifier}-result.json", row)
            print(json.dumps({"case": identifier, "status": row["status"]}), flush=True)
        return row

    cases = list(data["cases"])
    random.Random(20260906).shuffle(cases)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, case): case["target"]["id"] for case in cases}
        for future in as_completed(futures):
            rows.append(future.result())
            write(output / "results.json", sorted(rows, key=lambda r: r["id"]))
    write(output / "status.json", {"status": "completed" if all(r["status"] == "completed" for r in rows) else "has_errors",
        "scheduled_cases": len(cases), "completed": sum(r["status"] == "completed" for r in rows),
        "new_api_attempts": sum(r["new_api_attempts"] for r in rows),
        "returned_responses": sum(r["returned_responses"] for r in rows),
        "recorded_usage": {k: sum(r["usage"][k] for r in rows) for k in ["model_calls", "input_tokens", "output_tokens"]}})
    return 0 if all(r["status"] == "completed" for r in rows) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", help="Comma-separated development IDs; default all")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--per-call-tokens", type=int, default=4000)
    parser.add_argument("--output-tokens", type=int, default=64000)
    parser.add_argument("--max-calls", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=900)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-repairs", type=int, default=1)
    parser.add_argument("--target-extension", action="store_true",
                        help="Plan immutable target clauses and decision probes before material analysis")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prompt-key", action="store_true")
    args = parser.parse_args()
    if not (1 <= args.rounds <= 3 and 1 <= args.workers <= 2 and 0 <= args.max_repairs <= 1):
        parser.error("rounds=1..3, workers=1..2, max-repairs=0..1")
    if not (500 <= args.per_call_tokens <= 16000 and args.output_tokens >= args.per_call_tokens
            and 1 <= args.max_calls <= 128 and 0 < args.seconds <= 1800):
        parser.error("Invalid bounded development budget")
    logging.disable(logging.CRITICAL)
    if args.prompt_key:
        if not sys.stdin.isatty():
            raise SystemExit("Echo-disabled terminal required")
        os.environ["OPENAI_API_KEY"] = getpass.getpass("API credential (hidden): ")
    try:
        return run(args)
    finally:
        if args.prompt_key:
            os.environ.pop("OPENAI_API_KEY", None)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "stopped", "error_type": type(exc).__name__}), flush=True)
        raise SystemExit(1)
