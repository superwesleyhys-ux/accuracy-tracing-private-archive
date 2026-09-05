"""Fresh original-only baseline under the staged development resource caps.

The pinned original agent and fixed-corpus adapter are unchanged. Its report-only
label extractor is charged to the same case budget. No gold or old responses are
read; every scheduled failure is retained and output directories cannot overwrite.
"""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import getpass
import hashlib
import importlib
import io
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
from original_compare import OriginalCorpusClient, extract
from semantic_adapter import DATA_RULE


def load_original(original):
    """Reject a cached import from a different original checkout."""
    sys.path.insert(0, str(original))
    module = importlib.import_module("agent.core")
    if Path(module.__file__).resolve() != original / "agent" / "core.py":
        raise ValueError("Imported original agent differs from the requested pinned checkout")
    return module.NewsTracingAgent


def run(args):
    data = load_inputs(args.inputs)
    if args.cases:
        selected = set(args.cases.split(","))
        if not selected <= {c["target"]["id"] for c in data["cases"]}:
            raise ValueError("Unknown selected development case")
        data = {**data, "cases": [c for c in data["cases"] if c["target"]["id"] in selected]}
    if not data["cases"] or any(not isinstance(c["target"]["id"], str) or
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", c["target"]["id"])
            for c in data["cases"]):
        raise ValueError("Cases must have safe, bounded artifact IDs")
    original = Path(args.original).resolve()
    if not (original / "agent" / "core.py").is_file():
        raise ValueError("Pinned original agent/core.py is required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    budget = Budget(calls=args.max_calls, output_tokens=args.output_tokens,
                    per_call_output_tokens=args.per_call_tokens, seconds=args.seconds)
    names = ("staged_baseline_run.py", "original_compare.py", "model_io.py",
             "loop_compare.py", "semantic_adapter.py")
    source_files = list((ROOT / "newsverify").glob("*.py")) + [ROOT / "experiments" / n for n in names]
    original_files = list((original / "agent").rglob("*.py"))
    fingerprint = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    config = {"experiment": "original_staged_budget_development_v1", "model": args.model,
        "reasoning_effort": args.reasoning_effort, "budget_per_case": asdict(budget),
        "original_depth": 1, "workers": args.workers, "ordering_seed": 20260906,
        "case_ids": [c["target"]["id"] for c in data["cases"]], "scheduled_cases": len(data["cases"]),
        "gold_read_during_inference": False, "new_response_only": True, "automatic_transport_retries": 0,
        "dataset_status": "previously_seen_development_cases_not_hidden_benchmark",
        "provider": "fixed eligible snapshots, no open-web collection",
        "extractor": "unchanged original report-only label extractor; one call charged to case budget",
        "staged_extractor_asymmetry": "staged uses a validated native decision without an extractor call",
        "input_tokens_capped": False, "actual_spend_equal": False,
        "input_sha256": fingerprint(Path(args.inputs)),
        "source_sha256": {f.relative_to(ROOT).as_posix(): fingerprint(f) for f in sorted(source_files)},
        "original_source_sha256": {f.relative_to(original).as_posix(): fingerprint(f) for f in sorted(original_files)}}
    write(output / "config.json", config)
    write(output / "inputs.json", data)
    for base, files, prefix in ((ROOT, source_files, "accuracy"), (original, original_files, "original")):
        for source in files:
            dest = output / "executed-code" / prefix / source.relative_to(base)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
    if not os.environ.get("OPENAI_API_KEY"):
        write(output / "status.json", {"status": "blocked_missing_auth", "new_api_attempts": 0})
        return 2
    # Resolve once before worker threads; constructor failures below still yield rows.
    NewsTracingAgent = load_original(original)
    from rich.console import Console

    def one(case):
        identifier = case["target"]["id"]

        class RecordedClient(BudgetClient):
            def __init__(self):
                super().__init__(args.model, budget, reasoning_effort=args.reasoning_effort)
                self.recording_lock = threading.Lock()

            def call(self, *a, **kw):
                with self.recording_lock:
                    try:
                        answer = super().call(*a, **kw)
                        record = self.records[-1]
                        if record.get("actual_model") != args.model:
                            record.update(error_type="UnexpectedModel", error_code="unexpected_model")
                            raise ValueError("Returned model differs from configured model")
                        if record["usage"]["output_tokens"] > record["max_completion_tokens"]:
                            record.update(error_type="OutputCapExceeded", error_code="per_call_cap_exceeded")
                            raise ValueError("Observed completion cap exceeded")
                        return answer
                    finally:
                        write(output / f"{identifier}-calls.json", self.records)
                        print(json.dumps({"case": identifier, "arm": "original", "calls": self.calls}), flush=True)

        row = {"id": identifier, "arm": "original", "assessment_mode": case["target"]["assessment_mode"],
               "status": "error", "prediction": None}
        client = None
        try:
            client = RecordedClient()
            target = p.Target(**{**case["target"], "evidence_scope": tuple(case["target"]["evidence_scope"])})
            cutoff = p._time(target.as_of, "as_of")
            materials = [p.MaterialVersion(**m) for m in case["materials"]]
            eligible = [m for m in materials if not p._material_eligibility(m, cutoff)]
            clean = {**case, "materials": [asdict(m) for m in eligible]}
            write(output / f"{identifier}-input.json", clean)
            agent = NewsTracingAgent(OriginalCorpusClient(client, clean), max_depth=1,
                                     console=Console(file=io.StringIO(), quiet=True))
            task = DATA_RULE + "\nAssess the claim under this fixed contract:\n" + json.dumps(asdict(target))
            report = asdict(asyncio.run(agent.run(task)))
            write(output / f"{identifier}-report.json", report)
            answer = report["direct_response"]
            row["extractor_answer_text"] = answer
            row["extraction"] = extract(client, asdict(target), answer)
            if any(r.get("error_type") for r in client.records):
                raise ValueError("A failed model call cannot be ignored")
            ids = {m.url: m.version_id for m in eligible}
            row["prediction"] = {"decision": row["extraction"]["decision"],
                "origins": sorted({ids[s["url"]] for s in report["sources"]
                    if s.get("is_original") and s["url"] in ids}),
                "edges": [], "provenance_evaluable": True, "origin_evaluable": True,
                "edges_evaluable": False, "edge_export_limit": "No native source-to-source edge representation"}
            row["status"] = "completed"
        except Exception as exc:
            row["error_type"] = type(exc).__name__
        finally:
            row["usage"] = client.usage() if client else {k: 0 for k in
                ("model_calls", "input_tokens", "output_tokens", "seconds")}
            records = client.records if client else []
            row["new_api_attempts"] = len(records)
            row["returned_responses"] = sum(bool(r.get("response_id")) for r in records)
            write(output / f"{identifier}-calls.json", records)
            write(output / f"{identifier}-result.json", row)
            print(json.dumps({"case": identifier, "arm": "original", "status": row["status"]}), flush=True)
        return row

    cases = list(data["cases"])
    random.Random(20260906).shuffle(cases)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(one, case) for case in cases]):
            rows.append(future.result())
            write(output / "results.json", sorted(rows, key=lambda r: r["id"]))
    success = all(r["status"] == "completed" for r in rows)
    write(output / "status.json", {"status": "completed" if success else "has_errors",
        "scheduled_cases": len(rows), "completed": sum(r["status"] == "completed" for r in rows),
        "new_api_attempts": sum(r["new_api_attempts"] for r in rows),
        "returned_responses": sum(r["returned_responses"] for r in rows),
        "recorded_usage": {k: sum(r["usage"][k] for r in rows)
            for k in ("model_calls", "input_tokens", "output_tokens")}})
    return 0 if success else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "original", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--cases", help="Comma-separated development IDs; default all")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--per-call-tokens", type=int, default=4000)
    parser.add_argument("--output-tokens", type=int, default=64000)
    parser.add_argument("--max-calls", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=900)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prompt-key", action="store_true")
    args = parser.parse_args()
    if not (1 <= args.workers <= 2 and 500 <= args.per_call_tokens <= 16000
            and args.output_tokens >= args.per_call_tokens and 1 <= args.max_calls <= 128
            and 0 < args.seconds <= 1800):
        parser.error("Invalid bounded development budget or workers")
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
