"""Offline development scoring; execution success and reference agreement differ."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from proof_score import DECISIONS, _graph_score


def score(gold_path, run_path):
    run = Path(run_path)
    gold = json.loads(Path(gold_path).read_text())
    config = json.loads((run / "config.json").read_text())
    rows = json.loads((run / "results.json").read_text())
    ids = config["case_ids"]
    if len(ids) != len(set(ids)) or len(rows) != len(ids) or {r["id"] for r in rows} != set(ids):
        raise ValueError("Every scheduled case needs exactly one result, including failures")
    gold_cases = {c["id"]: c for c in gold["cases"]}
    if len(gold_cases) != len(gold["cases"]) or not set(ids) <= gold_cases.keys():
        raise ValueError("Missing or duplicate reference cases")
    cases = {key: gold_cases[key] for key in ids}
    matrix, table = {}, []
    for row in rows:
        case = cases[row["id"]]
        if row["assessment_mode"] != case["assessment_mode"]:
            raise ValueError("Input/reference assessment modes differ")
        if row["status"] not in ("error", "completed"):
            raise ValueError("Invalid execution status")
        first = next((r["decision"] for r in row["checkpoints"] if r["round"] == 1), None)
        item = {"id": row["id"], "reference": case["decision"], "status": row["status"],
            "first_round_decision": first, "final_decision": row["prediction"],
            "correct": row["status"] == "completed" and row["prediction"] == case["decision"],
            "verification_rounds": [r["round"] for r in row["checkpoints"]]}
        normalized = deepcopy(row)
        if row["status"] == "completed":
            if row["prediction"] not in DECISIONS or not row["native_prediction"]["assessment_valid"]:
                raise ValueError("Completed result lacks a valid native prediction")
            if item["verification_rounds"] != list(range(1, config["trace_config"]["max_rounds"] + 1)):
                raise ValueError("A completed case did not execute required verification rounds")
            report = json.loads((run / (row["id"] + "-report.json")).read_text())
            if report["errors"] or report["decision_status"] != row["native_prediction"]["assessments"][row["assessment_mode"]]["decision"]:
                raise ValueError("Native result disagrees with engine report")
            normalized["prediction"] = {"decision": row["prediction"], "provenance_evaluable": True,
                "origins": sorted({o["version_id"] for o in report["origins"]}),
                "edges": sorted({(e["from_version"], e["to_version"], e["kind"]) for e in report["relations"]
                    if e["status"] == "direct" and e["to_version"] and e["kind"] in ("cites", "quotes")})}
            item["provenance_status"] = report["provenance_status"]
        elif row["prediction"] is not None:
            raise ValueError("Execution error must not be relabeled as an answer")
        matrix[(row["id"], "staged")] = normalized
        table.append(item)
    completed = [r for r in table if r["status"] == "completed"]
    correct = sum(r["correct"] for r in table)
    paired = [r for r in completed if r["first_round_decision"] is not None]
    return {"experiment": config["experiment"], "scheduled": len(rows), "completed": len(completed),
        "execution_errors": len(rows) - len(completed), "correct_against_provisional_reference": correct,
        "execution_gate_passed": len(completed) == len(rows),
        "provisional_label_gate_passed": correct == len(rows),
        "task_success_rate": correct / len(rows), "conditional_label_agreement": correct / len(completed) if completed else None,
        "paired_completed_cases": len(paired),
        "fixes": sum(r["first_round_decision"] != r["reference"] and r["correct"] for r in paired),
        "breaks": sum(r["first_round_decision"] == r["reference"] and not r["correct"] for r in paired),
        "origins": _graph_score(cases, matrix, "staged", "origins"),
        "edges": _graph_score(cases, matrix, "staged", "edges"),
        "cases": sorted(table, key=lambda r: r["id"]),
        "usage": {k: sum(r["usage"][k] for r in rows) for k in ("model_calls", "input_tokens", "output_tokens", "seconds")},
        "reference_sha256": hashlib.sha256(Path(gold_path).read_bytes()).hexdigest(),
        "results_sha256": hashlib.sha256((run / "results.json").read_bytes()).hexdigest(),
        "limits": ["Previously seen development cases; reference labels are AI-reviewed, not human-adjudicated.",
                   "This is a changed pipeline and resource configuration, not an equal-budget superiority test.",
                   "Fixed snapshots and forced outer rounds; no open-web collection benchmark.",
                   "Semantic acceptance by a model critic is not independent ground truth."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    result = score(args.gold, args.run)
    (Path(args.run) / "scores.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("scheduled", "completed", "execution_errors",
        "correct_against_provisional_reference", "execution_gate_passed", "provisional_label_gate_passed")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
