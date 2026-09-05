"""Offline paired original/staged development scoring with fixed denominators.

Consumes retained artifacts only; never imports a model adapter or reads credentials.
Equal caps are a resource opportunity comparison, not equal actual spend or a
held-out accuracy experiment. The original extractor and staged native label are
explicitly different measurement paths.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from newsverify import provenance as p
from newsverify.decisions import present_decision, round_decisions
from proof_score import (DECISIONS, MODES, USAGE_FIELDS, _arm_score, _edges,
                         _graph_score, _label_evaluable, _origins, _pair_counts, _ratio)
from staged_score import score as staged_score

EXPERIMENTS = {"original": "original_staged_budget_development_v1",
               "staged": "staged_psi_development_v1"}
PROVIDER = "fixed eligible snapshots, no open-web collection"
DATASET = "previously_seen_development_cases_not_hidden_benchmark"


def _read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {path.name}: {key}")
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=unique)


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value, context, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{context} must be finite and nonnegative")
    if integer and int(value) != value:
        raise ValueError(f"{context} must be an integer")
    return value


def _schedule(config, arm):
    if config.get("experiment") != EXPERIMENTS[arm]:
        raise ValueError(f"Unexpected {arm} experiment")
    ids = config.get("case_ids")
    if not isinstance(ids, list) or not ids or any(not isinstance(i, str) or
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", i) for i in ids):
        raise ValueError("Scheduled case IDs must be safe and nonempty")
    if len(set(ids)) != len(ids) or config.get("scheduled_cases") != len(ids):
        raise ValueError("Invalid or duplicate scheduled cases")
    for field, expected in (("gold_read_during_inference", False), ("new_response_only", True),
                            ("automatic_transport_retries", 0), ("provider", PROVIDER),
                            ("dataset_status", DATASET)):
        if config.get(field) != expected or (isinstance(expected, bool) and type(config.get(field)) is not bool):
            raise ValueError(f"Invalid source/execution assumption: {field}")
    if not isinstance(config.get("model"), str) or not config["model"]:
        raise ValueError("Missing model")
    if config.get("reasoning_effort") not in ("low", "medium", "high"):
        raise ValueError("Missing or invalid reasoning_effort")
    if not isinstance(config.get("input_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", config["input_sha256"]):
        raise ValueError("Missing full input fingerprint")
    budget = config.get("budget_per_case")
    if not isinstance(budget, dict) or set(budget) != {"calls", "output_tokens", "per_call_output_tokens", "seconds"}:
        raise ValueError("Missing exact per-case resource caps")
    for key, value in budget.items():
        if not _number(value, f"budget.{key}", key != "seconds"):
            raise ValueError("Resource caps must be positive")
    if budget["per_call_output_tokens"] > budget["output_tokens"]:
        raise ValueError("Per-call cap exceeds case output cap")
    if arm == "original" and config.get("original_depth") != 1:
        raise ValueError("Original depth must preserve the baseline contract")
    if arm == "staged":
        trace = config.get("trace_config", {})
        if type(trace.get("max_rounds")) is not int or not 1 <= trace["max_rounds"] <= 3 or trace.get("experimental_force_rounds") is not True:
            raise ValueError("Invalid forced staged verification contract")
    return ids


def _inputs(data, ids):
    if not isinstance(data, dict) or set(data) != {"schema_version", "cases"} or data["schema_version"] != 1:
        raise ValueError("Invalid inference-only input schema")
    cases, eligible = {}, {}
    for case in data["cases"]:
        if set(case) != {"target", "materials", "seed_ids", "full_evidence_control"}:
            raise ValueError("Unexpected case fields; gold must be separate")
        target = case["target"]
        if set(target) != {"id", "text", "as_of", "source_version_id", "assessment_mode", "evidence_scope"}:
            raise ValueError("Invalid target contract")
        case_id = target["id"]
        if case_id in cases or case_id not in ids or target["assessment_mode"] not in MODES:
            raise ValueError("Duplicate/unscheduled input or invalid assessment mode")
        if not isinstance(target["text"], str) or not target["text"].strip():
            raise ValueError("Missing target text")
        materials = [p.MaterialVersion(**m) for m in case["materials"]]
        versions = [m.version_id for m in materials]
        urls = [m.url for m in materials]
        if len(versions) != len(set(versions)) or len(urls) != len(set(urls)):
            raise ValueError("Ambiguous duplicate material version or URL")
        if not isinstance(target["evidence_scope"], list) or len(set(target["evidence_scope"])) != len(target["evidence_scope"]) or not set(target["evidence_scope"]) <= set(versions):
            raise ValueError("Invalid evidence scope")
        if target["source_version_id"] is not None and target["source_version_id"] not in versions:
            raise ValueError("Unknown source version")
        cutoff = p._time(target["as_of"], "as_of")
        eligible[case_id] = [m for m in materials if not p._material_eligibility(m, cutoff)]
        cases[case_id] = case
    if set(cases) != set(ids):
        raise ValueError("Inputs must contain every scheduled case exactly once")
    return cases, eligible


def _references(gold, inputs):
    if not isinstance(gold, dict) or not isinstance(gold.get("cases"), list):
        raise ValueError("Gold requires cases")
    all_cases = {}
    for case in gold["cases"]:
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in all_cases:
            raise ValueError("Missing or duplicate gold case IDs")
        if case.get("decision") not in DECISIONS | {None, "unknown"}:
            raise ValueError("Invalid reference label")
        if not isinstance(case.get("event_id"), str) or not case["event_id"]:
            raise ValueError("Missing reference event ID")
        for flag in ("label_evaluable", "origin_evaluable", "edges_evaluable"):
            if flag in case and type(case[flag]) is not bool:
                raise ValueError(f"Reference {flag} must be boolean")
        for dimension, parser in (("origins", _origins), ("edges", _edges)):
            if case.get(dimension) is not None:
                parser(case[dimension], "reference " + dimension)
        all_cases[case_id] = case
    if not set(inputs) <= set(all_cases):
        raise ValueError("Missing reference cases")
    cases = {i: all_cases[i] for i in inputs}
    for case_id, case in cases.items():
        if case.get("assessment_mode", gold.get("assessment_mode")) != inputs[case_id]["target"]["assessment_mode"]:
            raise ValueError("Input/reference assessment modes differ")
        versions = {m["version_id"] for m in inputs[case_id]["materials"]}
        if not set(case.get("origins") or []) <= versions or any(a not in versions or b not in versions for a, b, _ in case.get("edges") or []):
            raise ValueError("Reference graph lies outside supplied corpus")
    return cases


def _rows(run, config, ids, inputs, arm, response_ids):
    rows = _read(run / "results.json")
    if not isinstance(rows, list) or len(rows) != len(ids) or {r.get("id") for r in rows} != set(ids):
        raise ValueError("Every scheduled case needs exactly one result, including failures")
    for row in rows:
        case_id = row["id"]
        if row.get("arm", arm) != arm or (arm == "original" and row.get("arm") != arm):
            raise ValueError("Invalid result arm")
        if row.get("assessment_mode") != inputs[case_id]["target"]["assessment_mode"]:
            raise ValueError("Result assessment mode differs from task")
        if row.get("status") not in ("completed", "error"):
            raise ValueError("Invalid execution status")
        if row["status"] == "error" and row.get("prediction") is not None:
            raise ValueError("Execution error must not be relabeled as an answer")
        usage = row.get("usage", {})
        for key in USAGE_FIELDS:
            _number(usage.get(key), "usage." + key, key != "seconds")
        attempts = _number(row.get("new_api_attempts"), "new_api_attempts", True)
        returned = _number(row.get("returned_responses"), "returned_responses", True)
        if attempts != usage["model_calls"] or returned > attempts:
            raise ValueError("Attempt/response counts disagree with usage")
        per_case = run / f"{case_id}-result.json"
        if per_case.exists() and _read(per_case) != row:
            raise ValueError("Aggregate row differs from retained case result")
        calls_path = run / f"{case_id}-calls.json"
        calls = _read(calls_path) if calls_path.exists() else []
        if len(calls) != attempts or sum(bool(c.get("response_id")) for c in calls) != returned:
            raise ValueError("Attempt/response counts disagree with retained calls")
        token_totals = {"input_tokens": 0, "output_tokens": 0}
        for sequence, call in enumerate(calls, 1):
            if call.get("sequence") != sequence or any(call.get(k) for k in ("cached_from", "cache_accounted", "replayed")):
                raise ValueError("Invalid sequence or response replay")
            cap = _number(call.get("max_completion_tokens"), "call output cap", True)
            if not 0 < cap <= config["budget_per_case"]["per_call_output_tokens"]:
                raise ValueError("Invalid per-call cap")
            response_id = call.get("response_id")
            if response_id:
                if response_id in response_ids:
                    raise ValueError("Duplicate response ID across fresh calls")
                response_ids.add(response_id)
            for key in token_totals:
                token_totals[key] += _number(call.get("usage", {}).get(key, 0), "call usage." + key, True)
            if row["status"] == "completed" and (not response_id or call.get("actual_model") != config["model"] or
                    call.get("error_type") or call.get("error_code") or call.get("has_refusal") or
                    call.get("finish_reason") != "stop" or call.get("usage", {}).get("output_tokens", cap + 1) > cap):
                raise ValueError("Completed case contains a failed or mismatched model call")
        if any(usage[key] != total for key, total in token_totals.items()):
            raise ValueError("Token usage disagrees with retained calls")
        if attempts > config["budget_per_case"]["calls"]:
            raise ValueError("Attempt cap exceeded")
        if row["status"] == "completed" and (not attempts or returned != attempts or
                usage["output_tokens"] > config["budget_per_case"]["output_tokens"] or usage["seconds"] > config["budget_per_case"]["seconds"]):
            raise ValueError("Completed result violates execution budget")
    return rows


def _original(row, report, eligible):
    prediction, extraction = row.get("prediction"), row.get("extraction", {})
    if not isinstance(prediction, dict) or prediction.get("decision") not in DECISIONS:
        raise ValueError("Completed original result lacks a valid prediction")
    answer = row.get("extractor_answer_text")
    quote = extraction.get("basis_quote")
    if not isinstance(answer, str) or answer != report.get("direct_response") or extraction.get("extractable") is not True or not isinstance(quote, str) or not quote.strip() or quote not in answer or extraction.get("decision") != prediction["decision"]:
        raise ValueError("Ungrounded or changed original report extraction")
    if prediction.get("provenance_evaluable") is not True or prediction.get("origin_evaluable") is not True or prediction.get("edges_evaluable") is not False or prediction.get("edges") != [] or not prediction.get("edge_export_limit"):
        raise ValueError("Original graph export must preserve unsupported edges")
    ids = {m.url: m.version_id for m in eligible}
    sources = report.get("sources")
    if not isinstance(sources, list) or any(not isinstance(s, dict) or not isinstance(s.get("url"), str) or "is_original" not in s for s in sources):
        raise ValueError("Invalid original native source flags")
    expected = {ids[s["url"]] for s in sources if s["is_original"] and s["url"] in ids}
    if _origins(prediction.get("origins"), "original origins") != expected:
        raise ValueError("Original origins differ from native source flags")
    return {"unmapped_urls": sorted({s["url"] for s in sources if s["is_original"] and s["url"] not in ids}),
            "native_flags_boolean": all(type(s["is_original"]) is bool for s in sources)}


def _staged(row, report, config, input_case, eligible):
    if report.get("target") != input_case["target"] or report.get("config") != config["trace_config"]:
        raise ValueError("Staged report changes the frozen task or trace contract")
    if report.get("assessment_valid") is not True or report.get("errors"):
        raise ValueError("Completed staged report is invalid")
    native = present_decision(report)
    if row.get("native_prediction") != native or row.get("prediction") != native["decision"]:
        raise ValueError("Staged label differs from validated native decision")
    if row.get("checkpoints") != round_decisions(report) or row.get("engine_usage") != report.get("usage"):
        raise ValueError("Staged checkpoints or usage differ from engine report")
    if report["usage"]["verification_calls"] != config["trace_config"]["max_rounds"]:
        raise ValueError("Required staged verifications were not completed")
    versions = {m.version_id for m in eligible}
    origins = sorted({o["version_id"] for o in report["origins"]})
    edges = sorted({(e["from_version"], e["to_version"], e["kind"]) for e in report["relations"]
        if e["status"] == "direct" and e["to_version"] and e["kind"] in ("cites", "quotes")})
    if not set(origins) <= versions or any(a not in versions or b not in versions for a, b, _ in edges):
        raise ValueError("Staged graph refers to an ineligible material")
    normalized = deepcopy(row)
    normalized["prediction"] = {"decision": native["decision"], "provenance_evaluable": True,
                                "origins": origins, "edges": edges}
    return normalized


def _paired(cases, matrix, completed_only=False):
    pairs, table = [], []
    for case_id, case in cases.items():
        if not _label_evaluable(case):
            continue
        a, b = matrix[(case_id, "original")], matrix[(case_id, "staged")]
        if completed_only and any(row["status"] != "completed" for row in (a, b)):
            continue
        av = a["status"] == "completed" and a["prediction"]["decision"] == case["decision"]
        bv = b["status"] == "completed" and b["prediction"]["decision"] == case["decision"]
        pairs.append((case["event_id"], av, bv))
        table.append(case_id)
    return {"case_pairs": len(pairs), "case_ids": table, **_pair_counts(pairs),
            "original_correct": sum(a for _, a, _ in pairs), "staged_correct": sum(b for _, _, b in pairs),
            "staged_minus_original": _ratio(sum(int(b) - int(a) for _, a, b in pairs), len(pairs))}


def score(gold_path, original_run, staged_run):
    runs = {"original": Path(original_run), "staged": Path(staged_run)}
    configs = {arm: _read(run / "config.json") for arm, run in runs.items()}
    schedules = {arm: _schedule(configs[arm], arm) for arm in runs}
    for key in ("model", "reasoning_effort", "budget_per_case", "input_sha256"):
        if configs["original"][key] != configs["staged"][key]:
            raise ValueError(f"Runs have unequal {key}")
    if set(schedules["original"]) != set(schedules["staged"]):
        raise ValueError("Runs have different scheduled cases")
    data = {arm: _read(run / "inputs.json") for arm, run in runs.items()}
    inputs, eligible = _inputs(data["original"], schedules["original"])
    staged_inputs, _ = _inputs(data["staged"], schedules["staged"])
    if inputs != staged_inputs:
        raise ValueError("Runs have different selected corpus or task contracts")
    gold = _read(Path(gold_path))
    cases = _references(gold, inputs)
    matrix, response_ids, unmapped, invalid_flags = {}, set(), {}, []
    for arm, run in runs.items():
        rows = _rows(run, configs[arm], schedules[arm], inputs, arm, response_ids)
        for row in rows:
            normalized = row
            if row["status"] == "completed":
                report = _read(run / f"{row['id']}-report.json")
                if arm == "original":
                    audit = _original(row, report, eligible[row["id"]])
                    unmapped[row["id"]] = audit["unmapped_urls"]
                    if not audit["native_flags_boolean"]:
                        invalid_flags.append(row["id"])
                        normalized = deepcopy(row)
                        normalized["prediction"]["origin_evaluable"] = False
                else:
                    normalized = _staged(row, report, configs[arm], inputs[row["id"]], eligible[row["id"]])
            if arm == "staged" and row["status"] == "error" and row.get("checkpoints"):
                report = _read(run / f"{row['id']}-report.json")
                if report.get("target") != inputs[row["id"]]["target"] or row["checkpoints"] != round_decisions(report):
                    raise ValueError("Failed staged case has invalid retained checkpoints")
            matrix[(row["id"], arm)] = normalized
    # Keep the existing staged scorer's independent report/round/graph gate intact.
    staged_audit = staged_score(gold_path, staged_run)
    arms = {arm: _arm_score(cases, matrix, arm) for arm in runs}
    for arm in arms:
        arms[arm]["new_api_attempts"] = sum(matrix[(i, arm)]["new_api_attempts"] for i in cases)
        arms[arm]["returned_responses"] = sum(matrix[(i, arm)]["returned_responses"] for i in cases)
        arms[arm]["origins"]["denominator_note"] = "Own completed evaluable cases only; failures are excluded, so arm denominators may differ."
    arms["original"]["edges"].update(status="unsupported_native_export", precision=None, recall=None,
        note="The original agent has no native source-to-source edge representation; zero exported edges is not zero edge accuracy.")
    arms["original"]["origin_unavailable_nonboolean_native_flags"] = sorted(invalid_flags)
    arms["original"]["unmapped_native_original_sources"] = unmapped
    arms["original"]["unmapped_native_original_source_count"] = sum(map(len, unmapped.values()))
    shared = {i: case for i, case in cases.items() if all(matrix[(i, arm)]["status"] == "completed" for arm in runs)}
    shared_origins = {i: c for i, c in shared.items() if i not in invalid_flags}
    round_pairs = [(cases[r["id"]]["event_id"], r["first_round_decision"] == cases[r["id"]]["decision"],
                    r["correct"]) for r in staged_audit["cases"] if _label_evaluable(cases[r["id"]])]
    round_comparison = {"scheduled_label_evaluable_cases": len(round_pairs), **_pair_counts(round_pairs),
        "first_round_correct": sum(a for _, a, _ in round_pairs), "final_correct": sum(b for _, _, b in round_pairs),
        "first_round_task_success": _ratio(sum(a for _, a, _ in round_pairs), len(round_pairs)),
        "final_task_success": _ratio(sum(b for _, _, b in round_pairs), len(round_pairs)),
        "final_minus_first_round": _ratio(sum(int(b) - int(a) for _, a, b in round_pairs), len(round_pairs)),
        "note": "All scheduled label-evaluable cases; missing first checkpoints and failed final executions are task failures."}
    table = []
    for i, case in cases.items():
        item = {"id": i, "event_id": case["event_id"], "assessment_mode": inputs[i]["target"]["assessment_mode"],
                "reference": case["decision"], "label_evaluable": _label_evaluable(case)}
        for arm in runs:
            row = matrix[(i, arm)]
            item[arm] = {"status": row["status"], "decision": row["prediction"]["decision"] if row["status"] == "completed" else None,
                "task_success": row["status"] == "completed" and row["prediction"]["decision"] == case["decision"] if _label_evaluable(case) else None,
                "usage": row["usage"]}
        table.append(item)
    return {"schema_version": "staged-compare-score-v1", "comparison": "staged_vs_original_equal_caps_development",
        "scheduled_cases_per_arm": len(cases), "independent_events": len({c["event_id"] for c in cases.values()}),
        "case_ids": sorted(cases), "model": configs["original"]["model"],
        "reasoning_effort": configs["original"]["reasoning_effort"], "budget_per_case": configs["original"]["budget_per_case"],
        "arms": arms, "all_scheduled_task_comparison": _paired(cases, matrix),
        "shared_completed_label_comparison": _paired(cases, matrix, completed_only=True),
        "shared_completed_origins": {"case_ids": sorted(shared), "completed_case_pairs": len(shared),
            "source_output_excluded_case_ids": sorted(set(shared) - set(shared_origins)),
            "arms": {arm: _graph_score(shared_origins, matrix, arm, "origins") for arm in runs}},
        "staged_round1_to_final": {**{key: staged_audit[key] for key in
            ("scheduled", "completed", "execution_errors", "paired_completed_cases", "fixes", "breaks", "cases")},
            "all_scheduled_task_comparison": round_comparison},
        "usage_staged_minus_original": {key: arms["staged"]["usage_total"][key] - arms["original"]["usage_total"][key] for key in USAGE_FIELDS},
        "cases": table, "reference_sha256": _hash(Path(gold_path)),
        "input_sha256": configs["original"]["input_sha256"],
        "selected_inputs_sha256": hashlib.sha256(json.dumps(inputs, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "artifacts_sha256": {arm: {name: _hash(run / name) for name in ("config.json", "inputs.json", "results.json")} for arm, run in runs.items()},
        "conclusion": "development_comparison_only_no_proof_of_realworld_superiority",
        "limits": [
            "Previously seen development cases and provisional AI references; no independent human adjudication or held-out population estimate.",
            "Equal configured caps, unequal actual spend. Input tokens are measured, not capped; per-case seconds are summed service time, not experiment wall time.",
            "Original labels use its unchanged report-only extractor, charged to its budget; staged labels use validated native output without an extractor. Quote grounding cannot certify semantic extraction correctness.",
            "All-scheduled label task success retains execution errors. Shared-completed comparisons condition on success by both arms and can select easier cases.",
            "Origin metrics use each arm's completed evaluable cases; compare shared-completed origins for a common denominator. Origins are corpus-relative documentary roots, not earliest internet publication.",
            "Unmapped original native source URLs are disclosed and excluded from corpus-relative origin precision; this is a measurement limitation. Nonboolean native originality flags make only that origin dimension unavailable, leaving label outcomes unchanged.",
            "Original edges are unsupported and have no accuracy score; staged edges have their own completed evaluable denominator.",
            "Fixed eligible excerpts only, no open-web collection benchmark. Equal declared full-input fingerprints are checked alongside identical selected input contracts; saved selections do not reconstruct the original full input file.",
        ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gold", "original-run", "staged-run", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = score(args.gold, args.original_run, args.staged_run)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"scheduled_cases_per_arm": result["scheduled_cases_per_arm"],
                      "all_scheduled_task_comparison": result["all_scheduled_task_comparison"]}))


if __name__ == "__main__":
    main()
