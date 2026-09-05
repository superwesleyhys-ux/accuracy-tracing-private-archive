"""Artifact-only comparison of original, staged, and target-extended runs.

The extension run selects the comparison denominator.  Baseline runs may contain
additional cases, but the scorer compares only identical selected target/material
contracts.  No inference adapter, credential, or network service is imported.

Probe coverage is structural: it shows that required checklist items exist in a
checksummed plan and are routed to executed stages.  It is not evidence that the
model answered those probes correctly and is reported separately from labels and
provenance accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from proof_score import USAGE_FIELDS, _arm_score, _label_evaluable, _ratio
from staged_compare_score import (
    DATASET,
    PROVIDER,
    _hash,
    _inputs,
    _original,
    _read,
    _references,
    _rows,
    _schedule,
    _staged,
)


EXTENSION_EXPERIMENT = "target_extended_psi_development_v1"
PLAN_STAGES = ("atoms", "lineage", "critic", "evidence", "world")
ROUTED_STAGES = ("atoms", "lineage", "evidence", "world")
DIMENSION_PROBES = {
    "negation": "polarity",
    "time": "time_boundary",
    "quantity_unit": "quantity_unit",
    "baseline_scope": "baseline_scope",
    "entity_identity": "entity_identity",
}


def _canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":")).encode()).hexdigest()


def _nonnegative(value, context, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value < 0 or (integer and int(value) != value)):
        raise ValueError(f"{context} must be a finite nonnegative " +
                         ("integer" if integer else "number"))
    return value


def _extension_schedule(config):
    if config.get("experiment") != EXTENSION_EXPERIMENT:
        raise ValueError("Unexpected target-extension experiment")
    if config.get("target_extension") is not True:
        raise ValueError("Target extension must be explicitly enabled")
    ids = config.get("case_ids")
    if (not isinstance(ids, list) or not ids or any(not isinstance(item, str) or
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", item) for item in ids) or
            len(ids) != len(set(ids)) or config.get("scheduled_cases") != len(ids)):
        raise ValueError("Invalid target-extension schedule")
    for field, expected in (("gold_read_during_inference", False),
                            ("new_response_only", True),
                            ("automatic_transport_retries", 0),
                            ("provider", PROVIDER), ("dataset_status", DATASET)):
        if config.get(field) != expected or (isinstance(expected, bool) and
                                              type(config.get(field)) is not bool):
            raise ValueError(f"Invalid source/execution assumption: {field}")
    trace = config.get("trace_config")
    if (not isinstance(trace, dict) or type(trace.get("max_rounds")) is not int or
            not 1 <= trace["max_rounds"] <= 3 or
            trace.get("experimental_force_rounds") is not True):
        raise ValueError("Invalid forced target-extension verification contract")
    if type(config.get("max_inner_repairs")) is not int or config["max_inner_repairs"] not in (0, 1):
        raise ValueError("Invalid target-extension repair budget")
    if not isinstance(config.get("model"), str) or not config["model"]:
        raise ValueError("Missing model")
    if config.get("reasoning_effort") not in ("low", "medium", "high"):
        raise ValueError("Missing or invalid reasoning_effort")
    if not isinstance(config.get("input_sha256"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", config["input_sha256"]):
        raise ValueError("Missing full input fingerprint")
    budget = config.get("budget_per_case")
    if not isinstance(budget, dict) or set(budget) != {
            "calls", "output_tokens", "per_call_output_tokens", "seconds"}:
        raise ValueError("Missing exact per-case resource caps")
    for key, value in budget.items():
        _nonnegative(value, "budget." + key, key != "seconds")
        if value <= 0:
            raise ValueError("Resource caps must be positive")
    if budget["per_call_output_tokens"] > budget["output_tokens"]:
        raise ValueError("Per-call cap exceeds case output cap")
    return ids


def _calls_usage(run, ids, rows):
    by_id = {row["id"]: row for row in rows}
    result = {"model_calls": 0, "input_tokens": 0, "output_tokens": 0,
              "reasoning_tokens": 0, "visible_output_tokens": 0,
              "service_seconds": 0, "reasoning_reporting_complete": True}
    for case_id in ids:
        row = by_id[case_id]
        calls_path = run / f"{case_id}-calls.json"
        calls = _read(calls_path) if calls_path.exists() else []
        if len(calls) != row["usage"]["model_calls"]:
            raise ValueError("Call count disagrees with row usage")
        reasoning = 0
        complete = True
        for call in calls:
            if "reasoning_tokens" not in call:
                complete = False
                continue
            value = _nonnegative(call["reasoning_tokens"], "reasoning_tokens", True)
            if value > call.get("usage", {}).get("output_tokens", 0):
                raise ValueError("Reasoning tokens exceed output tokens")
            reasoning += int(value)
        for field in USAGE_FIELDS:
            value = row["usage"][field]
            if field == "seconds":
                result["service_seconds"] += value
            else:
                result[field] += value
        result["reasoning_tokens"] += reasoning
        result["visible_output_tokens"] += row["usage"]["output_tokens"] - reasoning
        result["reasoning_reporting_complete"] &= complete
    if not result["reasoning_reporting_complete"]:
        result["reasoning_tokens"] = None
        result["visible_output_tokens"] = None
    return result


def _history(run, case_id, suffix):
    path = run / f"{case_id}-{suffix}.json"
    if not path.exists():
        return []
    value = _read(path)
    if not isinstance(value, list):
        raise ValueError(f"{suffix} must be an array")
    for sequence, item in enumerate(value, 1):
        if not isinstance(item, dict) or item.get("sequence") != sequence:
            raise ValueError(f"{suffix} has an invalid sequence")
        repair = item.get("repair")
        if isinstance(repair, dict):
            if set(repair) != {"structure", "extension"}:
                raise ValueError(f"{suffix}.repair has invalid counters")
            for key, count in repair.items():
                _nonnegative(count, f"{suffix}.repair.{key}", True)
        else:
            _nonnegative(repair, f"{suffix}.repair", True)
        if not isinstance(item.get("stage"), str) or not isinstance(item.get("status"), str):
            raise ValueError(f"{suffix} has an invalid stage or status")
    return value


def _repair_counter(item, name=None):
    value = item["repair"]
    if isinstance(value, dict):
        return value[name] if name else sum(value.values())
    return value


def _loop_counts(run, ids):
    totals = Counter()
    per_case = []
    for case_id in ids:
        planner = _history(run, case_id, "target-planner-history")
        psi = _history(run, case_id, "psi-history")
        verification = _history(run, case_id, "verification-history")
        item = {
            "id": case_id,
            "target_planner_calls": len(planner),
            "target_plan_repair_requests": sum(x["status"] in
                ("repair_requested", "structure_repair_requested") for x in planner),
            "target_plan_repair_followup_calls": sum(_repair_counter(x) > 0 for x in planner),
            "target_structure_repair_requests": sum(x["status"] ==
                "structure_repair_requested" for x in planner),
            "target_structure_repair_followup_calls": sum(
                _repair_counter(x, "structure") > 0 for x in planner
                if isinstance(x["repair"], dict)),
            "target_extension_repair_requests": sum(x["status"] ==
                "repair_requested" for x in planner),
            "target_extension_repair_followup_calls": sum(
                _repair_counter(x, "extension") > 0 for x in planner
                if isinstance(x["repair"], dict)),
            "material_critic_calls": sum(x["stage"] == "critic" for x in psi),
            "material_critic_repair_requests": sum(x["stage"] == "critic" and
                x["status"] == "repair_requested" for x in psi),
            "material_stage_repair_attempts": sum(x["stage"] in ("atoms", "lineage") and
                _repair_counter(x) > 0 for x in psi),
            "judgement_critic_calls": sum(x["stage"] == "judgement_critic" for x in verification),
            "judgement_critic_repair_requests": sum(x["stage"] == "judgement_critic" and
                x["status"] == "repair_requested" for x in verification),
            "judgement_layer_repair_attempts": sum(x["stage"] in ("evidence", "world") and
                _repair_counter(x) > 0 for x in verification),
        }
        per_case.append(item)
        totals.update({key: value for key, value in item.items() if key != "id"})
    return {"totals": dict(totals), "cases": per_case}


def _required_probe_bindings(claim, assessment_mode):
    by_kind = {}
    dimension_ids = []
    for item in claim.get("dimensions", []):
        if (not isinstance(item, dict) or not isinstance(item.get("id"), str) or
                not item["id"] or not isinstance(item.get("kind"), str)):
            raise ValueError("claim has an invalid dimension")
        dimension_ids.append(item["id"])
        by_kind.setdefault(item["kind"], []).append(item["id"])
    if len(dimension_ids) != len(set(dimension_ids)):
        raise ValueError("claim has duplicate dimension ids")

    def ids(*kinds):
        return tuple(sorted(identifier for kind in kinds for identifier in by_kind.get(kind, ())))

    required = {("semantic_core", ids("subject", "predicate")),
                ("source_lineage", ())}
    for dimension, probe in DIMENSION_PROBES.items():
        if dimension == "entity_identity":
            required.update((probe, (identifier,)) for identifier in by_kind.get(dimension, ()))
        elif dimension in by_kind:
            required.add((probe, ids(dimension)))
    if set(by_kind) & {"condition", "modality"}:
        required.add(("condition_modality", ids("condition", "modality")))
    if assessment_mode == "world":
        required.add(("source_independence", ()))
    return required


def _ids(items, context):
    if not isinstance(items, list):
        raise ValueError(context + " must be an array")
    values = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
            raise ValueError(context + " has an invalid id")
        values.append(item["id"])
    if len(values) != len(set(values)):
        raise ValueError(context + " has duplicate ids")
    return values


def _audit_plan(path, target, row, psi, verification):
    plan = _read(path)
    if not isinstance(plan, dict):
        raise ValueError("target plan must be an object")
    required_top = {"target_signature", "logic", "claims", "probes", "notes",
                    "sha256", "projections"}
    if not required_top <= set(plan):
        raise ValueError("target plan is missing required fields")
    if plan["target_signature"] != _canonical_digest(target):
        raise ValueError("target plan signature differs from immutable target")
    payload = {key: plan[key] for key in
               ("target_signature", "logic", "claims", "probes", "notes")}
    if plan["sha256"] != _canonical_digest(payload):
        raise ValueError("target plan checksum mismatch")
    if row.get("target_plan_sha256") != plan["sha256"]:
        raise ValueError("result row target plan checksum mismatch")
    claims, probes = plan["claims"], plan["probes"]
    claim_ids, probe_ids = _ids(claims, "target claims"), _ids(probes, "decision probes")
    known_claims = set(claim_ids)
    claims_by_id = {item["id"]: item for item in claims}
    by_id = {}
    actual_bindings = set()
    for probe in probes:
        if probe.get("claim_id") not in known_claims or not isinstance(probe.get("kind"), str):
            raise ValueError("decision probe references an unknown claim or kind")
        dimension_ids = probe.get("dimension_ids")
        if (not isinstance(dimension_ids, list) or
                any(not isinstance(item, str) or not item for item in dimension_ids) or
                len(dimension_ids) != len(set(dimension_ids))):
            raise ValueError("decision probe has invalid dimension bindings")
        owned = {item["id"] for item in claims_by_id[probe["claim_id"]].get("dimensions", [])}
        if not set(dimension_ids) <= owned:
            raise ValueError("decision probe binds a dimension outside its claim")
        routes = probe.get("routes")
        if (not isinstance(routes, list) or len(routes) != len(set(routes)) or
                not set(routes) <= set(ROUTED_STAGES)):
            raise ValueError("decision probe has invalid routes")
        by_id[probe["id"]] = probe
        actual_bindings.add((probe["claim_id"], probe["kind"],
                             tuple(sorted(dimension_ids))))
    required_bindings = {(claim["id"], kind, dimension_ids) for claim in claims
                         for kind, dimension_ids in _required_probe_bindings(
                             claim, target["assessment_mode"])}
    if actual_bindings - required_bindings:
        raise ValueError("decision probe binding does not match target dimensions")
    covered_bindings = required_bindings & actual_bindings

    projections = plan["projections"]
    if not isinstance(projections, dict) or set(projections) != set(PLAN_STAGES):
        raise ValueError("target plan must retain every stage projection")
    projection_slots = covered_projection_slots = 0
    projected_by_stage = {}
    for stage in PLAN_STAGES:
        view = projections[stage]
        if (not isinstance(view, dict) or view.get("target_signature") != plan["target_signature"] or
                view.get("plan_sha256") != plan["sha256"] or view.get("stage") != stage):
            raise ValueError("target plan projection metadata mismatch")
        view_probe_ids = _ids(view.get("probes"), stage + " projected probes")
        view_claim_ids = _ids(view.get("claims"), stage + " projected claims")
        expected_probe_ids = (set(probe_ids) if stage == "critic" else
                              {item["id"] for item in probes if stage in item["routes"]})
        expected_claim_ids = (set(claim_ids) if stage == "critic" else
                              {by_id[item]["claim_id"] for item in expected_probe_ids})
        projection_slots += len(expected_probe_ids)
        covered_projection_slots += len(expected_probe_ids & set(view_probe_ids))
        if set(view_probe_ids) != expected_probe_ids or set(view_claim_ids) != expected_claim_ids:
            raise ValueError("target plan projection does not match deterministic routing")
        projected_by_stage[stage] = len(expected_probe_ids)

    observed = ({item["stage"] for item in psi} |
                {item["stage"] for item in verification})
    delivered = sum(count for stage, count in projected_by_stage.items()
                    if stage != "critic" and stage in observed)
    routed_total = sum(projected_by_stage[stage] for stage in ROUTED_STAGES)
    judgement_reviewed = len(probe_ids) if any(item["stage"] == "judgement_critic" and
        item["status"] == "accepted" for item in verification) else 0
    return {
        "claims": len(claims),
        "dimensions": sum(len(item.get("dimensions", [])) for item in claims),
        "probes": len(probes),
        "required_probe_slots": len(required_bindings),
        "required_probe_slots_covered": len(covered_bindings),
        "projection_slots": projection_slots,
        "projection_slots_covered": covered_projection_slots,
        "routed_stage_slots": routed_total,
        "routed_stage_slots_delivered_to_executed_stage": delivered,
        "unique_probes_reviewed_by_an_accepted_judgement_critic": judgement_reviewed,
        "projected_probes_by_stage": projected_by_stage,
    }


def _plan_coverage(run, ids, inputs, rows):
    by_id = {row["id"]: row for row in rows}
    totals = Counter()
    cases = []
    for case_id in ids:
        row = by_id[case_id]
        path = run / f"{case_id}-target-plan.json"
        planner = _history(run, case_id, "target-planner-history")
        psi = _history(run, case_id, "psi-history")
        verification = _history(run, case_id, "verification-history")
        item = {"id": case_id, "status": row["status"], "plan_present": path.exists(),
                "plan_valid": False}
        if path.exists():
            audit = _audit_plan(path, inputs[case_id]["target"], row, psi, verification)
            item.update(plan_valid=True, **audit)
            totals.update({key: value for key, value in audit.items()
                           if isinstance(value, int) and not isinstance(value, bool)})
            for stage, count in audit["projected_probes_by_stage"].items():
                totals["projected_probes_" + stage] += count
        elif row["status"] == "completed":
            raise ValueError("Completed extension case has no retained target plan")
        if row["status"] == "completed" and not any(x["stage"] == "extension" and
                x["status"] == "accepted" for x in planner):
            raise ValueError("Completed extension case has no accepted plan review")
        cases.append(item)
    totals["plans_present"] = sum(item["plan_present"] for item in cases)
    totals["plans_valid"] = sum(item["plan_valid"] for item in cases)
    totals["scheduled_cases"] = len(cases)
    for numerator, denominator, name in (
            ("required_probe_slots_covered", "required_probe_slots", "required_probe_coverage"),
            ("projection_slots_covered", "projection_slots", "projection_coverage"),
            ("routed_stage_slots_delivered_to_executed_stage", "routed_stage_slots",
             "routed_stage_delivery_coverage")):
        totals[name] = _ratio(totals[numerator], totals[denominator])
    return {"totals": dict(totals), "cases": cases,
            "meaning": "Structural plan, routing, and stage-delivery coverage only; not semantic correctness or independent evidence."}


def _round_change(cases, matrix, arm):
    table = []
    for case_id, case in cases.items():
        if not _label_evaluable(case):
            continue
        row = matrix[(case_id, arm)]
        first = next((item.get("decision") for item in row.get("checkpoints", [])
                      if item.get("round") == 1), None)
        final = row["prediction"]["decision"] if row["status"] == "completed" else None
        paired = first is not None and final is not None
        table.append({"id": case_id, "reference": case["decision"],
                      "first_round": first, "final": final, "paired": paired,
                      "first_correct": first == case["decision"] if first is not None else None,
                      "final_correct": final == case["decision"] if final is not None else False})
    paired = [item for item in table if item["paired"]]
    return {
        "label_evaluable_scheduled_cases": len(table),
        "paired_completed_cases_with_round1": len(paired),
        "first_round_correct": sum(item["first_correct"] for item in paired),
        "final_correct": sum(item["final_correct"] for item in paired),
        "fixes": sum(not item["first_correct"] and item["final_correct"] for item in paired),
        "breaks": sum(item["first_correct"] and not item["final_correct"] for item in paired),
        "cases": table,
        "denominator_note": "Fixes/breaks use completed cases with an observed round-1 decision; scheduled task success remains in arms.",
    }


def _paired(cases, matrix, baseline, candidate, completed_only=False):
    pairs = []
    ids = []
    for case_id, case in cases.items():
        if not _label_evaluable(case):
            continue
        a, b = matrix[(case_id, baseline)], matrix[(case_id, candidate)]
        if completed_only and (a["status"] != "completed" or b["status"] != "completed"):
            continue
        av = a["status"] == "completed" and a["prediction"]["decision"] == case["decision"]
        bv = b["status"] == "completed" and b["prediction"]["decision"] == case["decision"]
        pairs.append((av, bv))
        ids.append(case_id)
    return {"case_pairs": len(pairs), "case_ids": ids,
            "baseline": baseline, "candidate": candidate,
            "fixes": sum(not a and b for a, b in pairs),
            "breaks": sum(a and not b for a, b in pairs),
            "both_successful": sum(a and b for a, b in pairs),
            "both_unsuccessful": sum(not a and not b for a, b in pairs),
            "baseline_correct": sum(a for a, _ in pairs),
            "candidate_correct": sum(b for _, b in pairs),
            "candidate_minus_baseline": _ratio(sum(int(b) - int(a) for a, b in pairs), len(pairs)),
            "denominator": "shared completed label-evaluable cases" if completed_only else
                           "all selected label-evaluable scheduled cases; errors are failures"}


def _usage_delta(candidate, baseline):
    result = {}
    for key in ("model_calls", "input_tokens", "output_tokens", "reasoning_tokens",
                "visible_output_tokens", "service_seconds"):
        a, b = candidate.get(key), baseline.get(key)
        result[key] = None if a is None or b is None else a - b
    return result


def score(gold_path, original_run, staged_run, extension_run):
    runs = {"original": Path(original_run), "staged": Path(staged_run),
            "extension": Path(extension_run)}
    configs = {arm: _read(path / "config.json") for arm, path in runs.items()}
    schedules = {"original": _schedule(configs["original"], "original"),
                 "staged": _schedule(configs["staged"], "staged"),
                 "extension": _extension_schedule(configs["extension"])}
    selected_ids = schedules["extension"]
    if not set(selected_ids) <= set(schedules["original"]) or not set(selected_ids) <= set(schedules["staged"]):
        raise ValueError("Extension cases are not a subset of both baselines")
    for key in ("model", "reasoning_effort", "budget_per_case", "input_sha256",
                "provider", "dataset_status"):
        values = [configs[arm].get(key) for arm in runs]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"Runs have unequal {key}")
    if configs["extension"]["trace_config"] != configs["staged"]["trace_config"]:
        raise ValueError("Staged and extension runs have unequal trace_config")

    data = {arm: _read(path / "inputs.json") for arm, path in runs.items()}
    original_inputs, original_eligible = _inputs(data["original"], schedules["original"])
    staged_inputs, staged_eligible = _inputs(data["staged"], schedules["staged"])
    extension_inputs, extension_eligible = _inputs(data["extension"], selected_ids)
    selected_inputs = {case_id: original_inputs[case_id] for case_id in selected_ids}
    if ({case_id: staged_inputs[case_id] for case_id in selected_ids} != selected_inputs or
            extension_inputs != selected_inputs):
        raise ValueError("Runs have different selected corpus or task contracts")
    gold = _read(Path(gold_path))
    cases = _references(gold, selected_inputs)

    response_ids = set()
    raw_rows = {
        "original": _rows(runs["original"], configs["original"], schedules["original"],
                          original_inputs, "original", response_ids),
        "staged": _rows(runs["staged"], configs["staged"], schedules["staged"],
                        staged_inputs, "staged", response_ids),
        "extension": _rows(runs["extension"], configs["extension"], selected_ids,
                           extension_inputs, "staged", response_ids),
    }
    raw_by_id = {arm: {row["id"]: row for row in rows} for arm, rows in raw_rows.items()}
    matrix = {}
    for case_id in selected_ids:
        original = raw_by_id["original"][case_id]
        if original["status"] == "completed":
            audit = _original(original, _read(runs["original"] / f"{case_id}-report.json"),
                              original_eligible[case_id])
            if not audit["native_flags_boolean"]:
                original = deepcopy(original)
                original["prediction"]["origin_evaluable"] = False
        matrix[(case_id, "original")] = original
        for arm, eligible in (("staged", staged_eligible), ("extension", extension_eligible)):
            row = raw_by_id[arm][case_id]
            normalized = row
            if row["status"] == "completed":
                normalized = _staged(row, _read(runs[arm] / f"{case_id}-report.json"),
                                     configs[arm], selected_inputs[case_id], eligible[case_id])
            matrix[(case_id, arm)] = normalized

    arms = {arm: _arm_score(cases, matrix, arm) for arm in runs}
    arms["original"]["edges"].update(status="unsupported_native_export", precision=None,
        recall=None, note="The original agent has no native source-to-source edge representation.")
    usage = {arm: _calls_usage(runs[arm], selected_ids,
                               [raw_by_id[arm][case_id] for case_id in selected_ids])
             for arm in runs}
    for arm in arms:
        arms[arm]["usage_total"] = usage[arm]
        arms[arm]["usage_mean_per_scheduled_case"] = {
            key: (value / len(selected_ids) if isinstance(value, (int, float)) and
                  not isinstance(value, bool) else value)
            for key, value in usage[arm].items()}

    loop_audit = {arm: _loop_counts(runs[arm], selected_ids)
                  for arm in ("staged", "extension")}
    coverage = _plan_coverage(runs["extension"], selected_ids, extension_inputs,
                              [raw_by_id["extension"][case_id] for case_id in selected_ids])
    label_pairs = {}
    for baseline, candidate in (("original", "staged"), ("original", "extension"),
                                ("staged", "extension")):
        name = candidate + "_vs_" + baseline
        label_pairs[name] = {"all_scheduled": _paired(cases, matrix, baseline, candidate),
                             "shared_completed": _paired(cases, matrix, baseline, candidate, True)}

    return {
        "schema_version": "target-extension-compare-score-v1",
        "comparison": "target_extension_vs_staged_and_original_development",
        "selected_case_ids": selected_ids,
        "scheduled_cases_per_arm": len(selected_ids),
        "model": configs["extension"]["model"],
        "reasoning_effort": configs["extension"]["reasoning_effort"],
        "arms": arms,
        "label_comparisons": label_pairs,
        "round1_to_final": {arm: _round_change(cases, matrix, arm)
                            for arm in ("staged", "extension")},
        "usage_extension_minus_staged": _usage_delta(usage["extension"], usage["staged"]),
        "usage_extension_minus_original": _usage_delta(usage["extension"], usage["original"]),
        "loop_audit": loop_audit,
        "target_plan_probe_coverage": coverage,
        "reference_sha256": _hash(Path(gold_path)),
        "artifacts_sha256": {arm: {name: _hash(path / name)
            for name in ("config.json", "inputs.json", "results.json")}
            for arm, path in runs.items()},
        "conclusion": "development_diagnostic_only_no_proof_of_accuracy_improvement",
        "limits": [
            "Previously seen development cases and provisional AI-reviewed references; not an independently adjudicated held-out benchmark.",
            "The extension run selects the denominator. Baseline totals are recalculated on exactly those cases, not copied from full-run summaries.",
            "Equal configured caps do not imply equal actual spend. service_seconds is summed per-call/client elapsed service time, not wall-clock experiment duration.",
            "Reasoning tokens are provider-reported token counts, not a quality measure. More or fewer reasoning tokens cannot by itself establish accuracy.",
            "Probe coverage is structural checklist/routing coverage. It does not show that a probe was answered correctly or grounded in independent evidence.",
            "Origins and edges are corpus-relative. The original agent has no native edge export, so its edge accuracy remains unsupported.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--original-run", required=True)
    parser.add_argument("--staged-run", required=True)
    parser.add_argument("--extension-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = score(args.gold, args.original_run, args.staged_run, args.extension_run)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"scheduled_cases_per_arm": result["scheduled_cases_per_arm"],
        "completed": {arm: value["completed"] for arm, value in result["arms"].items()},
        "correct": {arm: value["correct"] for arm, value in result["arms"].items()},
        "required_probe_coverage": result["target_plan_probe_coverage"]["totals"].get(
            "required_probe_coverage")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
