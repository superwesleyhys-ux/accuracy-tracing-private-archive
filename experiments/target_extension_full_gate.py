"""Deterministic, artifact-only gate for a frozen v3/v4 eight-case full run.

The gate never imports an inference adapter and never calls a model or network
service.  It recomputes the independent comparison score from retained run
artifacts, verifies both the current frozen tree and the captured runtime tree,
and then applies the preregistered full-run conjuncts.  Resource use is required
to be present and internally consistent, but calls, tokens, and elapsed service
time are reported rather than optimized against a post-hoc threshold.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path, PurePosixPath
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from target_extension_compare_score import score as comparison_score
from target_extension_smoke_gate import (
    PLAN_SCHEMA_V4,
    V4_EXPERIMENT_BY_PROVIDER,
    V4_PROVIDER_LABELS,
    _call_audit,
    _case_inputs,
    _checkpoint_audit,
    _exact_config,
    _execution_control_declarations,
    _frozen_baseline_audit,
    _frozen_hash_audit,
    _frozen_prior_artifact_audit,
    _gate_secret_paths,
    _outer_loop_audit,
    _preserved_run_declarations,
    _rows,
    _sha256,
    _secret_shape_audit,
    _status_audit,
    _v4_audit_state,
    check_smoke_gate,
)


PLAN_SCHEMA_V3 = "decision-probe-v3"
EXPERIMENT_V3 = "target_extended_psi_development_v3"
EXPECTED_CASE_COUNT = 8
RUNTIME_EXPERIMENT_FILES = {
    "experiments/extended_semantic.py",
    "experiments/loop_compare.py",
    "experiments/model_io.py",
    "experiments/semantic_adapter.py",
    "experiments/staged_run.py",
    "experiments/staged_semantic.py",
}
USAGE_FIELDS = (
    "model_calls",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "visible_output_tokens",
    "service_seconds",
)


def _read(path: Path) -> Any:
    """Read JSON while rejecting duplicate keys at every nesting level."""
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {path.name}: {key}")
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=unique)


class _Checks:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(self, identifier: str, passed: bool, expected: Any, actual: Any,
            detail: str = "") -> None:
        self.items.append({
            "id": identifier,
            "passed": bool(passed),
            "expected": expected,
            "actual": actual,
            "detail": detail,
        })

    def finish(self, comparison_error: str | None = None,
               usage: dict[str, Any] | None = None) -> dict[str, Any]:
        failed = [item["id"] for item in self.items if not item["passed"]]
        return {
            "schema_version": "target-extension-full-gate-v1",
            "passed": not failed,
            "checks_run": len(self.items),
            "checks_passed": len(self.items) - len(failed),
            "failed_checks": failed,
            "checks": self.items,
            "comparison_audit_error": comparison_error,
            "reported_usage": usage,
            "interpretation": (
                "Passing establishes the frozen eight-case development contract only. "
                "It is not proof of a general accuracy increase or of a causal benefit "
                "from a second verification round."
            ),
        }


def _safe_relative_report_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return (not path.is_absolute() and path.parts and path.parts[0] == "reports" and
            ".." not in path.parts and "." not in path.parts)


def _freeze_contract(freeze: dict[str, Any], gold: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    full_cases = freeze.get("full_cases")
    gold_rows = gold.get("cases") if isinstance(gold, dict) else None
    gold_ids = [item.get("id") for item in gold_rows
                if isinstance(item, dict)] if isinstance(gold_rows, list) else []
    repair_fields = (
        "max_target_structure_repairs",
        "max_target_extension_repairs",
        "max_extension_output_repairs",
        "max_material_structure_repairs_per_return",
        "max_material_semantic_repairs",
        "max_judgement_repairs",
        "max_probe_result_structure_repairs",
    )
    budget = freeze.get("budget_per_case")
    budget_ok = (isinstance(budget, dict) and set(budget) == {
        "calls", "output_tokens", "per_call_output_tokens", "seconds"} and
        all(isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(value) and value > 0 for value in budget.values()) and
        budget.get("per_call_output_tokens", math.inf) <=
        budget.get("output_tokens", -math.inf))
    repairs = {field: freeze.get(field) for field in repair_fields}
    repairs_ok = all(type(value) is int and value in (0, 1)
                     for value in repairs.values())
    cases_ok = (isinstance(full_cases, list) and len(full_cases) == EXPECTED_CASE_COUNT and
                len(full_cases) == len(set(full_cases)) and
                all(isinstance(case_id, str) and case_id for case_id in full_cases) and
                len(gold_ids) == EXPECTED_CASE_COUNT and len(gold_ids) == len(set(gold_ids)) and
                full_cases == gold_ids and all(
                    isinstance(item.get("decision"), str) for item in gold_rows))
    declared_files = freeze.get("files")
    baselines = freeze.get("frozen_baselines")
    prior = freeze.get("prior_artifacts")
    actual = {
        "status": freeze.get("status"),
        "experiment": freeze.get("experiment"),
        "target_plan_schema": freeze.get("target_plan_schema"),
        "model": freeze.get("model"),
        "reasoning_effort": freeze.get("reasoning_effort"),
        "outer_rounds": freeze.get("outer_rounds"),
        "workers": freeze.get("workers"),
        "automatic_transport_retries": freeze.get("automatic_transport_retries"),
        "provider_mode": freeze.get("provider_mode"),
        "strict_retrieval_attribution": freeze.get("strict_retrieval_attribution"),
        "retrieval_attribution_mode": freeze.get("retrieval_attribution_mode"),
        "provider": freeze.get("provider"),
        "full_cases": full_cases,
        "gold_case_ids": gold_ids,
        "follow_on_run": freeze.get("follow_on_run"),
        "budget_per_case": budget,
        "repair_caps": repairs,
        "frozen_file_count": len(declared_files) if isinstance(declared_files, dict) else None,
        "fixed_baseline_arms": sorted(baselines) if isinstance(baselines, dict) else None,
        "prior_artifact_count": len(prior) if isinstance(prior, dict) else None,
    }
    schema = freeze.get("target_plan_schema")
    provider_mode = freeze.get("provider_mode")
    if schema == PLAN_SCHEMA_V4:
        version_ok = (
            provider_mode in V4_EXPERIMENT_BY_PROVIDER and
            freeze.get("experiment") == V4_EXPERIMENT_BY_PROVIDER.get(provider_mode) and
            freeze.get("provider") == V4_PROVIDER_LABELS.get(provider_mode) and
            freeze.get("strict_retrieval_attribution") is
                (provider_mode == "task_routed") and
            freeze.get("retrieval_attribution_mode") ==
                ("strict" if provider_mode == "task_routed" else "legacy")
        )
    else:
        version_ok = schema == PLAN_SCHEMA_V3 and freeze.get("experiment") == EXPERIMENT_V3
    passed = (
        freeze.get("status") == "frozen_before_live_calls" and
        version_ok and
        freeze.get("model") == "gpt-6-astra" and
        freeze.get("reasoning_effort") == "medium" and
        freeze.get("outer_rounds") == 2 and
        freeze.get("workers") == 1 and
        freeze.get("automatic_transport_retries") == 0 and cases_ok and
        _safe_relative_report_path(freeze.get("follow_on_run")) and budget_ok and
        repairs_ok and isinstance(declared_files, dict) and bool(declared_files) and
        isinstance(baselines, dict) and set(baselines) == {"original", "staged"} and
        isinstance(prior, dict) and bool(prior)
    )
    return passed, actual


def _baseline_manifest_audit(freeze: dict[str, Any],
                             case_ids: list[str]) -> list[dict[str, Any]]:
    """Ensure every baseline artifact that can affect scored outcomes is frozen."""
    declared = freeze.get("frozen_baselines")
    if not isinstance(declared, dict):
        return [{"error": "missing frozen baseline manifest"}]
    required = {"config.json", "inputs.json", "results.json"}
    required |= {f"{case_id}-{suffix}.json" for case_id in case_ids
                 for suffix in ("calls", "report", "result")}
    errors: list[dict[str, Any]] = []
    for arm in ("original", "staged"):
        files = declared.get(arm, {}).get("files")
        actual = set(files) if isinstance(files, dict) else set()
        arm_required = set(required)
        if arm == "staged":
            arm_required |= {f"{case_id}-{suffix}.json" for case_id in case_ids
                             for suffix in ("psi-history", "verification-history")}
        missing = sorted(arm_required - actual)
        if missing:
            errors.append({"arm": arm, "missing": missing})
    return errors


def _exact_full_config(freeze: dict[str, Any], config: dict[str, Any]) -> tuple[bool, Any]:
    shadow = deepcopy(freeze)
    shadow["smoke_cases"] = freeze.get("full_cases")
    base_ok, actual = _exact_config(shadow, config)
    actual = {**actual, "target_extension": config.get("target_extension")}
    return base_ok and config.get("target_extension") is True, actual


def _runtime_hash_audit(freeze: dict[str, Any], extension_run: Path,
                        config: dict[str, Any]) -> list[dict[str, Any]]:
    """Require the runner's complete captured source set to match the freeze.

    The current-tree audit covers scorer, gates, tests and documentation.  The
    executed tree intentionally contains only code imported by the live runner;
    this function prevents a partial ``source_sha256`` declaration from making
    that distinction look like a pass.
    """
    declared = freeze.get("files")
    recorded = config.get("source_sha256")
    if not isinstance(declared, dict) or not isinstance(recorded, dict):
        return [{"error": "missing frozen or executed source hash map"}]
    expected_paths = {relative for relative in declared
                      if relative.startswith("newsverify/") and relative.endswith(".py")}
    expected_paths |= RUNTIME_EXPERIMENT_FILES
    mismatches: list[dict[str, Any]] = []
    if set(recorded) != expected_paths:
        mismatches.append({"file": "config.source_sha256.keys",
                           "expected": sorted(expected_paths),
                           "actual": sorted(recorded)})
    for relative in sorted(expected_paths):
        expected = declared.get(relative)
        recorded_hash = recorded.get(relative)
        captured = extension_run / "executed-code" / relative
        captured_hash = _sha256(captured) if captured.is_file() else None
        if not isinstance(expected, str) or recorded_hash != expected or captured_hash != expected:
            mismatches.append({"file": relative, "expected": expected,
                               "recorded": recorded_hash, "captured": captured_hash})
    return mismatches


def _required_artifacts(extension_run: Path, case_ids: list[str],
                        required_rounds: list[int], *, v4: bool = False,
                        adaptive: bool = False) -> list[str]:
    required = [extension_run / name for name in
                ("config.json", "inputs.json", "results.json", "status.json")]
    suffixes = ["calls", "report", "result", "target-plan", "verification-history"]
    if v4:
        suffixes.extend(("psi-history", "retrieval"))
    required.extend(extension_run / f"{case_id}-{suffix}.json"
                    for case_id in case_ids for suffix in suffixes)
    checkpoint_rounds = [1] if adaptive else required_rounds
    required.extend(extension_run / f"{case_id}-checkpoint-{round_number}.json"
                    for case_id in case_ids for round_number in checkpoint_rounds)
    return sorted(path.name for path in required if not path.is_file())


def _strict_json_audit(extension_run: Path, case_ids: list[str],
                       required_rounds: list[int], *, v4: bool = False,
                       adaptive: bool = False) -> list[dict[str, Any]]:
    names = ["config.json", "inputs.json", "results.json", "status.json"]
    suffixes = ["calls", "report", "result", "target-plan", "verification-history"]
    if v4:
        suffixes.extend(("psi-history", "retrieval"))
    names.extend(f"{case_id}-{suffix}.json" for case_id in case_ids
                 for suffix in suffixes)
    checkpoint_rounds = [1] if adaptive else required_rounds
    names.extend(f"{case_id}-checkpoint-{round_number}.json"
                 for case_id in case_ids for round_number in checkpoint_rounds)
    if adaptive:
        names.extend(path.name for case_id in case_ids
                     for path in sorted(extension_run.glob(
                         f"{case_id}-checkpoint-*.json"))
                     if path.name not in names)
    errors: list[dict[str, Any]] = []
    for name in names:
        try:
            _read(extension_run / name)
        except Exception as exc:
            errors.append({"file": name, "error": f"{type(exc).__name__}: {exc}"})
    return errors


def _valid_usage(comparison: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    usage = comparison.get("arms", {}).get("extension", {}).get("usage_total")
    if not isinstance(usage, dict):
        return False, {"error": "missing extension usage_total"}
    state = {field: usage.get(field) for field in USAGE_FIELDS}
    integer_fields = set(USAGE_FIELDS) - {"service_seconds"}
    valid = True
    for field, value in state.items():
        if (not isinstance(value, (int, float)) or isinstance(value, bool) or
                not math.isfinite(value) or value < 0 or
                (field in integer_fields and int(value) != value)):
            valid = False
    valid = (valid and state["model_calls"] > 0 and state["input_tokens"] > 0 and
             state["output_tokens"] > 0 and state["service_seconds"] > 0 and
             usage.get("reasoning_reporting_complete") is True)
    return valid, {**state,
                   "reasoning_reporting_complete":
                       usage.get("reasoning_reporting_complete")}


def _comparison_hash_state(freeze: dict[str, Any], comparison: dict[str, Any],
                           extension_run: Path) -> tuple[bool, dict[str, Any]]:
    artifacts = comparison.get("artifacts_sha256")
    baselines = freeze.get("frozen_baselines")
    if not isinstance(artifacts, dict) or not isinstance(baselines, dict):
        return False, {"error": "missing comparison or baseline artifact hashes"}
    expected: dict[str, dict[str, Any]] = {}
    for arm in ("original", "staged"):
        files = baselines.get(arm, {}).get("files", {})
        expected[arm] = {name: files.get(name)
                         for name in ("config.json", "inputs.json", "results.json")}
    expected["extension"] = {name: _sha256(extension_run / name)
                             for name in ("config.json", "inputs.json", "results.json")}
    actual = {arm: artifacts.get(arm) for arm in ("original", "staged", "extension")}
    reference_ok = comparison.get("reference_sha256") == freeze.get(
        "provisional_reference_sha256")
    return actual == expected and reference_ok, {
        "expected": expected,
        "actual": actual,
        "expected_reference": freeze.get("provisional_reference_sha256"),
        "actual_reference": comparison.get("reference_sha256"),
    }


def _add_comparison_checks(checks: _Checks, freeze: dict[str, Any],
                           comparison: dict[str, Any], extension_run: Path,
                           required_rounds: list[int] | dict[str, list[int]],
                           search_rounds: dict[str, int] | None = None,
                           probe_owned_novel_second_pass_cases: set[str] | None = None
                           ) -> dict[str, Any] | None:
    cases = freeze["full_cases"]
    scheduled = len(cases)
    metadata = {
        "selected_case_ids": comparison.get("selected_case_ids"),
        "scheduled_cases_per_arm": comparison.get("scheduled_cases_per_arm"),
        "model": comparison.get("model"),
        "reasoning_effort": comparison.get("reasoning_effort"),
    }
    checks.add("comparison.frozen_denominator",
               metadata == {"selected_case_ids": cases,
                            "scheduled_cases_per_arm": scheduled,
                            "model": freeze.get("model"),
                            "reasoning_effort": freeze.get("reasoning_effort")},
               {"selected_case_ids": cases, "scheduled_cases_per_arm": scheduled,
                "model": freeze.get("model"),
                "reasoning_effort": freeze.get("reasoning_effort")}, metadata)

    hash_ok, hash_state = _comparison_hash_state(freeze, comparison, extension_run)
    checks.add("comparison.artifact_hashes", hash_ok,
               "scorer hashes equal frozen baselines, full-run aggregates and reference",
               hash_state)

    arms = comparison.get("arms", {})
    baseline_state = {arm: {key: arms.get(arm, {}).get(key)
                            for key in ("scheduled_cases", "completed", "execution_errors")}
                      for arm in ("original", "staged")}
    checks.add("labels.fixed_baselines_complete", all(value == {
        "scheduled_cases": scheduled, "completed": scheduled, "execution_errors": 0}
        for value in baseline_state.values()),
        f"both frozen baselines complete all {scheduled} cases without execution errors",
        baseline_state)

    extension = arms.get("extension", {})
    extension_state = {key: extension.get(key) for key in
                       ("scheduled_cases", "completed", "execution_errors",
                        "label_evaluable_cases", "correct")}
    checks.add("labels.final_provisional_accuracy", extension_state == {
        "scheduled_cases": scheduled, "completed": scheduled, "execution_errors": 0,
        "label_evaluable_cases": scheduled, "correct": scheduled},
        f"extension completes and agrees with {scheduled}/{scheduled} provisional labels",
        extension_state)

    pair_state = {}
    pair_ok = True
    for baseline in ("original", "staged"):
        values = comparison.get("label_comparisons", {}).get(
            f"extension_vs_{baseline}", {}).get("all_scheduled", {})
        compact = {key: values.get(key) for key in
                   ("case_pairs", "case_ids", "fixes", "breaks", "candidate_correct")}
        pair_state[baseline] = compact
        fixes = compact["fixes"]
        pair_ok &= (compact["case_pairs"] == scheduled and compact["case_ids"] == cases and
                    type(fixes) is int and 0 <= fixes <= scheduled and
                    compact["breaks"] == 0 and compact["candidate_correct"] == scheduled)
    checks.add("labels.zero_baseline_breaks", pair_ok,
               "eight paired cases, zero breaks against each frozen baseline; fixes allowed",
               pair_state)

    round_change = comparison.get("round1_to_final", {}).get("extension", {})
    round_state = {key: round_change.get(key) for key in
                   ("label_evaluable_scheduled_cases",
                    "paired_completed_cases_with_round1", "final_correct", "fixes", "breaks")}
    fixes = round_state["fixes"]
    round_ok = (round_state["label_evaluable_scheduled_cases"] == scheduled and
                round_state["paired_completed_cases_with_round1"] == scheduled and
                round_state["final_correct"] == scheduled and type(fixes) is int and
                0 <= fixes <= scheduled and round_state["breaks"] == 0)
    provider_mode = freeze.get("provider_mode")
    if freeze.get("target_plan_schema") == PLAN_SCHEMA_V4:
        round_cases = round_change.get("cases")
        per_case_ok = (isinstance(round_cases, list) and len(round_cases) == scheduled and
                       [item.get("id") if isinstance(item, dict) else None
                        for item in round_cases] == cases)
        expected_fixes = 0
        if per_case_ok:
            for item in round_cases:
                first = item.get("first_round")
                final = item.get("final")
                case_id = item["id"]
                row_ok = (isinstance(first, str) and isinstance(final, str) and
                          item.get("paired") is True and
                          item.get("final_correct") is True and
                          item.get("first_correct") is
                          (item.get("reference") == first))
                if provider_mode == "fixed_reanalysis":
                    row_ok &= item.get("first_correct") is True and first == final
                elif (probe_owned_novel_second_pass_cases is not None and
                      first != final):
                    row_ok &= case_id in probe_owned_novel_second_pass_cases
                per_case_ok &= row_ok
                expected_fixes += int(item.get("first_correct") is False and
                                      item.get("final_correct") is True)
        round_ok &= (per_case_ok and fixes == expected_fixes and
                     (provider_mode != "fixed_reanalysis" or fixes == 0))
        round_state["cases"] = round_cases
        round_state["probe_owned_novel_second_pass_cases"] = (
            sorted(probe_owned_novel_second_pass_cases)
            if probe_owned_novel_second_pass_cases is not None else None)
    checks.add("labels.zero_round_breaks", round_ok,
               ("all cases retain round 1 and finish correct with zero breaks; any label change belongs to a real second verification"
                if provider_mode == "task_routed" else
                "all cases retain round 1 and finish correct with zero breaks; fixed v4 reanalysis has no round change"
                if provider_mode == "fixed_reanalysis" else
                "all cases retain round 1 and finish correct with zero breaks; fixes allowed"),
               round_state)

    coverage = comparison.get("target_plan_probe_coverage", {})
    totals = coverage.get("totals", {})
    plan_cases = coverage.get("cases", [])
    expected_plan_schema = freeze.get("target_plan_schema")
    plan_state = {
        "plans_present": totals.get("plans_present"),
        "plans_valid": totals.get("plans_valid"),
        "case_ids": [item.get("id") for item in plan_cases if isinstance(item, dict)],
        "contracts": [item.get("plan_contract") for item in plan_cases
                      if isinstance(item, dict)],
        "terminal_roots": [item.get("terminal_origin_audit", {}).get("terminal_roots")
                           if isinstance(item.get("terminal_origin_audit"), dict) else None
                           for item in plan_cases if isinstance(item, dict)],
    }
    plans_ok = (totals.get("plans_present") == scheduled and
                totals.get("plans_valid") == scheduled and len(plan_cases) == scheduled and
                plan_state["case_ids"] == cases and
                all(isinstance(item, dict) and item.get("plan_valid") is True and
                    item.get("plan_contract") == expected_plan_schema and
                    isinstance(item.get("terminal_origin_audit"), dict) and
                    item["terminal_origin_audit"].get("terminal_roots") is True
                    for item in plan_cases))
    plan_check_id = ("probe_audit.eight_valid_v4_plans"
                     if expected_plan_schema == PLAN_SCHEMA_V4
                     else "probe_audit.eight_valid_v3_plans")
    checks.add(plan_check_id, plans_ok,
               f"eight independently valid {expected_plan_schema} plans with terminal-root reconstruction",
               plan_state)

    structural = {key: totals.get(key) for key in
                  ("required_probe_coverage", "projection_coverage",
                   "executed_stage_projection_coverage")}
    checks.add("probe_audit.structural_coverage",
               all(value == 1.0 for value in structural.values()),
               "binding, projection and executed-stage coverage are each 1.0", structural)

    results = comparison.get("target_probe_result_audit", {}).get("totals", {})
    case_cycle_counts = {
        case_id: (len(required_rounds.get(case_id, []))
                  if isinstance(required_rounds, dict) else len(required_rounds))
        for case_id in cases}
    expected_cycles = sum(case_cycle_counts.values())
    cycle_state = {key: results.get(key) for key in
                   ("cases_with_results", "scheduled_cases",
                    "accepted_judgement_cycles", "audited_judgement_cycles")}
    cycle_check_id = ("probe_audit.adaptive_cycles"
                      if isinstance(required_rounds, dict) else
                      "probe_audit.sixteen_cycles")
    checks.add(cycle_check_id, cycle_state == {
        "cases_with_results": scheduled, "scheduled_cases": scheduled,
        "accepted_judgement_cycles": expected_cycles,
        "audited_judgement_cycles": expected_cycles},
        f"{expected_cycles} accepted and independently audited judgement cycles",
        cycle_state)
    result_cases = comparison.get("target_probe_result_audit", {}).get("cases", [])
    per_case_state = [{key: item.get(key) for key in
                       ("id", "status", "available", "accepted_judgement_cycles",
                        "audited_judgement_cycles", "result_slot_coverage",
                        "results_with_basis", "result_slots",
                        "grounded_conclusive_rate")}
                      for item in result_cases if isinstance(item, dict)]
    per_case_ok = (len(result_cases) == scheduled and
                   [item.get("id") for item in result_cases] == cases and all(
                       isinstance(item, dict) and item.get("status") == "completed" and
                       item.get("available") is True and
                       item.get("accepted_judgement_cycles") ==
                           case_cycle_counts.get(item.get("id")) and
                       item.get("audited_judgement_cycles") ==
                           case_cycle_counts.get(item.get("id")) and
                       item.get("result_slot_coverage") == 1.0 and
                       item.get("results_with_basis") == item.get("result_slots") and
                       item.get("grounded_conclusive_rate") == 1.0
                       for item in result_cases))
    checks.add("probe_audit.per_case_cycles_and_coverage", per_case_ok,
               "each case has exactly its artifact-observed audited cycles and full coverage",
               per_case_state)
    expected_slots, result_slots = (results.get("expected_result_slots"),
                                    results.get("result_slots"))
    per_stage = results.get("by_stage", {})
    result_state = {
        "expected_result_slots": expected_slots,
        "result_slots": result_slots,
        "results_with_basis": results.get("results_with_basis"),
        "result_slot_coverage": results.get("result_slot_coverage"),
        "grounded_conclusive_rate": results.get("grounded_conclusive_rate"),
        "stage_coverage": {stage: per_stage.get(stage, {}).get("result_slot_coverage")
                           for stage in ("evidence", "world")},
    }
    result_ok = (type(expected_slots) is int and expected_slots > 0 and
                 result_slots == expected_slots and
                 results.get("results_with_basis") == result_slots and
                 results.get("result_slot_coverage") == 1.0 and
                 results.get("grounded_conclusive_rate") == 1.0 and
                 all(value == 1.0 for value in result_state["stage_coverage"].values()))
    checks.add("probe_audit.result_and_grounding_coverage", result_ok,
               "result-slot, retained-basis, per-stage and grounded-conclusive coverage are 1.0",
               result_state)

    if expected_plan_schema == PLAN_SCHEMA_V4:
        v4_ok, v4_state = _v4_audit_state(
            comparison, cases, required_rounds,
            search_rounds=search_rounds,
            provider_mode=freeze.get("provider_mode"),
            probe_owned_novel_second_pass_cases=
                probe_owned_novel_second_pass_cases)
        checks.add("probe_audit.v4_end_to_end_ledgers", v4_ok,
                   "all eight cases have complete target/material/follow-up/retrieval/probe-delta ledgers with zero breaks",
                   v4_state)

    origins = extension.get("origins", {})
    origin_state = {key: origins.get(key) for key in
                    ("evaluable_cases", "matched_predictions", "predicted_items",
                     "precision", "recall", "recall_denominator")}
    origin_ok = (type(origin_state["evaluable_cases"]) is int and
                 origin_state["evaluable_cases"] > 0 and
                 type(origin_state["recall_denominator"]) is int and
                 origin_state["recall_denominator"] > 0 and
                 origin_state["precision"] == 1.0 and origin_state["recall"] == 1.0)
    checks.add("provenance.origins", origin_ok,
               "nonempty evaluable origin precision and recall are both 1.0", origin_state)

    edges = extension.get("edges", {})
    edge_state = {key: edges.get(key) for key in
                  ("evaluable_cases", "matched_predictions", "predicted_items",
                   "precision", "recall", "gold_edges", "recall_denominator")}
    edge_ok = (type(edge_state["evaluable_cases"]) is int and
               edge_state["evaluable_cases"] > 0 and type(edge_state["gold_edges"]) is int and
               edge_state["gold_edges"] > 0 and edge_state["precision"] == 1.0 and
               edge_state["recall"] == 1.0)
    checks.add("provenance.direct_edges", edge_ok,
               "nonempty evaluable direct-edge precision and recall are both 1.0", edge_state)

    usage_ok, usage = _valid_usage(comparison)
    checks.add("resources.reported_not_thresholded", usage_ok,
               "positive calls/input/output/time and complete nonnegative token accounting; "
               "no efficiency threshold", usage)
    return usage


def _evaluate_gate(freeze: dict[str, Any], comparison: dict[str, Any] | None,
                   comparison_error: str | None, freeze_path: Path, gold_path: Path,
                   original_run: Path, staged_run: Path,
                   extension_run: Path, smoke_result: dict[str, Any] | None = None,
                   smoke_error: str | None = None) -> dict[str, Any]:
    checks = _Checks()
    repo_root = freeze_path.resolve().parents[1]
    gold = _read(gold_path)
    contract_ok, contract_state = _freeze_contract(freeze, gold)
    checks.add("freeze.full_contract", contract_ok,
               "frozen v3/v4 Astra/medium two-round eight-case contract with fixed hashes and caps",
               contract_state)

    case_ids = freeze.get("full_cases") if isinstance(freeze.get("full_cases"), list) else []
    rounds = freeze.get("outer_rounds")
    required_rounds = (list(range(1, rounds + 1)) if type(rounds) is int and rounds > 0 else [])
    try:
        relative_run = extension_run.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        relative_run = str(extension_run.resolve())
    checks.add("freeze.follow_on_run", relative_run == freeze.get("follow_on_run"),
               freeze.get("follow_on_run"), relative_run)

    smoke_state = ({key: smoke_result.get(key) for key in
                    ("schema_version", "passed", "checks_run", "checks_passed",
                     "failed_checks")} if isinstance(smoke_result, dict) else smoke_error)
    checks.add("freeze.smoke_prerequisite",
               isinstance(smoke_result, dict) and smoke_result.get("passed") is True and
               smoke_result.get("schema_version") == "target-extension-smoke-gate-v1",
               "the frozen smoke run still passes a fresh artifact-only recomputation",
               smoke_state)

    current_mismatches, current_skipped = _frozen_hash_audit(
        freeze, repo_root, gold_path, extension_run, "current")
    checks.add("freeze.current_hashes", not current_mismatches and not current_skipped,
               "every current frozen file plus inputs/reference matches exactly",
               {"mismatches": current_mismatches, "skipped": current_skipped})
    executed_mismatches, executed_skipped = _frozen_hash_audit(
        freeze, repo_root, gold_path, extension_run, "executed")
    checks.add("freeze.executed_hashes", not executed_mismatches,
               "captured/recorded runtime hashes plus inputs/reference match exactly",
               {"mismatches": executed_mismatches,
                "intentionally_current_only": executed_skipped},
               "Scorer, gate, test and documentation files are current-tree artifacts, not "
               "runner imports; the strict captured runtime set is checked separately.")

    baselines_declared = (isinstance(freeze.get("frozen_baselines"), dict) and
                          set(freeze["frozen_baselines"]) == {"original", "staged"})
    baseline_mismatches = _frozen_baseline_audit(
        freeze, repo_root, original_run, staged_run)
    baseline_manifest_errors = _baseline_manifest_audit(freeze, case_ids)
    checks.add("freeze.baseline_hashes", baselines_declared and
               not baseline_mismatches and not baseline_manifest_errors,
               "both exact baselines, including every scored call/report/result ledger",
               {"hash_mismatches": baseline_mismatches,
                "manifest_errors": baseline_manifest_errors}
               if baselines_declared else "missing declaration")
    prior_declared = isinstance(freeze.get("prior_artifacts"), dict) and bool(
        freeze.get("prior_artifacts"))
    prior_mismatches = _frozen_prior_artifact_audit(freeze, repo_root)
    checks.add("freeze.prior_artifact_hashes", prior_declared and not prior_mismatches,
               "every preregistered prior comparison/gate artifact",
               prior_mismatches if prior_declared else "missing declaration")
    preserved = _preserved_run_declarations(freeze)
    missing_preserved = [relative for relative in preserved
                         if not (repo_root / relative).is_dir()]
    checks.add("freeze.preserved_run_directories", not missing_preserved,
               "every declared preserved run directory remains present", missing_preserved)

    secret_findings = _secret_shape_audit(
        _gate_secret_paths(freeze, repo_root, freeze_path, gold_path,
                           original_run, staged_run, extension_run), repo_root)
    checks.add("security.no_secret_shapes", not secret_findings,
               "no recognized API-key or access-token shape occurs in frozen inputs, source, baselines, prior artifacts, or full-run artifacts",
               secret_findings,
               "Findings contain only paths and detector names, never matched bytes.")

    v4 = freeze.get("target_plan_schema") == PLAN_SCHEMA_V4
    adaptive = v4 and freeze.get("provider_mode") == "task_routed"
    missing = _required_artifacts(
        extension_run, case_ids, required_rounds, v4=v4, adaptive=adaptive)
    checks.add("execution.required_artifacts", not missing,
               "all aggregate, per-case and checkpoint full-run artifacts", missing)
    if missing or not contract_ok:
        checks.add("comparison.independent_audit", False,
                   "independent comparison scorer completes", comparison_error or
                   "not evaluated after a malformed freeze or missing artifact")
        return checks.finish(comparison_error)

    strict_errors = _strict_json_audit(
        extension_run, case_ids, required_rounds, v4=v4, adaptive=adaptive)
    checks.add("execution.strict_json", not strict_errors,
               "every required run artifact parses with no duplicate JSON keys",
               strict_errors)
    if strict_errors:
        checks.add("comparison.independent_audit", False,
                   "independent comparison scorer completes", comparison_error or
                   "not evaluated after strict JSON failure")
        return checks.finish(comparison_error)

    config = _read(extension_run / "config.json")
    inputs = _case_inputs(_read(extension_run / "inputs.json"))
    rows = _rows(_read(extension_run / "results.json"))
    status = _read(extension_run / "status.json")
    reports = {case_id: _read(extension_run / f"{case_id}-report.json")
               for case_id in case_ids}
    retrievals = ({case_id: _read(extension_run / f"{case_id}-retrieval.json")
                   for case_id in case_ids} if v4 else {})

    config_ok, config_state = _exact_full_config(freeze, config)
    checks.add("execution.frozen_config", config_ok,
               "full-run configuration equals the complete frozen contract", config_state)
    controls_ok, expected_controls, actual_controls = _execution_control_declarations(
        freeze, config, extension_run)
    checks.add("execution.transport_and_repair_declarations", controls_ok,
               expected_controls, actual_controls)
    runtime_hash_errors = _runtime_hash_audit(freeze, extension_run, config)
    checks.add("execution.captured_runtime_hashes", not runtime_hash_errors,
               "complete captured runner source set equals its frozen hashes",
               runtime_hash_errors)
    checks.add("execution.case_set", list(rows) == case_ids and list(inputs) == case_ids,
               case_ids, {"results": list(rows), "inputs": list(inputs)})

    row_errors = {case_id: {"status": row.get("status"),
                            "error_type": row.get("error_type"),
                            "error_stage": row.get("error_stage")}
                  for case_id, row in rows.items()
                  if row.get("status") != "completed" or row.get("error_type") or
                  row.get("error_stage")}
    report_errors = {case_id: {"assessment_valid": report.get("assessment_valid"),
                               "errors": report.get("errors")}
                     for case_id, report in reports.items()
                     if report.get("assessment_valid") is not True or report.get("errors")}
    checks.add("execution.completed_without_errors", not row_errors and not report_errors,
               "all eight cases completed with valid reports and no error fields",
               {"row_errors": row_errors, "report_errors": report_errors})
    actual_rounds: list[int] | dict[str, list[int]] = required_rounds
    search_round_counts: dict[str, int] | None = None
    loop_state: dict[str, Any] | None = None
    if v4:
        loop_ok, loop_state, case_rounds, search_round_counts = _outer_loop_audit(
            rows, reports, retrievals, case_ids, rounds,
            freeze.get("provider_mode"))
        actual_rounds = case_rounds
        loop_check_id = ("execution.adaptive_task_routed_outer_loop" if adaptive else
                         "execution.fixed_reanalysis_outer_loop")
        checks.add(loop_check_id, loop_ok,
                   ("every case has 1..2 real verification cycles and complete hit-or-exhausted task outcomes"
                    if adaptive else
                    "every case has exactly two forced fixed-reanalysis cycles"),
                   loop_state)
        if adaptive:
            demonstrated = bool(loop_state.get(
                "probe_owned_novel_second_pass_cases"))
            checks.add("execution.task_routed_loop_demonstrated", demonstrated,
                       "at least one full-run case has a later frozen-probe task hit followed by a real second verification",
                       {"loop_opportunity_cases": loop_state.get("loop_opportunity_cases"),
                        "second_pass_cases": loop_state.get("second_pass_cases"),
                        "probe_owned_novel_second_pass_cases": loop_state.get(
                            "probe_owned_novel_second_pass_cases")})
    checkpoint_errors = _checkpoint_audit(
        extension_run, rows, case_ids, actual_rounds)
    checks.add("execution.retained_rounds", not checkpoint_errors,
               "one canonical checkpoint per real verification cycle and no synthetic checkpoint",
               checkpoint_errors)
    call_errors = _call_audit(extension_run, rows, config, case_ids)
    checks.add("execution.calls_models_and_caps", not call_errors,
               "all calls returned, used the frozen model and stayed within preregistered caps",
               call_errors)
    status_errors = _status_audit(status, rows, len(case_ids))
    checks.add("execution.aggregate_status", not status_errors,
               "aggregate completion and usage reconcile exactly", status_errors)

    gold_by_id = {item["id"]: item["decision"] for item in gold["cases"]}
    final_labels = {case_id: rows.get(case_id, {}).get("prediction") for case_id in case_ids}
    checks.add("labels.artifact_final_labels", final_labels == gold_by_id,
               gold_by_id, final_labels)

    checks.add("comparison.independent_audit",
               comparison_error is None and comparison is not None,
               "independent comparison scorer completes without error", comparison_error)
    usage = None
    if comparison is not None:
        usage = _add_comparison_checks(
            checks, freeze, comparison, extension_run, actual_rounds,
            search_rounds=search_round_counts,
            probe_owned_novel_second_pass_cases=(set(loop_state.get(
                "probe_owned_novel_second_pass_cases", []))
                if adaptive and isinstance(loop_state, dict) else None))
    else:
        for identifier in (
                "comparison.frozen_denominator", "comparison.artifact_hashes",
                "labels.fixed_baselines_complete", "labels.final_provisional_accuracy",
                "labels.zero_baseline_breaks", "labels.zero_round_breaks",
                ("probe_audit.eight_valid_v4_plans" if v4 else
                 "probe_audit.eight_valid_v3_plans"), "probe_audit.structural_coverage",
                ("probe_audit.adaptive_cycles" if adaptive else
                 "probe_audit.sixteen_cycles"),
                "probe_audit.per_case_cycles_and_coverage",
                "probe_audit.result_and_grounding_coverage",
                "provenance.origins", "provenance.direct_edges",
                "resources.reported_not_thresholded"):
            checks.add(identifier, False, "independent comparison audit data", None)
        if v4:
            checks.add("probe_audit.v4_end_to_end_ledgers", False,
                       "independent v4 scorer audit data", None)
    return checks.finish(comparison_error, usage)


def _fatal_result(message: str) -> dict[str, Any]:
    checks = _Checks()
    checks.add("gate.artifact_read", False,
               "all gate inputs are readable, valid JSON artifacts", message)
    return checks.finish(message)


def check_full_gate(freeze_path: Path | str, gold_path: Path | str,
                    original_run: Path | str, staged_run: Path | str,
                    extension_run: Path | str) -> dict[str, Any]:
    freeze_path = Path(freeze_path)
    gold_path = Path(gold_path)
    original_run = Path(original_run)
    staged_run = Path(staged_run)
    extension_run = Path(extension_run)
    try:
        freeze = _read(freeze_path)
    except Exception as exc:
        return _fatal_result(f"{type(exc).__name__}: {exc}")
    comparison = None
    comparison_error = None
    smoke_result = None
    smoke_error = None
    try:
        repo_root = freeze_path.resolve().parents[1]
        next_run = freeze.get("next_run")
        if not _safe_relative_report_path(next_run):
            raise ValueError("freeze.next_run is not a safe report path")
        smoke_result = check_smoke_gate(
            freeze_path, gold_path, original_run, staged_run,
            repo_root / next_run, hash_mode="current")
    except Exception as exc:
        smoke_error = f"{type(exc).__name__}: {exc}"
    try:
        comparison = comparison_score(gold_path, original_run, staged_run, extension_run)
    except Exception as exc:
        comparison_error = f"{type(exc).__name__}: {exc}"
    try:
        return _evaluate_gate(freeze, comparison, comparison_error, freeze_path, gold_path,
                              original_run, staged_run, extension_run,
                              smoke_result, smoke_error)
    except Exception as exc:
        return _fatal_result(f"{type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--original-run", required=True)
    parser.add_argument("--staged-run", required=True)
    parser.add_argument("--extension-run", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = check_full_gate(args.freeze, args.gold, args.original_run,
                             args.staged_run, args.extension_run)
    if args.output:
        output = Path(args.output)
        if output.exists():
            parser.error("refusing to overwrite an existing gate artifact")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("passed", "checks_run", "checks_passed", "failed_checks")},
                     ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
