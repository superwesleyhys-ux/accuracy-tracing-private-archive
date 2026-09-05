"""Deterministic, artifact-only gate for a frozen target-extension smoke run.

The checker never calls a model or a network service.  It first invokes the
independent comparison scorer, which validates the generic run, plan,
projection, binding, policy, and grounded-result contracts.  It then enforces
the generic and case-specific conjuncts recorded in the selected freeze file.

Exit status is zero only when every mechanically executable conjunct passes.
Historical process claims that cannot be recovered from one filesystem
snapshot are disclosed separately and never silently counted as passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from target_extension_compare_score import score as comparison_score


IDENTITY_RELATIONS = {"exact", "alias", "description", "anaphora"}
IDENTITY_KINDS = {"entity_identity", "exact_designation"}
PLAN_SCHEMA_V3 = "decision-probe-v3"
PLAN_SCHEMA_V4 = "decision-probe-v4"
STRICT_PLAN_SCHEMAS = {PLAN_SCHEMA_V3, PLAN_SCHEMA_V4}
V4_EXPERIMENT_BY_PROVIDER = {
    "fixed_reanalysis": "target_extended_psi_development_v4_fixed_reanalysis",
    "task_routed": "target_extended_psi_development_v4_task_routed",
}
V4_PROVIDER_LABELS = {
    "fixed_reanalysis": "fixed eligible snapshots, no open-web collection",
    "task_routed": "task-routed fixed eligible snapshots, no open-web collection",
}
SECRET_SHAPE_DETECTORS = {
    "openai_api_key": re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "github_token": re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
    "aws_access_key": re.compile(rb"AKIA[0-9A-Z]{16}"),
}

UNCHECKABLE_CLAUSES = [
    {
        "clause": "No failed run directory was overwritten before this check.",
        "reason": "A final filesystem snapshot has no append-only creation history. "
                  "The checker can confirm that the preserved directories still exist, "
                  "but cannot prove that their earlier bytes were never replaced.",
    },
    {
        "clause": "A failed gate stops every later full run.",
        "reason": "The checker returns a nonzero process status on failure, but a future "
                  "caller must consume that status; one completed artifact cannot prove "
                  "future orchestration behavior.",
    },
]


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _secret_shape_audit(paths: list[Path], display_root: Path) -> list[dict[str, str]]:
    """Return detector names and file paths without ever retaining secret bytes."""
    findings: list[dict[str, str]] = []
    root = display_root.resolve()
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        try:
            label = resolved.relative_to(root).as_posix()
        except ValueError:
            findings.append({"file": str(resolved), "detector": "outside_audit_root"})
            continue
        try:
            payload = resolved.read_bytes()
        except OSError:
            findings.append({"file": label, "detector": "unreadable"})
            continue
        for detector, pattern in SECRET_SHAPE_DETECTORS.items():
            if pattern.search(payload):
                findings.append({"file": label, "detector": detector})
    return findings


def _gate_secret_paths(freeze: dict[str, Any], repo_root: Path,
                       freeze_path: Path, gold_path: Path,
                       original_run: Path, staged_run: Path,
                       extension_run: Path) -> list[Path]:
    paths = [freeze_path, gold_path]
    paths.extend(repo_root / relative for relative in freeze.get("files", {})
                 if isinstance(relative, str))
    for arm, directory in (("original", original_run), ("staged", staged_run)):
        manifest = freeze.get("frozen_baselines", {}).get(arm, {}).get("files", {})
        paths.extend(directory / relative for relative in manifest
                     if isinstance(relative, str))
    paths.extend(repo_root / relative for relative in freeze.get("prior_artifacts", {})
                 if isinstance(relative, str))
    if extension_run.is_dir():
        paths.extend(path for path in extension_run.rglob("*") if path.is_file())
    return paths


def _case_inputs(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases = data.get("cases")
    if not isinstance(cases, list):
        raise ValueError("inputs.cases must be an array")
    result: dict[str, dict[str, Any]] = {}
    for case in cases:
        identifier = case.get("target", {}).get("id") if isinstance(case, dict) else None
        if not isinstance(identifier, str) or not identifier or identifier in result:
            raise ValueError("inputs contain a missing or duplicate target id")
        result[identifier] = case
    return result


def _rows(results: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(results, list):
        raise ValueError("results must be an array")
    by_id: dict[str, dict[str, Any]] = {}
    for row in results:
        identifier = row.get("id") if isinstance(row, dict) else None
        if not isinstance(identifier, str) or not identifier or identifier in by_id:
            raise ValueError("results contain a missing or duplicate id")
        by_id[identifier] = row
    return by_id


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

    def finish(self, comparison_error: str | None = None) -> dict[str, Any]:
        failed = [item["id"] for item in self.items if not item["passed"]]
        return {
            "schema_version": "target-extension-smoke-gate-v1",
            "passed": not failed,
            "checks_run": len(self.items),
            "checks_passed": len(self.items) - len(failed),
            "failed_checks": failed,
            "checks": self.items,
            "comparison_audit_error": comparison_error,
            "uncheckable_clauses": UNCHECKABLE_CLAUSES,
            "interpretation": (
                "Passing is a development smoke-gate result, not proof of an "
                "accuracy increase or real-world correctness."
            ),
        }


def _exact_config(freeze: dict[str, Any], config: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    actual = {
        "experiment": config.get("experiment"),
        "target_plan_schema": config.get("target_plan_schema"),
        "model": config.get("model"),
        "reasoning_effort": config.get("reasoning_effort"),
        "case_ids": config.get("case_ids"),
        "scheduled_cases": config.get("scheduled_cases"),
        "rounds": config.get("trace_config", {}).get("max_rounds"),
        "max_documents": config.get("trace_config", {}).get("max_documents"),
        "max_decomposition_calls": config.get("trace_config", {}).get(
            "max_decomposition_calls"),
        "forced_rounds": config.get("trace_config", {}).get("experimental_force_rounds"),
        "automatic_transport_retries": config.get("automatic_transport_retries"),
        "budget_per_case": config.get("budget_per_case"),
        "workers": config.get("workers"),
        "target_structure_repairs": config.get("target_structure_repairs"),
        "target_extension_repairs": config.get("target_extension_repairs"),
        "extension_output_repairs": config.get(
            "target_extension_output_repairs", config.get("target_extension_repairs")),
        "material_structure_repairs": config.get("material_stage_repairs"),
        "material_semantic_repairs": config.get("max_inner_repairs"),
        "judgement_repairs": config.get("judgement_repairs"),
        "probe_result_structure_repairs": config.get("probe_result_structure_repairs"),
        "input_sha256": config.get("input_sha256"),
        "ordering_seed": config.get("ordering_seed"),
        "gold_read_during_inference": config.get("gold_read_during_inference"),
        "new_response_only": config.get("new_response_only"),
        "provider": config.get("provider"),
        "provider_mode": config.get("provider_mode"),
        "strict_retrieval_attribution": config.get("strict_retrieval_attribution"),
        "retrieval_attribution_mode": config.get("retrieval_attribution_mode"),
        "dataset_status": config.get("dataset_status"),
    }
    expected = {
        "experiment": freeze.get("experiment"),
        "target_plan_schema": freeze.get("target_plan_schema"),
        "model": freeze.get("model"),
        "reasoning_effort": freeze.get("reasoning_effort"),
        "case_ids": freeze.get("smoke_cases"),
        "scheduled_cases": len(freeze.get("smoke_cases", [])),
        "rounds": freeze.get("outer_rounds"),
        "max_documents": freeze.get("max_documents", 24),
        "max_decomposition_calls": freeze.get("max_decomposition_calls", 24),
        "forced_rounds": not (freeze.get("target_plan_schema") == PLAN_SCHEMA_V4 and
                               freeze.get("provider_mode") == "task_routed"),
        "automatic_transport_retries": freeze.get("automatic_transport_retries", 0),
        "budget_per_case": freeze.get("budget_per_case"),
        "workers": freeze.get("workers"),
        "target_structure_repairs": freeze.get("max_target_structure_repairs"),
        "target_extension_repairs": freeze.get("max_target_extension_repairs"),
        "extension_output_repairs": freeze.get("max_extension_output_repairs"),
        "material_structure_repairs": freeze.get("max_material_structure_repairs_per_return"),
        "material_semantic_repairs": freeze.get("max_material_semantic_repairs"),
        "judgement_repairs": freeze.get("max_judgement_repairs"),
        "probe_result_structure_repairs": freeze.get(
            "max_probe_result_structure_repairs",
            freeze.get("max_judgement_repairs"),
        ),
        "input_sha256": freeze.get("inputs_sha256"),
        "ordering_seed": freeze.get("ordering_seed", 20260906),
        "gold_read_during_inference": False,
        "new_response_only": True,
        "provider": freeze.get(
            "provider", "fixed eligible snapshots, no open-web collection"),
        "provider_mode": freeze.get("provider_mode"),
        "strict_retrieval_attribution": freeze.get("strict_retrieval_attribution"),
        "retrieval_attribution_mode": (
            "strict" if freeze.get("target_plan_schema") == PLAN_SCHEMA_V4 and
            freeze.get("provider_mode") == "task_routed" else
            "legacy" if freeze.get("target_plan_schema") == PLAN_SCHEMA_V4 and
            freeze.get("provider_mode") == "fixed_reanalysis" else
            freeze.get("retrieval_attribution_mode")),
        "dataset_status": freeze.get(
            "dataset_status",
            "previously_seen_development_cases_not_hidden_benchmark"),
    }
    return actual == expected, actual


def _execution_control_declarations(freeze: dict[str, Any], config: dict[str, Any],
                                    extension_run: Path) -> tuple[bool, dict[str, Any],
                                                                  dict[str, Any]]:
    actual = {
        "automatic_transport_retries": config.get("automatic_transport_retries"),
        "target_structure_repairs": config.get("target_structure_repairs"),
        "target_extension_repairs": config.get("target_extension_repairs"),
        "extension_output_repairs": config.get(
            "target_extension_output_repairs", config.get("target_extension_repairs")),
        "material_structure_repairs": config.get("material_stage_repairs"),
        "material_semantic_repairs": config.get("max_inner_repairs"),
        "judgement_repairs": config.get("judgement_repairs"),
        "probe_result_structure_repairs": config.get("probe_result_structure_repairs"),
    }
    expected = {
        "automatic_transport_retries": freeze.get("automatic_transport_retries", 0),
        "target_structure_repairs": freeze.get("max_target_structure_repairs"),
        "target_extension_repairs": freeze.get("max_target_extension_repairs"),
        "extension_output_repairs": freeze.get("max_extension_output_repairs"),
        "material_structure_repairs": freeze.get("max_material_structure_repairs_per_return"),
        "material_semantic_repairs": freeze.get("max_material_semantic_repairs"),
        "judgement_repairs": freeze.get("max_judgement_repairs"),
        "probe_result_structure_repairs": freeze.get(
            "max_probe_result_structure_repairs",
            freeze.get("max_judgement_repairs"),
        ),
    }
    model_io = extension_run / "executed-code" / "experiments" / "model_io.py"
    if model_io.exists():
        matches = re.findall(r"OpenAI\s*\(\s*max_retries\s*=\s*(\d+)",
                             model_io.read_text())
        actual["executed_sdk_max_retries"] = int(matches[0]) if len(matches) == 1 else None
        expected["executed_sdk_max_retries"] = freeze.get("automatic_transport_retries", 0)
    return actual == expected, expected, actual


def _checkpoint_audit(run: Path, rows: dict[str, dict[str, Any]],
                      case_ids: list[str],
                      required_rounds: list[int] | dict[str, list[int]]) -> list[str]:
    errors: list[str] = []
    for case_id in case_ids:
        expected_rounds = (required_rounds.get(case_id, [])
                           if isinstance(required_rounds, dict) else required_rounds)
        row = rows.get(case_id, {})
        checkpoints = row.get("checkpoints")
        rounds = [item.get("round") for item in checkpoints] if isinstance(checkpoints, list) else None
        if rounds != expected_rounds:
            errors.append(f"{case_id}: checkpoint rounds are {rounds!r}")
        retained = row.get("checkpoint_hashes")
        hashes = {item.get("round"): item.get("sha256") for item in retained
                  if isinstance(item, dict)} if isinstance(retained, list) else {}
        if sorted(hashes) != expected_rounds:
            errors.append(f"{case_id}: retained checkpoint hashes cover {sorted(hashes)!r}")
            continue
        for round_number in expected_rounds:
            path = run / f"{case_id}-checkpoint-{round_number}.json"
            if not path.exists():
                errors.append(f"{case_id}: missing {path.name}")
            elif _canonical_sha256(_read(path)) != hashes[round_number]:
                errors.append(f"{case_id}: {path.name} hash mismatch")
        unexpected = sorted(path.name for path in run.glob(f"{case_id}-checkpoint-*.json")
                            if path.name not in {
                                f"{case_id}-checkpoint-{round_number}.json"
                                for round_number in expected_rounds})
        if unexpected:
            errors.append(f"{case_id}: unexpected checkpoint artifacts {unexpected!r}")
    return errors


def _call_audit(run: Path, rows: dict[str, dict[str, Any]], config: dict[str, Any],
                case_ids: list[str]) -> list[str]:
    errors: list[str] = []
    budget = config.get("budget_per_case", {})
    expected_model = config.get("model")
    for case_id in case_ids:
        row = rows.get(case_id, {})
        path = run / f"{case_id}-calls.json"
        calls = _read(path) if path.exists() else []
        usage = row.get("usage", {})
        attempts = row.get("new_api_attempts")
        responses = row.get("returned_responses")
        if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts <= 0:
            errors.append(f"{case_id}: attempts are not a positive integer")
        if attempts != responses or attempts != len(calls) or attempts != usage.get("model_calls"):
            errors.append(f"{case_id}: attempts/responses/calls/usage disagree")
        token_totals = {"input_tokens": 0, "output_tokens": 0}
        for sequence, call in enumerate(calls, 1):
            call_usage = call.get("usage", {}) if isinstance(call, dict) else {}
            cap = call.get("max_completion_tokens") if isinstance(call, dict) else None
            if call.get("sequence") != sequence:
                errors.append(f"{case_id}: call {sequence} has a bad sequence")
            if (not isinstance(cap, int) or isinstance(cap, bool) or cap <= 0 or
                    cap > budget.get("per_call_output_tokens", -1)):
                errors.append(f"{case_id}: call {sequence} has a bad output cap")
            if (not call.get("response_id") or call.get("actual_model") != expected_model or
                    call.get("error_type") or call.get("error_code") or
                    call.get("has_refusal") or call.get("finish_reason") != "stop"):
                errors.append(f"{case_id}: call {sequence} failed or used a mismatched model")
            for field in token_totals:
                value = call_usage.get(field)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    errors.append(f"{case_id}: call {sequence} has invalid {field}")
                else:
                    token_totals[field] += value
            if isinstance(cap, int) and call_usage.get("output_tokens", cap + 1) > cap:
                errors.append(f"{case_id}: call {sequence} exceeded its output cap")
        for field, total in token_totals.items():
            if usage.get(field) != total:
                errors.append(f"{case_id}: retained calls disagree on {field}")
        numeric_limits = {
            "model_calls": budget.get("calls"),
            "output_tokens": budget.get("output_tokens"),
            "seconds": budget.get("seconds"),
        }
        for field, limit in numeric_limits.items():
            value = usage.get(field)
            if (not isinstance(value, (int, float)) or isinstance(value, bool) or
                    not math.isfinite(value) or value < 0 or
                    not isinstance(limit, (int, float)) or value > limit):
                errors.append(f"{case_id}: {field} exceeds or violates its budget")
    return errors


def _status_audit(status: dict[str, Any], rows: dict[str, dict[str, Any]],
                  scheduled: int) -> list[str]:
    errors: list[str] = []
    values = list(rows.values())
    expected_usage = {
        field: sum(row.get("usage", {}).get(field, 0) for row in values)
        for field in ("model_calls", "input_tokens", "output_tokens")
    }
    if status.get("status") != "completed":
        errors.append("aggregate status is not completed")
    if status.get("scheduled_cases") != scheduled or status.get("completed") != scheduled:
        errors.append(f"aggregate scheduled/completed counts are not {scheduled}/{scheduled}")
    attempts = sum(row.get("new_api_attempts", 0) for row in values)
    responses = sum(row.get("returned_responses", 0) for row in values)
    if status.get("new_api_attempts") != attempts or status.get("returned_responses") != responses:
        errors.append("aggregate attempt/response counts disagree with case rows")
    if attempts != responses:
        errors.append("aggregate returned responses do not equal attempts")
    if status.get("recorded_usage") != expected_usage:
        errors.append("aggregate recorded usage disagrees with case rows")
    return errors


def _plan_dimensions(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {dimension["id"]: dimension for claim in plan.get("claims", [])
            for dimension in claim.get("dimensions", [])}


def _canonical_identity_question(anchor: dict[str, Any], match_policy: str) -> str:
    """Mirror the program-owned v3 identity wording without importing runtime code."""
    quoted = json.dumps(anchor.get("quote"), ensure_ascii=False)
    if match_policy == "same_referent":
        return ("Does the evidence identify the same referent as " + quoted +
                " by exact mention, alias, unambiguous description or anaphora?")
    if match_policy == "exact_designation":
        return ("Does the evidence establish " + quoted +
                " as the exact asserted name, title, label or designation?")
    raise ValueError("identity question requires a referent match policy")


def _one_sentence_has_p07_naming_evidence(text: str) -> bool:
    """Require date, naming cue, and exact designation in one sentence/line."""
    segments = [" ".join(segment.split()) for line in text.splitlines()
                for segment in re.split(r"(?<=[.!?])\s+", line) if segment.strip()]
    naming = re.compile(r"\b(?:rename|renamed|renames|renaming|named|called|"
                        r"titled|designated)\b", re.IGNORECASE)
    designation = re.compile(r"(?<![A-Za-z0-9])GOES-19(?![A-Za-z0-9])",
                             re.IGNORECASE)
    date = re.compile(r"\bJuly\s+7,\s+2024\b", re.IGNORECASE)
    return any(naming.search(segment) and designation.search(segment) and date.search(segment)
               for segment in segments)


def _p02_checks(checks: _Checks, plan: dict[str, Any], report: dict[str, Any],
                target: dict[str, Any], required_rounds: list[int]) -> None:
    claims = plan.get("claims", [])
    parents = [claim for claim in claims if claim.get("role") == "attribution"]
    children = [claim for claim in claims if claim.get("role") == "attributed_content"]
    linked = (len(parents) == 1 and bool(children) and
              all(claim.get("parent_claim_id") == parents[0].get("id") for claim in children))
    checks.add("p02.attribution_structure", linked,
               "one attribution parent plus linked attributed-content child(ren)",
               {"parents": len(parents), "children": len(children), "linked": linked})

    text = target.get("text", "")
    cue = "reported that"
    cue_start = text.lower().find(cue)
    boundary = cue_start + len(cue) if cue_start >= 0 else -1
    parent_ends = [dimension.get("anchor", {}).get("end") for parent in parents
                   for dimension in parent.get("dimensions", [])]
    boundary_ok = (boundary >= 0 and all(isinstance(end, int) and end <= boundary
                                         for end in parent_ends))
    checks.add("p02.attribution_boundary", boundary_ok,
               f"all parent dimensions end at or before target offset {boundary}", parent_ends)

    child_dimensions = [dimension for child in children for dimension in child.get("dimensions", [])]
    required = {
        "final total": ("baseline_scope", "final total"),
        "Bennu": ("entity_identity", "Bennu"),
        "OSIRIS-REx": ("entity_identity", "OSIRIS-REx"),
        "70.3 grams": ("quantity_unit", "70.3 grams"),
    }
    found = {name: any(dimension.get("kind") == kind and phrase.lower() in
                       dimension.get("anchor", {}).get("quote", "").lower()
                       for dimension in child_dimensions)
             for name, (kind, phrase) in required.items()}
    checks.add("p02.content_dimensions", all(found.values()),
               "final-total, Bennu, OSIRIS-REx, and 70.3 grams are child dimensions", found)

    dimensions = _plan_dimensions(plan)
    final_dimension_ids = {identifier for identifier, dimension in dimensions.items()
                           if dimension.get("kind") == "baseline_scope" and
                           "final total" in dimension.get("anchor", {}).get("quote", "").lower()}
    probe_ids = {probe.get("id") for probe in plan.get("probes", [])
                 if probe.get("kind") == "baseline_scope" and
                 set(probe.get("dimension_ids", [])) & final_dimension_ids}
    outcomes: list[dict[str, Any]] = []
    for record in report.get("verification_history", []):
        for stage in ("evidence", "world"):
            matches = [item for item in record.get(stage + "_probe_results", [])
                       if item.get("probe_id") in probe_ids]
            outcomes.append({
                "round": record.get("round"),
                "stage": stage,
                "matches": len(matches),
                "contradicted": (len(matches) == 1 and matches[0].get("status") == "contradicted"),
                "grounded_in_m03": (len(matches) == 1 and any(
                    span.get("version_id") == "m03" for span in matches[0].get("basis", [])
                    if isinstance(span, dict))),
            })
    expected_slots = {(round_number, stage) for round_number in required_rounds
                      for stage in ("evidence", "world")}
    actual_slots = {(item["round"], item["stage"]) for item in outcomes}
    passed = (bool(probe_ids) and actual_slots == expected_slots and
              all(item["contradicted"] and item["grounded_in_m03"] for item in outcomes))
    checks.add("p02.final_total_each_round", passed,
               "one m03-grounded contradicted result in evidence and world for rounds 1 and 2",
               {"probe_ids": sorted(probe_ids), "outcomes": outcomes})


def _p08_identity_probes(plan: dict[str, Any]) -> tuple[set[str], list[dict[str, Any]]]:
    product = "usgs unified geologic map of the moon"
    dimensions = _plan_dimensions(plan)
    product_dimension_ids = {identifier for identifier, dimension in dimensions.items()
                             if dimension.get("kind") == "entity_identity" and
                             product in dimension.get("anchor", {}).get("quote", "").lower()}
    probes = [probe for probe in plan.get("probes", [])
              if probe.get("kind") == "entity_identity" and
              set(probe.get("dimension_ids", [])) & product_dimension_ids]
    return {probe.get("id") for probe in probes}, probes


def _p08_checks(checks: _Checks, plan: dict[str, Any], report: dict[str, Any],
                case: dict[str, Any], required_rounds: list[int]) -> None:
    identity_spans: list[tuple[str, str, int, int]] = []
    has_designation = False
    overlap: list[tuple[str, str]] = []
    for claim in plan.get("claims", []):
        local: list[tuple[str, int, int]] = []
        for dimension in claim.get("dimensions", []):
            if dimension.get("kind") not in IDENTITY_KINDS:
                continue
            has_designation |= dimension.get("kind") == "exact_designation"
            anchor = dimension.get("anchor", {})
            local.append((dimension.get("id"), anchor.get("start"), anchor.get("end")))
            identity_spans.append((claim.get("id"), dimension.get("id"),
                                   anchor.get("start"), anchor.get("end")))
        for index, left in enumerate(local):
            for right in local[index + 1:]:
                if (all(isinstance(value, int) for value in (left[1], left[2], right[1], right[2])) and
                        max(left[1], right[1]) < min(left[2], right[2])):
                    overlap.append((left[0], right[0]))
    designation_probes = [probe.get("id") for probe in plan.get("probes", [])
                          if probe.get("kind") == "exact_designation"]
    identity_ok = not has_designation and not designation_probes and not overlap
    checks.add("p08.identity_structure", identity_ok,
               "no exact_designation dimension/probe and no within-claim identity overlap",
               {"identity_spans": identity_spans, "designation_probes": designation_probes,
                "overlaps": overlap})

    probe_ids, probes = _p08_identity_probes(plan)
    policies_ok = bool(probes) and all(probe.get("match_policy") == "same_referent"
                                      for probe in probes)
    outcomes: list[dict[str, Any]] = []
    for record in report.get("verification_history", []):
        results = record.get("evidence_probe_results", [])
        for probe_id in sorted(probe_ids):
            matches = [item for item in results if item.get("probe_id") == probe_id]
            outcomes.append({
                "round": record.get("round"),
                "probe_id": probe_id,
                "matches": len(matches),
                "supported": len(matches) == 1 and matches[0].get("status") == "supported",
                "relation": matches[0].get("referent_relation") if len(matches) == 1 else None,
                "m15_basis": len(matches) == 1 and any(
                    span.get("version_id") == "m15" for span in matches[0].get("basis", [])
                    if isinstance(span, dict)),
            })
    expected_slots = {(round_number, probe_id) for round_number in required_rounds
                      for probe_id in probe_ids}
    actual_slots = {(item["round"], item["probe_id"]) for item in outcomes}
    results_ok = (policies_ok and actual_slots == expected_slots and
                  all(item["supported"] and item["relation"] in IDENTITY_RELATIONS and
                      item["m15_basis"] for item in outcomes))
    checks.add("p08.map_identity_results", results_ok,
               "every map-product identity probe uses same_referent and is supported from m15 "
               "in evidence rounds 1 and 2 by an allowed referent relation",
               {"probe_ids": sorted(probe_ids), "policies_ok": policies_ok,
                "outcomes": outcomes})

    gap_records: list[dict[str, Any]] = []
    for source, gaps in (("active", report.get("gaps", [])),
                         ("registry", report.get("gap_registry", []))):
        for gap in gaps if isinstance(gaps, list) else []:
            if (isinstance(gap, dict) and (gap.get("probe_id") in probe_ids or
                                           gap.get("dimension") in IDENTITY_KINDS)):
                gap_records.append({"source": source, "id": gap.get("id"),
                                    "probe_id": gap.get("probe_id")})
    for record in report.get("verification_history", []):
        for gap in record.get("gaps", []) if isinstance(record, dict) else []:
            if (isinstance(gap, dict) and (gap.get("probe_id") in probe_ids or
                                           gap.get("dimension") in IDENTITY_KINDS)):
                gap_records.append({"source": f"round-{record.get('round')}",
                                    "id": gap.get("id"), "probe_id": gap.get("probe_id")})
    checks.add("p08.no_identity_gap", not gap_records,
               "no map-product/identity gap is emitted or retained", gap_records)

    materials = {item.get("version_id"): item for item in case.get("materials", [])}
    visible_urls = {materials[identifier].get("url") for identifier in case.get("seed_ids", [])
                    if identifier in materials}
    repeated: list[dict[str, Any]] = []
    for operation in report.get("operations", []):
        if operation.get("action") == "search":
            for task in operation.get("tasks", []):
                if (task.get("action") == "fetch" and task.get("locator") in visible_urls and
                        task.get("locator") in {materials.get(identifier, {}).get("url")
                                                for identifier in ("m14", "m15")}):
                    repeated.append({"sequence": operation.get("sequence"),
                                     "round": operation.get("round"),
                                     "locator": task.get("locator")})
        if operation.get("action") in {"snapshot_saved", "duplicate_observed"}:
            material = materials.get(operation.get("version_id"), {})
            if material.get("url"):
                visible_urls.add(material["url"])
    checks.add("p08.no_repeated_visible_fetch", not repeated,
               "no fetch task repeats an already-visible m14/m15 URL", repeated)

    edges = [item for item in report.get("relations", []) if
             item.get("from_version") == "m14" and item.get("to_version") == "m15" and
             item.get("kind") == "cites" and item.get("status") == "direct"]
    origins = [item for item in report.get("origins", []) if item.get("version_id") == "m15"]
    checks.add("p08.provenance_graph", bool(edges) and bool(origins),
               "m14 -> m15 direct cites edge and m15 origin", {
                   "matching_edges": len(edges), "matching_origins": len(origins)})

    if plan.get("schema_version") not in STRICT_PLAN_SCHEMAS:
        return

    origin_values = [item.get("version_id") if isinstance(item, dict) else None
                     for item in report.get("origins", [])]
    origin_ids = {value for value in origin_values if isinstance(value, str)}
    checks.add("p08.v3_terminal_origin",
               all(isinstance(value, str) for value in origin_values) and
               origin_ids == {"m15"},
               "the only exported terminal origin is m15", origin_values)

    analyses = report.get("analyses")
    raw_origin_records: list[Any] = []
    analyses_valid = isinstance(analyses, dict)
    if analyses_valid:
        for analysis in analyses.values():
            origins_for_version = analysis.get("origins") if isinstance(analysis, dict) else None
            if not isinstance(origins_for_version, list):
                analyses_valid = False
                continue
            raw_origin_records.extend(origins_for_version)
    raw_candidate_values = [item.get("version_id") if isinstance(item, dict) and
                            item.get("target_id") == "p08" else None
                            for item in raw_origin_records]
    raw_candidate_ids = {value for value in raw_candidate_values
                         if isinstance(value, str)}
    raw_candidates_ok = (analyses_valid and
                         all(isinstance(value, str) for value in raw_candidate_values) and
                         raw_candidate_ids == {"m14", "m15"})
    checks.add("p08.v3_raw_origin_candidates", raw_candidates_ok,
               "current per-version analyses retain raw candidates m14 and m15",
               raw_candidate_values)

    combination_claims = []
    for claim in plan.get("claims", []):
        claim_text = " ".join((claim.get("statement", ""),
                               claim.get("anchor", {}).get("quote", ""))).casefold()
        if "combin" in claim_text and "apollo-era" in claim_text and "newer" in claim_text:
            combination_claims.append(claim)
    combination = combination_claims[0] if len(combination_claims) == 1 else {}
    time_dimensions = [item for item in combination.get("dimensions", [])
                       if item.get("kind") == "time"]
    apollo = [item for item in time_dimensions
              if "apollo-era" in item.get("anchor", {}).get("quote", "").casefold()]
    newer = [item for item in time_dimensions
             if "newer" in item.get("anchor", {}).get("quote", "").casefold()]
    selected = apollo + newer if len(apollo) == len(newer) == 1 else []
    spans = [(item.get("anchor", {}).get("start"), item.get("anchor", {}).get("end"))
             for item in selected]
    nonoverlap = (len(spans) == 2 and all(isinstance(value, int)
                  for span in spans for value in span) and
                  max(spans[0][0], spans[1][0]) >= min(spans[0][1], spans[1][1]))
    dimensions_ok = (len(combination_claims) == 1 and len(time_dimensions) == 2 and
                     len(selected) == 2 and nonoverlap)
    checks.add("p08.v3_independent_time_dimensions", dimensions_ok,
               "one Apollo-era and one newer-data time dimension with non-overlapping anchors",
               {"combination_claim_count": len(combination_claims),
                "time_dimensions": [{"id": item.get("id"),
                                     "quote": item.get("anchor", {}).get("quote"),
                                     "start": item.get("anchor", {}).get("start"),
                                     "end": item.get("anchor", {}).get("end")}
                                    for item in time_dimensions],
                "nonoverlap": nonoverlap})

    expected_dimension_ids = {item.get("id") for item in selected}
    time_probes = [probe for probe in plan.get("probes", [])
                   if probe.get("claim_id") == combination.get("id") and
                   probe.get("kind") == "time_boundary" and
                   set(probe.get("dimension_ids", [])) & expected_dimension_ids]
    probe_by_dimension = {
        dimension_id: [probe for probe in time_probes
                       if probe.get("dimension_ids") == [dimension_id]]
        for dimension_id in expected_dimension_ids
    }
    time_binding_ok = (dimensions_ok and len(expected_dimension_ids) == 2 and
                       len(time_probes) == 2 and
                       all(len(probes) == 1 for probes in probe_by_dimension.values()))
    checks.add("p08.v3_singleton_time_probes", time_binding_ok,
               "exactly one singleton time_boundary probe per independent time dimension",
               {identifier: [probe.get("id") for probe in probes]
                for identifier, probes in probe_by_dimension.items()})

    time_probe_ids = {probes[0].get("id") for probes in probe_by_dimension.values()
                      if len(probes) == 1}
    time_outcomes: list[dict[str, Any]] = []
    for record in report.get("verification_history", []):
        for stage in ("evidence", "world"):
            results = record.get(stage + "_probe_results", [])
            for probe_id in sorted(time_probe_ids):
                matches = [item for item in results if item.get("probe_id") == probe_id]
                time_outcomes.append({
                    "round": record.get("round"), "stage": stage,
                    "probe_id": probe_id, "matches": len(matches),
                    "supported": len(matches) == 1 and matches[0].get("status") == "supported",
                    "m15_basis": len(matches) == 1 and any(
                        span.get("version_id") == "m15"
                        for span in matches[0].get("basis", []) if isinstance(span, dict)),
                })
    expected_time_slots = {(round_number, stage, probe_id)
                           for round_number in required_rounds
                           for stage in ("evidence", "world")
                           for probe_id in time_probe_ids}
    actual_time_slots = {(item["round"], item["stage"], item["probe_id"])
                         for item in time_outcomes}
    time_results_ok = (time_binding_ok and len(time_probe_ids) == 2 and
                       actual_time_slots == expected_time_slots and
                       all(item["supported"] and item["m15_basis"]
                           for item in time_outcomes))
    checks.add("p08.v3_time_results_each_round", time_results_ok,
               "both time probes are supported from m15 in evidence/world for every round",
               time_outcomes)
    active_v3_gaps = [gap for gap in report.get("gaps", [])
                      if isinstance(gap, dict) and
                      (gap.get("id") == "lineage:p08" or
                       gap.get("probe_id") in time_probe_ids or
                       gap.get("dimension") in {"time", "time_boundary"})]
    checks.add("p08.v3_no_active_lineage_or_time_gap", not active_v3_gaps,
               "a complete terminal chain and supported time probes retain no active lineage/time gap",
               active_v3_gaps)


def _p07_checks(checks: _Checks, plan: dict[str, Any], report: dict[str, Any],
                required_rounds: list[int]) -> None:
    claims = plan.get("claims", [])
    single = plan.get("logic") == "single" and len(claims) == 1
    checks.add("p07.single_claim", single, "single logic with exactly one claim",
               {"logic": plan.get("logic"), "claims": len(claims)})
    claim = claims[0] if len(claims) == 1 else {}
    dimensions = claim.get("dimensions", [])
    old = [item for item in dimensions if item.get("kind") == "entity_identity" and
           item.get("anchor", {}).get("quote", "").strip().lower() == "goes-u"]
    new = [item for item in dimensions if item.get("kind") == "exact_designation" and
           item.get("anchor", {}).get("quote", "").strip().lower() == "goes-19"]
    names_ok = len(old) == 1 and len(new) == 1
    checks.add("p07.old_and_new_name_dimensions", names_ok,
               "one GOES-U entity_identity and one GOES-19 exact_designation",
               {"goes_u_entity_identity": len(old), "goes_19_exact_designation": len(new)})

    identifiers = [item.get("id") for item in dimensions]
    identity = [item for item in dimensions if item.get("kind") in IDENTITY_KINDS]
    overlaps = []
    for index, left in enumerate(identity):
        for right in identity[index + 1:]:
            a, b = left.get("anchor", {}), right.get("anchor", {})
            if (all(isinstance(value, int) for value in
                    (a.get("start"), a.get("end"), b.get("start"), b.get("end"))) and
                    max(a["start"], b["start"]) < min(a["end"], b["end"])):
                overlaps.append((left.get("id"), right.get("id")))
    probe_identifiers = [item.get("id") for item in plan.get("probes", [])]
    duplicate_free = (len(identifiers) == len(set(identifiers)) and
                      len(probe_identifiers) == len(set(probe_identifiers)) and not overlaps)
    checks.add("p07.no_duplicates_or_identity_overlap", duplicate_free,
               "unique dimension/probe ids and no identity/designation overlap",
               {"dimension_ids": identifiers, "probe_ids": probe_identifiers,
                "overlaps": overlaps})

    relations = [item for item in plan.get("probes", [])
                 if item.get("kind") == "designation_relation"]
    relation_ok = (len(relations) == 1 and
                   set(relations[0].get("dimension_ids", [])) == set(identifiers) and
                   len(relations[0].get("dimension_ids", [])) == len(identifiers) and
                   relations[0].get("routes") == ["atoms", "evidence", "world"] and
                   relations[0].get("gate") == "always" and
                   relations[0].get("match_policy") == "semantic_constraint")
    checks.add("p07.designation_relation", relation_ok,
               "one all-dimension designation_relation with frozen routes/gate/policy",
               relations)

    dimensions_by_id = {item.get("id"): item for item in dimensions}
    time_ids = {identifier for identifier, item in dimensions_by_id.items()
                if item.get("kind") == "time" and
                item.get("anchor", {}).get("quote", "").strip().lower() == "june 25, 2024"}
    time_probes = [item for item in plan.get("probes", [])
                   if item.get("kind") == "time_boundary" and
                   set(item.get("dimension_ids", [])) == time_ids]
    contradiction_ids = ({item.get("id") for item in time_probes} |
                         {item.get("id") for item in relations})
    contradiction_outcomes = []
    for record in report.get("verification_history", []):
        for stage in ("evidence", "world"):
            results = record.get(stage + "_probe_results", [])
            for probe_id in sorted(contradiction_ids):
                matches = [item for item in results if item.get("probe_id") == probe_id]
                basis = matches[0].get("basis", []) if len(matches) == 1 else []
                m13_naming = any(span.get("version_id") == "m13" and
                                 _one_sentence_has_p07_naming_evidence(
                                     span.get("quote", ""))
                                 for span in basis if isinstance(span, dict))
                contradiction_outcomes.append({
                    "round": record.get("round"), "stage": stage, "probe_id": probe_id,
                    "matches": len(matches),
                    "contradicted": len(matches) == 1 and
                                    matches[0].get("status") == "contradicted",
                    "m13_july_7_naming_basis": m13_naming,
                })
    expected_contradictions = {(round_number, stage, probe_id)
                               for round_number in required_rounds
                               for stage in ("evidence", "world")
                               for probe_id in contradiction_ids}
    actual_contradictions = {(item["round"], item["stage"], item["probe_id"])
                             for item in contradiction_outcomes}
    contradiction_ok = (len(time_probes) == 1 and len(relations) == 1 and
                         actual_contradictions == expected_contradictions and
                         all(item["contradicted"] and item["m13_july_7_naming_basis"]
                             for item in contradiction_outcomes))
    checks.add("p07.time_and_designation_contradictions", contradiction_ok,
               "time_boundary and designation_relation are contradicted from m13's July 7, "
               "2024 basis in evidence/world for every round",
               {"time_probe_ids": [item.get("id") for item in time_probes],
                "relation_probe_ids": [item.get("id") for item in relations],
                "outcomes": contradiction_outcomes})

    name_probe_specs = []
    if names_ok:
        name_probe_specs = [
            ("old", next((item for item in plan.get("probes", [])
                          if item.get("kind") == "entity_identity" and
                          item.get("dimension_ids") == [old[0].get("id")]), None),
             "same_referent", IDENTITY_RELATIONS),
            ("new", next((item for item in plan.get("probes", [])
                          if item.get("kind") == "exact_designation" and
                          item.get("dimension_ids") == [new[0].get("id")]), None),
             "exact_designation", {"exact"}),
        ]
    name_outcomes = []
    for record in report.get("verification_history", []):
        for stage in ("evidence", "world"):
            results = record.get(stage + "_probe_results", [])
            for label, probe, policy, allowed in name_probe_specs:
                probe_id = probe.get("id") if probe else None
                matches = [item for item in results if item.get("probe_id") == probe_id]
                expected_name = "GOES-U" if label == "old" else "GOES-19"
                lexical = re.compile(r"(?<![A-Za-z0-9])" + re.escape(expected_name) +
                                     r"(?![A-Za-z0-9])", re.IGNORECASE)
                name_outcomes.append({
                    "round": record.get("round"), "stage": stage, "name": label,
                    "probe_id": probe_id,
                    "policy": probe.get("match_policy") if probe else None,
                    "supported": len(matches) == 1 and matches[0].get("status") == "supported",
                    "relation": matches[0].get("referent_relation") if len(matches) == 1 else None,
                    "m13_basis": len(matches) == 1 and any(
                        span.get("version_id") == "m13" and
                        lexical.search(span.get("quote", ""))
                        for span in matches[0].get("basis", [])
                        if isinstance(span, dict)),
                    "expected_policy": policy, "allowed_relations": sorted(allowed),
                })
    expected_name_slots = len(required_rounds) * 2 * 2
    name_results_ok = (len(name_probe_specs) == 2 and len(name_outcomes) == expected_name_slots and
                       all(item["probe_id"] and item["policy"] == item["expected_policy"] and
                           item["supported"] and item["relation"] in item["allowed_relations"] and
                           item["m13_basis"] for item in name_outcomes))
    checks.add("p07.name_identity_results", name_results_ok,
               "GOES-U same-referent and GOES-19 exact-designation results are supported "
               "from m13 in both layers of every round", name_outcomes)

    case_specific_protected_probe_ids: set[str] = set()
    plan_schema = plan.get("schema_version")
    if plan_schema in STRICT_PLAN_SCHEMAS:
        schema_label = "v4" if plan_schema == PLAN_SCHEMA_V4 else "v3"
        noaa_dimensions = [item for item in dimensions
                           if item.get("kind") == "entity_identity" and
                           item.get("anchor", {}).get("quote", "").strip().casefold() == "noaa"]
        noaa_dimension = noaa_dimensions[0] if len(noaa_dimensions) == 1 else {}
        noaa_probes = [item for item in plan.get("probes", [])
                       if item.get("kind") == "entity_identity" and
                       item.get("dimension_ids") == [noaa_dimension.get("id")]]
        noaa_probe = noaa_probes[0] if len(noaa_probes) == 1 else {}
        if isinstance(noaa_probe.get("id"), str):
            case_specific_protected_probe_ids.add(noaa_probe["id"])
        canonical_question = (_canonical_identity_question(
            noaa_dimension.get("anchor", {}), "same_referent")
            if noaa_dimension else None)
        identity_contract_ok = (len(noaa_dimensions) == 1 and len(noaa_probes) == 1 and
                                noaa_probe.get("match_policy") == "same_referent" and
                                noaa_probe.get("question") == canonical_question)
        checks.add(f"p07.{schema_label}_noaa_identity_contract", identity_contract_ok,
                   "NOAA has the program-canonical same-referent question only",
                   {"dimensions": len(noaa_dimensions), "probes": len(noaa_probes),
                    "question": noaa_probe.get("question"),
                    "canonical_question": canonical_question})

        noaa_outcomes: list[dict[str, Any]] = []
        for record in report.get("verification_history", []):
            for stage in ("evidence", "world"):
                results = record.get(stage + "_probe_results", [])
                matches = [item for item in results
                           if item.get("probe_id") == noaa_probe.get("id")]
                exact_noaa_basis = len(matches) == 1 and any(
                    span.get("version_id") == "m13" and
                    re.search(r"(?<![A-Za-z0-9])NOAA(?![A-Za-z0-9])",
                              span.get("quote", ""), re.IGNORECASE)
                    for span in matches[0].get("basis", []) if isinstance(span, dict))
                noaa_outcomes.append({
                    "round": record.get("round"), "stage": stage,
                    "matches": len(matches),
                    "supported": len(matches) == 1 and matches[0].get("status") == "supported",
                    "relation": matches[0].get("referent_relation")
                    if len(matches) == 1 else None,
                    "m13_basis": len(matches) == 1 and any(
                        span.get("version_id") == "m13"
                        for span in matches[0].get("basis", []) if isinstance(span, dict)),
                    "exact_noaa_basis": exact_noaa_basis,
                })
        expected_slots = {(round_number, stage) for round_number in required_rounds
                          for stage in ("evidence", "world")}
        actual_slots = {(item["round"], item["stage"]) for item in noaa_outcomes}
        noaa_results_ok = (identity_contract_ok and actual_slots == expected_slots and
                           all(item["supported"] and item["m13_basis"] and
                               item["exact_noaa_basis"] and
                               item["relation"] == "exact" for item in noaa_outcomes))
        checks.add(f"p07.{schema_label}_noaa_identity_results", noaa_results_ok,
                   "NOAA identity is exact and supported from m13 in both layers of every round",
                   noaa_outcomes)

        if plan_schema == PLAN_SCHEMA_V4:
            actor_dimensions = [item for item in dimensions
                                if item.get("kind") == "actor_role" and
                                item.get("anchor", {}).get("quote", "").strip().casefold() ==
                                "noaa"]
            actor_dimension = actor_dimensions[0] if len(actor_dimensions) == 1 else {}
            actor_probes = [item for item in plan.get("probes", [])
                            if item.get("kind") == "actor_role" and
                            item.get("dimension_ids") == [actor_dimension.get("id")]]
            actor_probe = actor_probes[0] if len(actor_probes) == 1 else {}
            actor_anchor = actor_dimension.get("anchor", {})
            claim_anchor = claim.get("anchor", {})
            canonical_actor_question = None
            if (isinstance(actor_anchor.get("quote"), str) and
                    type(claim_anchor.get("start")) is int and
                    type(claim_anchor.get("end")) is int):
                canonical_actor_question = (
                    "Does the evidence establish " +
                    json.dumps(actor_anchor["quote"], ensure_ascii=False) +
                    " in the actor or agent role asserted by target[" +
                    str(claim_anchor["start"]) + ":" + str(claim_anchor["end"]) + "]?"
                )
            actor_contract_ok = (
                len(actor_dimensions) == 1 and len(actor_probes) == 1 and
                actor_probe.get("match_policy") == "semantic_constraint" and
                actor_probe.get("routes") == ["atoms", "evidence", "world"] and
                actor_probe.get("gate") == "always" and
                actor_probe.get("question") == canonical_actor_question and
                actor_probe.get("id") != noaa_probe.get("id") and
                actor_dimension.get("id") != noaa_dimension.get("id") and
                noaa_dimension.get("id") not in
                set(actor_probe.get("dimension_ids", [])))
            actor_contract_state = {
                "dimensions": len(actor_dimensions), "probes": len(actor_probes),
                "question": actor_probe.get("question"),
                "canonical_question": canonical_actor_question,
            }
            checks.add("p07.v4_actor_role_contract", actor_contract_ok,
                       "one program-canonical singleton NOAA actor_role probe is distinct from identity",
                       actor_contract_state)
            actor_check_id = "p07.v4_actor_role_results"
            actor_check_description = (
                "the distinct actor_role probe stays unresolved on passive m13 evidence")
        else:
            subject_ids = {item.get("id") for item in dimensions
                           if item.get("kind") == "subject"}
            predicate_ids = {item.get("id") for item in dimensions
                             if item.get("kind") == "predicate"}
            semantic_ids = subject_ids | predicate_ids
            actor_probes = [item for item in plan.get("probes", [])
                            if item.get("kind") == "semantic_core" and
                            set(item.get("dimension_ids", [])) == semantic_ids and
                            len(item.get("dimension_ids", [])) == len(semantic_ids)]
            actor_probe = actor_probes[0] if len(actor_probes) == 1 else {}
            actor_dimension = {}
            actor_contract_ok = len(actor_probes) == 1
            actor_check_id = "p07.v3_actor_role_separate"
            actor_check_description = (
                "the distinct semantic-core actor check stays unresolved on passive m13 evidence")
            actor_contract_state = {
                "semantic_probe_ids": [item.get("id") for item in actor_probes],
            }
        actor_outcomes: list[dict[str, Any]] = []
        for record in report.get("verification_history", []):
            for stage in ("evidence", "world"):
                results = record.get(stage + "_probe_results", [])
                matches = [item for item in results
                           if item.get("probe_id") == actor_probe.get("id")]
                passive_m13_basis = len(matches) == 1 and any(
                    span.get("version_id") == "m13" and
                    re.search(r"(?<![A-Za-z0-9])NOAA(?:['’]s)?(?![A-Za-z0-9])",
                              span.get("quote", ""), re.IGNORECASE) and
                    re.search(r"\b(?:was|is)\s+(?:re)?named\b|\bwas\s+designated\b",
                              span.get("quote", ""), re.IGNORECASE)
                    for span in matches[0].get("basis", []) if isinstance(span, dict))
                actor_outcomes.append({
                    "round": record.get("round"), "stage": stage,
                    "matches": len(matches),
                    "unresolved": len(matches) == 1 and
                                  matches[0].get("status") == "unresolved",
                    "m13_basis": len(matches) == 1 and any(
                        span.get("version_id") == "m13"
                        for span in matches[0].get("basis", []) if isinstance(span, dict)),
                    "passive_m13_basis": passive_m13_basis,
                })
        actor_slots = {(item["round"], item["stage"]) for item in actor_outcomes}
        actor_separate_ok = (actor_contract_ok and identity_contract_ok and
                             actor_probe.get("id") != noaa_probe.get("id") and
                             actor_dimension.get("id") != noaa_dimension.get("id") and
                             noaa_dimension.get("id") not in
                             set(actor_probe.get("dimension_ids", [])) and
                             actor_slots == expected_slots and
                             all(item["unresolved"] and item["m13_basis"] and
                                 item["passive_m13_basis"]
                                 for item in actor_outcomes))
        actor_contract_state.update({
            "identity_probe_id": noaa_probe.get("id"),
            "outcomes": actor_outcomes,
        })
        checks.add(actor_check_id, actor_separate_ok, actor_check_description,
                   actor_contract_state)

    protected_probe_ids = ({item.get("id") for item in relations} |
                           {item.get("id") for item in time_probes} |
                           {item[1].get("id") for item in name_probe_specs if item[1]} |
                           case_specific_protected_probe_ids)
    gap_records = []
    gap_sources = [("active", report.get("gaps", [])),
                   ("registry", report.get("gap_registry", []))]
    gap_sources.extend((f"round-{record.get('round')}", record.get("gaps", []))
                       for record in report.get("verification_history", []))
    for source, gaps in gap_sources:
        for gap in gaps if isinstance(gaps, list) else []:
            if (isinstance(gap, dict) and
                    (gap.get("probe_id") in protected_probe_ids or
                     gap.get("dimension") in IDENTITY_KINDS |
                     {"designation_relation", "time", "time_boundary"})):
                gap_records.append({"source": source, "id": gap.get("id"),
                                    "probe_id": gap.get("probe_id")})
    checks.add("p07.no_relevant_gaps", not gap_records,
               "no time-boundary, designation-relation, exact-designation, or identity gap",
               gap_records)


def _p04_checks(checks: _Checks, report: dict[str, Any],
                plan: dict[str, Any] | None = None,
                required_rounds: list[int] | None = None) -> None:
    """Guard p04's unrelated source and crossed magnitude/baseline variables."""
    origin_values = [item.get("version_id") if isinstance(item, dict) else None
                     for item in report.get("origins", [])]
    checks.add("p04.no_m08_origin",
               all(isinstance(value, str) for value in origin_values) and
               "m08" not in origin_values,
               "m08 is excluded from final origins", origin_values)
    direct_m08 = [item for item in report.get("relations", [])
                  if isinstance(item, dict) and item.get("status") == "direct" and
                  item.get("kind") in {"quotes", "cites", "reprints", "translates", "derives"} and
                  "m08" in {item.get("from_version"), item.get("to_version")}]
    checks.add("p04.no_m08_propagation_edge", not direct_m08,
               "no direct documentary-propagation edge joins the unrelated m08 branch",
               direct_m08)
    if plan is None or required_rounds is None:
        return

    content_claims = [item for item in plan.get("claims", [])
                      if item.get("role") == "attributed_content"]
    quantity_dimensions = [item for claim in content_claims
                           for item in claim.get("dimensions", [])
                           if item.get("kind") == "quantity_unit" and
                           "1.29" in item.get("anchor", {}).get("quote", "")]
    baseline_dimensions = [item for claim in content_claims
                           for item in claim.get("dimensions", [])
                           if item.get("kind") == "baseline_scope" and
                           "1850" in item.get("anchor", {}).get("quote", "") and
                           "1900" in item.get("anchor", {}).get("quote", "")]
    variable_structure_ok = (plan.get("logic") == "attribution" and
                             len(content_claims) >= 1 and
                             len(quantity_dimensions) == 1 and
                             len(baseline_dimensions) == 1 and
                             quantity_dimensions[0].get("id") !=
                             baseline_dimensions[0].get("id"))
    checks.add("p04.crossed_variables_separate", variable_structure_ok,
               "1.29°C and the 1850–1900 baseline are separate attributed-content dimensions",
               {"logic": plan.get("logic"),
                "content_claims": len(content_claims),
                "quantity_ids": [item.get("id") for item in quantity_dimensions],
                "baseline_ids": [item.get("id") for item in baseline_dimensions]})

    probes = plan.get("probes", [])
    specs = []
    if len(quantity_dimensions) == 1:
        specs.append(("quantity_unit", quantity_dimensions[0].get("id")))
    if len(baseline_dimensions) == 1:
        specs.append(("baseline_scope", baseline_dimensions[0].get("id")))
    selected_probes = {}
    for kind, dimension_id in specs:
        matches = [item for item in probes if item.get("kind") == kind and
                   item.get("dimension_ids") == [dimension_id]]
        selected_probes[kind] = matches
    bindings_ok = (variable_structure_ok and set(selected_probes) ==
                   {"quantity_unit", "baseline_scope"} and
                   all(len(items) == 1 for items in selected_probes.values()))
    checks.add("p04.crossed_variable_probes", bindings_ok,
               "one singleton quantity and baseline probe is bound to each separate variable",
               {kind: [item.get("id") for item in items]
                for kind, items in selected_probes.items()})

    expected_slots = {(round_number, stage, kind)
                      for round_number in required_rounds
                      for stage in ("evidence", "world")
                      for kind in ("quantity_unit", "baseline_scope")}
    outcomes: list[dict[str, Any]] = []
    probe_kinds = {items[0].get("id"): kind
                   for kind, items in selected_probes.items() if len(items) == 1}
    for record in report.get("verification_history", []):
        for stage in ("evidence", "world"):
            results = record.get(stage + "_probe_results", [])
            for probe_id, kind in probe_kinds.items():
                matches = [item for item in results if item.get("probe_id") == probe_id]
                basis = matches[0].get("basis", []) if len(matches) == 1 else []
                quotes = [span.get("quote", "") for span in basis
                          if isinstance(span, dict)]
                pairing_129 = any("1.29" in quote and "20th-century" in quote
                                  for quote in quotes)
                pairing_146 = any("1.46" in quote and "1850" in quote and
                                  "1900" in quote for quote in quotes)
                outcomes.append({
                    "round": record.get("round"), "stage": stage, "kind": kind,
                    "matches": len(matches),
                    "contradicted": len(matches) == 1 and
                                    matches[0].get("status") == "contradicted",
                    "only_m07": bool(basis) and all(
                        isinstance(span, dict) and span.get("version_id") == "m07"
                        for span in basis),
                    "pair_1_29_to_20th_century": pairing_129,
                    "pair_1_46_to_1850_1900": pairing_146,
                })
    actual_slots = {(item["round"], item["stage"], item["kind"])
                    for item in outcomes}
    result_ok = (bindings_ok and actual_slots == expected_slots and
                 all(item["contradicted"] and item["only_m07"] and
                     item["pair_1_29_to_20th_century"] and
                     item["pair_1_46_to_1850_1900"] for item in outcomes))
    checks.add("p04.crossed_variable_results", result_ok,
               "both crossed-variable probes are contradicted from the two m07 pairings in every layer and round",
               outcomes)


def _frozen_hash_audit(freeze: dict[str, Any], repo_root: Path, gold_path: Path,
                       extension_run: Path, hash_mode: str) -> tuple[list[dict[str, Any]],
                                                                    list[str]]:
    if hash_mode not in {"current", "executed", "skip"}:
        raise ValueError("hash_mode must be current, executed, or skip")
    mismatches: list[dict[str, Any]] = []
    skipped: list[str] = []
    config_path = extension_run / "config.json"
    config = _read(config_path) if config_path.exists() else {}
    recorded = config.get("source_sha256", {}) if isinstance(config, dict) else {}
    for relative, expected in freeze.get("files", {}).items():
        if hash_mode == "skip":
            skipped.append(relative)
            continue
        if hash_mode == "current":
            path = repo_root / relative
            actual = _sha256(path) if path.exists() else None
        else:
            executed = extension_run / "executed-code" / relative
            if executed.exists():
                actual = _sha256(executed)
            elif relative in recorded:
                actual = recorded.get(relative)
            else:
                skipped.append(relative)
                continue
        if actual != expected:
            mismatches.append({"file": relative, "expected": expected, "actual": actual})
    if hash_mode == "current":
        canonical_inputs = repo_root / "experiments" / "proof_pilot" / "inputs.json"
        actual_input_hash = _sha256(canonical_inputs) if canonical_inputs.exists() else None
        input_label = (canonical_inputs.relative_to(repo_root).as_posix()
                       if canonical_inputs.exists() else str(canonical_inputs))
    elif hash_mode == "executed":
        actual_input_hash = config.get("input_sha256")
        input_label = "config.input_sha256"
    else:
        actual_input_hash = freeze.get("inputs_sha256")
        input_label = "skipped input hash"
    if actual_input_hash != freeze.get("inputs_sha256"):
        mismatches.append({"file": input_label, "expected": freeze.get("inputs_sha256"),
                           "actual": actual_input_hash})
    if hash_mode != "skip":
        gold_hash = _sha256(gold_path) if gold_path.exists() else None
        if gold_hash != freeze.get("provisional_reference_sha256"):
            mismatches.append({"file": str(gold_path),
                               "expected": freeze.get("provisional_reference_sha256"),
                               "actual": gold_hash})
    else:
        skipped.extend(["canonical inputs", "provisional reference"])
    return mismatches, skipped


def _preserved_run_declarations(freeze: dict[str, Any]) -> dict[str, Any]:
    """Combine the current field with the freeze07 legacy field."""
    preserved: dict[str, Any] = {}
    for field in ("preserved_runs", "preserved_failed_runs"):
        value = freeze.get(field, {})
        if isinstance(value, dict):
            preserved.update(value)
    return preserved


def _frozen_baseline_audit(freeze: dict[str, Any], repo_root: Path,
                           original_run: Path, staged_run: Path) -> list[dict[str, Any]]:
    """Verify the exact retained baselines used by a preregistered comparison."""
    declared = freeze.get("frozen_baselines")
    if declared is None:
        return []
    if not isinstance(declared, dict) or set(declared) != {"original", "staged"}:
        return [{"arm": "all", "error": "invalid frozen baseline declaration"}]
    mismatches: list[dict[str, Any]] = []
    for arm, actual_run in (("original", original_run), ("staged", staged_run)):
        contract = declared.get(arm)
        if (not isinstance(contract, dict) or set(contract) != {"path", "files"} or
                not isinstance(contract.get("path"), str) or
                not isinstance(contract.get("files"), dict)):
            mismatches.append({"arm": arm, "error": "invalid arm declaration"})
            continue
        try:
            actual_relative = actual_run.resolve().relative_to(repo_root).as_posix()
        except ValueError:
            actual_relative = str(actual_run.resolve())
        if actual_relative != contract["path"]:
            mismatches.append({"arm": arm, "file": "path",
                               "expected": contract["path"],
                               "actual": actual_relative})
        for relative, expected in contract["files"].items():
            if not isinstance(relative, str) or not isinstance(expected, str):
                mismatches.append({"arm": arm, "file": relative,
                                   "error": "invalid file hash declaration"})
                continue
            path = actual_run / relative
            actual = _sha256(path) if path.is_file() else None
            if actual != expected:
                mismatches.append({"arm": arm, "file": relative,
                                   "expected": expected, "actual": actual})
    return mismatches


def _frozen_prior_artifact_audit(freeze: dict[str, Any],
                                 repo_root: Path) -> list[dict[str, Any]]:
    """Bind every preregistered historical comparison/gate to its exact bytes."""
    declared = freeze.get("prior_artifacts")
    if declared is None:
        return []
    if not isinstance(declared, dict):
        return [{"error": "invalid prior artifact declaration"}]
    mismatches: list[dict[str, Any]] = []
    for relative, expected in declared.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            mismatches.append({"file": relative,
                               "error": "invalid file hash declaration"})
            continue
        path = repo_root / relative
        actual = _sha256(path) if path.is_file() else None
        if actual != expected:
            mismatches.append({"file": relative, "expected": expected,
                               "actual": actual})
    return mismatches


def _v4_audit_state(comparison: dict[str, Any], case_ids: list[str],
                    required_rounds: list[int] | dict[str, list[int]],
                    search_rounds: dict[str, int] | None = None,
                    provider_mode: str = "task_routed",
                    probe_owned_novel_second_pass_cases: set[str] | None = None
                    ) -> tuple[bool, dict[str, Any]]:
    """Fail closed over the scorer's independently reconstructed v4 ledgers.

    The scorer derives this section from the frozen target plan, raw model-call
    requests, psi history, final report and retrieval history.  The gate
    consumes only exact counts; it does not trust a runner-authored pass flag.
    """
    coverage = comparison.get("target_plan_probe_coverage")
    audit = coverage.get("v4_audit") if isinstance(coverage, dict) else None
    if (not isinstance(audit, dict) or
            set(audit) != {"applicable_cases", "totals", "cases"}):
        return False, {"error": "missing target_plan_probe_coverage.v4_audit"}
    totals = audit.get("totals")
    cases = audit.get("cases")
    if not isinstance(totals, dict) or not isinstance(cases, list):
        return False, {"error": "malformed v4 audit totals or cases", "audit": audit}

    fields = ("coverage_ledger", "material_probe_ledger",
              "strict_followups", "retrieval_attribution",
              "probe_delta_attribution", "live_artifact_receipts")

    def integer(value: Any) -> bool:
        return type(value) is int and value >= 0

    case_verification_counts = {
        case_id: len(required_rounds.get(case_id, []))
        if isinstance(required_rounds, dict) else len(required_rounds)
        for case_id in case_ids}
    case_search_counts = {
        case_id: (search_rounds.get(case_id, 0) if isinstance(search_rounds, dict)
                  else case_verification_counts[case_id])
        for case_id in case_ids}

    round_rows = comparison.get("round1_to_final", {}).get(
        "extension", {}).get("cases")
    label_change_cases = None
    if (isinstance(round_rows, list) and
            [item.get("id") if isinstance(item, dict) else None
             for item in round_rows] == case_ids):
        label_change_cases = {item["id"] for item in round_rows
                              if item.get("first_round") != item.get("final")}

    def validate(block: Any, *, expected_search_rounds: int,
                 expected_verification_rounds: int,
                 aggregate: bool = False,
                 expected_probe_owned_novel_cases: int | None = None,
                 expected_label_changes: int | None = None) -> bool:
        if not isinstance(block, dict) or set(block) != set(fields):
            return False
        coverage_block = block["coverage_ledger"]
        material_block = block["material_probe_ledger"]
        followup_block = block["strict_followups"]
        retrieval_block = block["retrieval_attribution"]
        delta_block = block["probe_delta_attribution"]
        receipt_block = block["live_artifact_receipts"]
        expected_keys = {
            "coverage_ledger": {"expected_segments", "ledger_entries",
                                "required_dimensions", "covered_dimensions", "breaks"},
            "material_probe_ledger": {"stage_calls", "audited_stage_calls",
                                      "expected_probe_checks", "probe_checks", "findings",
                                      "referenced_findings", "breaks"},
            "strict_followups": {"unresolved_probe_slots", "task_followups",
                                 "allowed_stops", "covered_slots", "breaks"},
            "retrieval_attribution": {"search_rounds", "issued_tasks",
                                      "provider_returns", "attributed_returns", "task_links",
                                      "valid_task_links", "later_probe_owned_tasks",
                                      "later_probe_owned_hit_tasks", "breaks"},
            "probe_delta_attribution": {
                "round_transitions", "probe_slots_compared", "semantic_deltas",
                "basis_drifts", "traced_semantic_deltas",
                "graph_traced_semantic_deltas",
                "probe_owned_novel_second_pass_cases", "label_changes",
                "label_changes_with_decisive_delta", "breaks"},
            "live_artifact_receipts": {
                "available", "retained_calls", "request_digests_verified",
                "psi_history_calls", "psi_calls_joined",
                "verification_history_calls", "verification_calls_joined",
                "accepted_material_transactions", "direct_attributed_transactions",
                "direct_stage_calls", "direct_stage_calls_with_exact_receipt",
                "provenance_only_direct_transactions", "revisit_transactions",
                "revisit_stage_calls_with_empty_receipt", "layer_calls",
                "layer_calls_with_exact_projection", "layer_receipt_deliveries",
                "second_pass_layer_calls", "second_pass_receipt_deliveries",
                "cross_layer_leaks", "breaks"},
        }
        for name, value in (("coverage_ledger", coverage_block),
                            ("material_probe_ledger", material_block),
                            ("strict_followups", followup_block),
                            ("retrieval_attribution", retrieval_block),
                            ("probe_delta_attribution", delta_block),
                            ("live_artifact_receipts", receipt_block)):
            if (not isinstance(value, dict) or set(value) != expected_keys[name] or
                    any(not integer(item) for item in value.values())):
                return False
        strict_retrieval_ok = (
            retrieval_block["attributed_returns"] == retrieval_block["provider_returns"] and
            retrieval_block["task_links"] > 0 and
            retrieval_block["valid_task_links"] == retrieval_block["task_links"] and
            retrieval_block["later_probe_owned_hit_tasks"] <=
                retrieval_block["later_probe_owned_tasks"] <=
                retrieval_block["issued_tasks"] and
            (not aggregate or retrieval_block["later_probe_owned_tasks"] > 0 and
             retrieval_block["later_probe_owned_hit_tasks"] > 0)
        ) if provider_mode == "task_routed" else (
            provider_mode == "fixed_reanalysis" and
            retrieval_block["attributed_returns"] == 0 and
            retrieval_block["task_links"] == 0 and
            retrieval_block["valid_task_links"] == 0 and
            retrieval_block["later_probe_owned_tasks"] == 0 and
            retrieval_block["later_probe_owned_hit_tasks"] == 0
        )
        delta_ok = (
            delta_block["breaks"] == 0 and
            delta_block["round_transitions"] ==
                (max(0, expected_verification_rounds - (0 if aggregate else 1))
                 if provider_mode == "task_routed" else 0) and
            delta_block["traced_semantic_deltas"] ==
                delta_block["semantic_deltas"] and
            delta_block["semantic_deltas"] <=
                delta_block["probe_slots_compared"] and
            delta_block["basis_drifts"] <=
                delta_block["probe_slots_compared"] and
            ((delta_block["round_transitions"] == 0) ==
             (delta_block["probe_slots_compared"] == 0)) and
            delta_block["graph_traced_semantic_deltas"] <=
                delta_block["traced_semantic_deltas"] and
            delta_block["label_changes_with_decisive_delta"] ==
                delta_block["label_changes"] and
            (expected_probe_owned_novel_cases is None or
             delta_block["probe_owned_novel_second_pass_cases"] ==
                expected_probe_owned_novel_cases) and
            (expected_label_changes is None or
             delta_block["label_changes"] == expected_label_changes)
        )
        if provider_mode == "task_routed":
            delta_ok &= (delta_block["probe_owned_novel_second_pass_cases"] <=
                         delta_block["round_transitions"] and
                         delta_block["label_changes"] <=
                         delta_block["probe_owned_novel_second_pass_cases"])
            if not aggregate:
                delta_ok &= (delta_block["probe_owned_novel_second_pass_cases"] <= 1 and
                             delta_block["label_changes"] <= 1)
            if aggregate:
                delta_ok &= delta_block["probe_owned_novel_second_pass_cases"] > 0
        else:
            delta_ok &= all(delta_block[key] == 0 for key in delta_block)
        if provider_mode == "task_routed":
            has_second_pass = (expected_verification_rounds >
                               (0 if aggregate else 1))
            receipt_ok = (
                receipt_block["available"] > 0 and
                receipt_block["retained_calls"] > 0 and
                receipt_block["request_digests_verified"] ==
                    receipt_block["retained_calls"] and
                receipt_block["psi_history_calls"] > 0 and
                receipt_block["psi_calls_joined"] ==
                    receipt_block["psi_history_calls"] and
                receipt_block["verification_history_calls"] > 0 and
                receipt_block["verification_calls_joined"] ==
                    receipt_block["verification_history_calls"] and
                receipt_block["accepted_material_transactions"] >=
                    receipt_block["direct_attributed_transactions"] ==
                    retrieval_block["attributed_returns"] and
                receipt_block["direct_stage_calls"] > 0 and
                receipt_block["direct_stage_calls_with_exact_receipt"] ==
                    receipt_block["direct_stage_calls"] and
                receipt_block["revisit_stage_calls_with_empty_receipt"] >=
                    3 * receipt_block["revisit_transactions"] and
                receipt_block["layer_calls"] > 0 and
                receipt_block["layer_calls_with_exact_projection"] ==
                    receipt_block["layer_calls"] and
                receipt_block["cross_layer_leaks"] == 0 and
                receipt_block["breaks"] == 0 and
                ((receipt_block["second_pass_layer_calls"] > 0 and
                  receipt_block["second_pass_receipt_deliveries"] > 0)
                 if has_second_pass else
                 (receipt_block["second_pass_layer_calls"] == 0 and
                  receipt_block["second_pass_receipt_deliveries"] == 0))
            )
        else:
            receipt_ok = (provider_mode == "fixed_reanalysis" and
                          all(value == 0 for value in receipt_block.values()))
        return (
            coverage_block["breaks"] == 0 and
            coverage_block["expected_segments"] > 0 and
            coverage_block["ledger_entries"] == coverage_block["expected_segments"] and
            coverage_block["required_dimensions"] > 0 and
            coverage_block["covered_dimensions"] == coverage_block["required_dimensions"] and
            material_block["breaks"] == 0 and
            material_block["stage_calls"] > 0 and
            material_block["audited_stage_calls"] == material_block["stage_calls"] and
            material_block["expected_probe_checks"] > 0 and
            material_block["probe_checks"] == material_block["expected_probe_checks"] and
            material_block["referenced_findings"] == material_block["findings"] and
            followup_block["breaks"] == 0 and
            followup_block["covered_slots"] == followup_block["unresolved_probe_slots"] and
            followup_block["task_followups"] + followup_block["allowed_stops"] ==
                followup_block["covered_slots"] and
            retrieval_block["breaks"] == 0 and
            retrieval_block["search_rounds"] == expected_search_rounds and
            retrieval_block["issued_tasks"] > 0 and
            retrieval_block["provider_returns"] > 0 and
            strict_retrieval_ok and delta_ok and receipt_ok
        )

    case_ids_seen = [item.get("id") for item in cases if isinstance(item, dict)]
    cases_ok = (len(cases) == len(case_ids) and case_ids_seen == case_ids and
                all(isinstance(item, dict) and
                    set(item) == {"id", *fields} and
                    validate({key: item.get(key) for key in fields},
                             expected_search_rounds=case_search_counts[case_id],
                             expected_verification_rounds=
                                 case_verification_counts[case_id],
                             expected_probe_owned_novel_cases=(
                                 int(case_id in probe_owned_novel_second_pass_cases)
                                 if probe_owned_novel_second_pass_cases is not None else None),
                             expected_label_changes=(
                                 int(case_id in label_change_cases)
                                 if label_change_cases is not None else None))
                    for case_id, item in zip(case_ids, cases)
                    ))
    totals_match_cases = cases_ok and all(
        totals.get(block, {}).get(field) == sum(
            item.get(block, {}).get(field, 0) for item in cases)
        for block in fields
        for field in totals.get(block, {})
    )
    passed = (audit.get("applicable_cases") == len(case_ids) and
              validate({key: totals.get(key) for key in fields},
                       expected_search_rounds=sum(case_search_counts.values()),
                       expected_verification_rounds=sum(
                           case_verification_counts.values()) - len(case_ids),
                       aggregate=True,
                       expected_probe_owned_novel_cases=(
                           len(probe_owned_novel_second_pass_cases)
                           if probe_owned_novel_second_pass_cases is not None else None),
                       expected_label_changes=(len(label_change_cases)
                                               if label_change_cases is not None else None)) and
              cases_ok and totals_match_cases)
    return passed, {
        "applicable_cases": audit.get("applicable_cases"),
        "expected_case_ids": case_ids,
        "case_ids": case_ids_seen,
        "totals": totals,
        "cases": cases,
        "totals_match_cases": totals_match_cases,
        "verification_rounds": case_verification_counts,
        "search_rounds": case_search_counts,
        "expected_probe_owned_novel_second_pass_cases": (
            sorted(probe_owned_novel_second_pass_cases)
            if probe_owned_novel_second_pass_cases is not None else None),
        "label_change_cases": (sorted(label_change_cases)
                               if label_change_cases is not None else None),
    }


def _outer_loop_audit(rows: dict[str, dict[str, Any]],
                      reports: dict[str, dict[str, Any]],
                      retrievals: dict[str, Any], case_ids: list[str],
                      max_rounds: int, provider_mode: str | None
                      ) -> tuple[bool, dict[str, Any], dict[str, list[int]], dict[str, int]]:
    """Independently classify fixed cycles and adaptive task-routed opportunities."""
    task_routed = provider_mode == "task_routed"
    verification_rounds: dict[str, list[int]] = {}
    search_rounds: dict[str, int] = {}
    cases = []
    for case_id in case_ids:
        row = rows.get(case_id, {})
        report = reports.get(case_id, {})
        retrieval = retrievals.get(case_id)
        usage = report.get("usage") if isinstance(report, dict) else None
        history = report.get("verification_history") if isinstance(report, dict) else None
        operations = report.get("operations") if isinstance(report, dict) else None
        valid = (type(max_rounds) is int and max_rounds > 0 and
                 isinstance(usage, dict) and isinstance(history, list) and
                 isinstance(operations, list) and isinstance(retrieval, list) and
                 row.get("engine_usage") == usage)
        operation_sequences = ([item.get("sequence") if isinstance(item, dict) else None
                                for item in operations]
                               if isinstance(operations, list) else [])
        valid &= operation_sequences == list(range(1, len(operation_sequences) + 1))
        accepted = ([item.get("round") if isinstance(item, dict) else None
                     for item in history] if isinstance(history, list) else [])
        verification_rounds[case_id] = accepted
        searched = usage.get("rounds") if isinstance(usage, dict) else None
        search_rounds[case_id] = searched if type(searched) is int else 0
        verification_calls = (usage.get("verification_calls")
                              if isinstance(usage, dict) else None)
        valid &= (type(verification_calls) is int and
                  accepted == list(range(1, verification_calls + 1)))
        retrieval_rounds = ([item.get("round") if isinstance(item, dict) else None
                             for item in retrieval]
                            if isinstance(retrieval, list) else [])
        valid &= (type(searched) is int and len(retrieval_rounds) == searched and
                  retrieval_rounds == list(range(1, searched + 1)))
        probe_owned_novel_later_hits = 0
        novel_later_returns = 0
        reanalysis_later_returns = 0
        returned_before: set[str] = set()
        duplicate_by_round: dict[int, set[str]] = {}
        probe_owned_novel_drivers_by_round: dict[int, list[dict[str, Any]]] = {}
        if task_routed and isinstance(retrieval, list):
            for record in retrieval:
                if not isinstance(record, dict):
                    valid = False
                    continue
                round_number = record.get("round")
                tasks = record.get("tasks")
                returned = record.get("returned")
                attribution = record.get("attribution")
                feedback = record.get("feedback")
                if not all(isinstance(value, list)
                           for value in (tasks, returned, attribution, feedback)):
                    valid = False
                    continue
                task_map = {item.get("id"): item for item in tasks
                            if isinstance(item, dict) and isinstance(item.get("id"), str)}
                task_order = {item.get("id"): index for index, item in enumerate(tasks)
                              if isinstance(item, dict) and
                              isinstance(item.get("id"), str)}
                valid &= len(task_map) == len(tasks)
                valid &= ([item.get("version_id") if isinstance(item, dict) else None
                           for item in attribution] == returned and
                          len(returned) == len(set(returned)))
                hit_tasks: set[str] = set()
                duplicates = set(returned) & returned_before
                if type(round_number) is int:
                    duplicate_by_round[round_number] = duplicates
                    probe_owned_novel_drivers_by_round[round_number] = []
                for item in attribution:
                    task_ids = item.get("task_ids") if isinstance(item, dict) else None
                    item_valid = (isinstance(task_ids, list) and bool(task_ids) and
                                  len(task_ids) == len(set(task_ids)) and
                                  set(task_ids) <= set(task_map))
                    valid &= item_valid
                    if not item_valid:
                        continue
                    if item["version_id"] in duplicates:
                        valid &= all(task_map[task_id].get("action") == "reanalyse"
                                     for task_id in task_ids)
                    if type(round_number) is int and round_number > 1:
                        novel = item["version_id"] not in duplicates
                        qualifying_task_ids = sorted((task_id for task_id in task_ids
                            if task_map[task_id].get("stage") == "verification" and
                            task_map[task_id].get("dimension") in {"evidence", "world"} and
                            task_map[task_id].get("probe_id") is not None and
                            task_map[task_id].get("action") in {"fetch", "search"}),
                            key=task_order.__getitem__)
                        novel_later_returns += int(novel)
                        reanalysis_later_returns += int(not novel)
                        if novel and qualifying_task_ids:
                            driver = {
                                "version_id": item["version_id"],
                                "task_ids": qualifying_task_ids,
                                "probe_ids": list(dict.fromkeys(
                                    task_map[task_id].get("probe_id")
                                    for task_id in qualifying_task_ids)),
                            }
                            probe_owned_novel_drivers_by_round[round_number].append(driver)
                            probe_owned_novel_later_hits += 1
                    hit_tasks.update(task_ids)
                exhausted: set[str] = set()
                for item in feedback:
                    task_id = item.get("task_id") if isinstance(item, dict) else None
                    item_valid = (task_id in task_map and
                                  item.get("status") == "corpus_exhausted" and
                                  item.get("probe_id") == task_map[task_id].get("probe_id") and
                                  task_id not in exhausted)
                    valid &= item_valid
                    if item_valid:
                        exhausted.add(task_id)
                valid &= not (hit_tasks & exhausted) and hit_tasks | exhausted == set(task_map)
                round_ops = [item for item in operations
                             if isinstance(item, dict) and
                             item.get("round") == round_number]
                search_ops = [item for item in round_ops
                              if item.get("action") == "search"]
                valid &= (len(search_ops) == 1 and
                          search_ops[0].get("tasks") == tasks)
                feedback_ops = [item.get("feedback") for item in round_ops
                                if item.get("action") == "retrieval_feedback"]
                valid &= feedback_ops == feedback
                for attribution_item in attribution:
                    attribution_ops = [item for item in round_ops
                        if item.get("action") == "retrieval_attribution_validated" and
                        item.get("version_id") == attribution_item.get("version_id")]
                    valid &= (len(attribution_ops) == 1 and
                              attribution_ops[0].get("trigger_task_ids") ==
                              attribution_item.get("task_ids"))
                if type(round_number) is int and round_number > 1:
                    expected_drivers = probe_owned_novel_drivers_by_round.get(
                        round_number, [])
                    driver_events = [{key: item.get(key) for key in
                                      ("version_id", "task_ids", "probe_ids")}
                                     for item in round_ops
                                     if item.get("action") ==
                                     "probe_owned_novel_return_accepted"]
                    valid &= driver_events == expected_drivers
                returned_before.update(returned)

            valid &= bool(retrieval and retrieval[0].get("returned"))

            by_round: dict[int, list[dict[str, Any]]] = {}
            for operation in operations if isinstance(operations, list) else []:
                if isinstance(operation, dict):
                    by_round.setdefault(operation.get("round"), []).append(operation)
            for round_number in accepted[1:]:
                returned = retrieval[round_number - 1].get("returned", [])
                round_ops = by_round.get(round_number, [])
                starts = [item for item in round_ops
                          if item.get("action") == "verification_started"]
                expected_drivers = probe_owned_novel_drivers_by_round.get(round_number, [])
                valid &= bool(returned) and bool(expected_drivers) and len(starts) == 1
                valid &= (len(starts) == 1 and
                          starts[0].get("probe_owned_novel_returns") == expected_drivers)
                driver_events = [{key: item.get(key) for key in
                                  ("version_id", "task_ids", "probe_ids")}
                                 for item in round_ops
                                 if item.get("action") ==
                                 "probe_owned_novel_return_accepted"]
                valid &= driver_events == expected_drivers
                start = starts[0].get("sequence") if len(starts) == 1 else -1
                valid &= all(item.get("sequence", start) < start for item in round_ops
                             if item.get("action") ==
                             "probe_owned_novel_return_accepted")
                for version_id in returned:
                    attr = [item.get("sequence") for item in round_ops
                            if item.get("action") == "retrieval_attribution_validated" and
                            item.get("version_id") == version_id]
                    action = ("duplicate_observed" if version_id in
                              duplicate_by_round.get(round_number, set()) else "snapshot_saved")
                    saved = [item.get("sequence") for item in round_ops
                             if item.get("action") == action and
                             item.get("version_id") == version_id and
                             item.get("eligible") is True]
                    decomposed = [item.get("sequence") for item in round_ops
                                  if item.get("action") == "decompose_completed" and
                                  item.get("version_id") == version_id]
                    valid &= (len(attr) == 1 and len(saved) == 1 and bool(decomposed) and
                              attr[0] < saved[0] < min(decomposed) < start)
            if type(verification_calls) is int and type(searched) is int:
                valid &= 1 <= verification_calls <= max_rounds
                valid &= verification_calls <= searched <= max_rounds
                if searched > verification_calls:
                    final = retrieval[-1]
                    final_round = final.get("round")
                    final_drivers = probe_owned_novel_drivers_by_round.get(final_round, [])
                    empty_exhaustion = (not final.get("returned") and
                                        bool(final.get("feedback")) and
                                        report.get("stop_reason") == "provider_exhausted")
                    nonqualifying_return = (bool(final.get("returned")) and
                                            not final_drivers and
                                            report.get("stop_reason") ==
                                            "no_probe_owned_novel_evidence" and
                                            report.get("decision_status") == "unresolved")
                    final_ops = by_round.get(final_round, [])
                    skipped = [item for item in final_ops
                               if item.get("action") == "verification_skipped"]
                    if nonqualifying_return:
                        valid &= (len(skipped) == 1 and
                                  skipped[0].get("reason") ==
                                  "no_probe_owned_novel_return")
                    valid &= (searched == verification_calls + 1 and
                              (empty_exhaustion or nonqualifying_return))
                elif verification_calls == searched == 1:
                    valid &= report.get("stop_reason") == "complete"
            valid &= report.get("retrieval_attribution_mode") == "strict"
        else:
            valid &= (provider_mode == "fixed_reanalysis" and
                      verification_calls == max_rounds and searched == max_rounds and
                      accepted == list(range(1, max_rounds + 1)) and
                      report.get("retrieval_attribution_mode") == "legacy-compatible")

        cases.append({
            "id": case_id, "passed": bool(valid),
            "verification_rounds": accepted, "search_rounds": searched,
            "stop_reason": report.get("stop_reason") if isinstance(report, dict) else None,
            "loop_opportunity": bool(task_routed and type(searched) is int and searched > 1),
            "second_pass": bool(task_routed and type(verification_calls) is int and
                                verification_calls > 1),
            "probe_owned_novel_later_hits": probe_owned_novel_later_hits,
            "novel_later_returns": novel_later_returns,
            "reanalysis_later_returns": reanalysis_later_returns,
        })
    state = {
        "mode": provider_mode, "cases": cases,
        "loop_opportunity_cases": [item["id"] for item in cases
                                   if item["loop_opportunity"]],
        "second_pass_cases": [item["id"] for item in cases if item["second_pass"]],
        "probe_owned_novel_second_pass_cases": [item["id"] for item in cases
            if item["probe_owned_novel_later_hits"] > 0 and item["second_pass"]],
    }
    return all(item["passed"] for item in cases), state, verification_rounds, search_rounds


def _smoke_label_audit(comparison: dict[str, Any], case_ids: list[str],
                       expected_labels: dict[str, str],
                       provider_mode: str | None,
                       probe_owned_novel_second_pass_cases: set[str] | None = None
                       ) -> dict[str, Any]:
    """Recompute smoke label gates without forbidding a task-routed fix."""
    scheduled = len(case_ids)
    task_routed = provider_mode == "task_routed"
    arms = comparison.get("arms", {})
    extension = arms.get("extension", {})
    accuracy_state = {key: extension.get(key) for key in ("completed", "correct")}
    accuracy_ok = accuracy_state == {"completed": scheduled, "correct": scheduled}

    baseline_state = {arm: {key: arms.get(arm, {}).get(key)
                            for key in ("completed", "correct")}
                      for arm in ("original", "staged")}
    baselines_ok = all(value == {"completed": scheduled, "correct": scheduled}
                       for value in baseline_state.values())

    pair_state: dict[str, dict[str, Any]] = {}
    pair_ok = True
    for baseline in ("original", "staged"):
        values = comparison.get("label_comparisons", {}).get(
            f"extension_vs_{baseline}", {}).get("all_scheduled", {})
        compact = {key: values.get(key) for key in
                   ("case_pairs", "case_ids", "fixes", "breaks", "candidate_correct")}
        pair_state[baseline] = compact
        fixes = compact["fixes"]
        pair_ok &= (compact["case_pairs"] == scheduled and
                    compact["case_ids"] == case_ids and
                    compact["candidate_correct"] == scheduled and
                    compact["breaks"] == 0 and type(fixes) is int and
                    0 <= fixes <= scheduled and
                    (provider_mode in {"task_routed", "fixed_reanalysis"} or
                     fixes == 0))

    round_change = comparison.get("round1_to_final", {}).get("extension", {})
    round_cases = round_change.get("cases")
    per_case_ok = isinstance(round_cases, list) and len(round_cases) == scheduled
    if per_case_ok:
        per_case_ok = [item.get("id") if isinstance(item, dict) else None
                       for item in round_cases] == case_ids
    expected_first_correct = 0
    expected_fixes = 0
    if per_case_ok:
        for item in round_cases:
            case_id = item["id"]
            expected = expected_labels.get(case_id)
            first = item.get("first_round")
            first_correct = first == expected
            row_ok = (
                isinstance(first, str) and item.get("reference") == expected and
                item.get("paired") is True and item.get("final") == expected and
                item.get("first_correct") is first_correct and
                item.get("final_correct") is True and
                (task_routed or first_correct)
            )
            if probe_owned_novel_second_pass_cases is not None:
                changed = first != item.get("final")
                row_ok &= ((case_id in probe_owned_novel_second_pass_cases) or
                           not changed)
            per_case_ok &= row_ok
            expected_first_correct += int(first_correct)
            expected_fixes += int(not first_correct)
    round_state = {key: round_change.get(key) for key in
                   ("label_evaluable_scheduled_cases",
                    "paired_completed_cases_with_round1", "first_round_correct",
                    "final_correct", "fixes", "breaks")}
    round_state["cases"] = round_cases
    round_ok = (
        per_case_ok and
        round_state["label_evaluable_scheduled_cases"] == scheduled and
        round_state["paired_completed_cases_with_round1"] == scheduled and
        round_state["first_round_correct"] == expected_first_correct and
        round_state["final_correct"] == scheduled and
        round_state["fixes"] == expected_fixes and
        round_state["breaks"] == 0 and
        (task_routed or expected_fixes == 0)
    )
    return {
        "task_routed": task_routed,
        "accuracy_ok": accuracy_ok, "accuracy_state": accuracy_state,
        "baselines_ok": baselines_ok, "baseline_state": baseline_state,
        "pairs_ok": pair_ok, "pair_state": pair_state,
        "round_ok": round_ok, "round_state": round_state,
        "probe_owned_novel_second_pass_cases": (
            sorted(probe_owned_novel_second_pass_cases)
            if probe_owned_novel_second_pass_cases is not None else None),
    }


def _evaluate_gate(freeze: dict[str, Any], comparison: dict[str, Any] | None,
                   comparison_error: str | None, freeze_path: Path, gold_path: Path,
                   original_run: Path, staged_run: Path, extension_run: Path,
                   hash_mode: str = "current") -> dict[str, Any]:
    checks = _Checks()
    repo_root = freeze_path.resolve().parents[1]
    smoke_cases = freeze.get("smoke_cases", [])
    rounds_count = freeze.get("outer_rounds")
    required_rounds = (list(range(1, rounds_count + 1))
                       if isinstance(rounds_count, int) and not isinstance(rounds_count, bool)
                       and rounds_count > 0 else [])
    gold = _read(gold_path)
    gold_rows = gold.get("cases", []) if isinstance(gold, dict) else []
    gold_by_id = {item.get("id"): item for item in gold_rows if isinstance(item, dict)}
    expected_labels = {case_id: gold_by_id.get(case_id, {}).get("decision")
                       for case_id in smoke_cases}
    smoke_contract_ok = (isinstance(smoke_cases, list) and bool(smoke_cases) and
                         len(smoke_cases) == len(set(smoke_cases)) and required_rounds and
                         all(isinstance(case_id, str) and case_id in gold_by_id and
                             isinstance(expected_labels[case_id], str)
                             for case_id in smoke_cases))
    checks.add("freeze.smoke_contract", smoke_contract_ok,
               "unique smoke cases, positive round count, and one gold decision per case",
               {"cases": smoke_cases, "rounds": required_rounds,
                "expected_labels": expected_labels})

    if freeze.get("target_plan_schema") == PLAN_SCHEMA_V4:
        prior = freeze.get("prior_artifacts")
        provider_mode = freeze.get("provider_mode")
        v4_freeze_ok = (
            freeze.get("status") == "frozen_before_live_calls" and
            freeze.get("model") == "gpt-6-astra" and
            freeze.get("reasoning_effort") == "medium" and
            freeze.get("outer_rounds") == 2 and
            freeze.get("workers") == 1 and
            freeze.get("automatic_transport_retries") == 0 and
            provider_mode in V4_EXPERIMENT_BY_PROVIDER and
            freeze.get("experiment") == V4_EXPERIMENT_BY_PROVIDER.get(provider_mode) and
            freeze.get("strict_retrieval_attribution") is
                (provider_mode == "task_routed") and
            freeze.get("retrieval_attribution_mode") ==
                ("strict" if provider_mode == "task_routed" else "legacy") and
            freeze.get("provider") == V4_PROVIDER_LABELS.get(provider_mode) and
            isinstance(prior, dict) and bool(prior)
        )
        checks.add("freeze.v4_execution_contract", v4_freeze_ok,
                   "v4 experiment/provider names match; task-routed is strict, fixed reanalysis is not; prior manifest is nonempty",
                   {"experiment": freeze.get("experiment"),
                    "status": freeze.get("status"),
                    "model": freeze.get("model"),
                    "reasoning_effort": freeze.get("reasoning_effort"),
                    "outer_rounds": freeze.get("outer_rounds"),
                    "workers": freeze.get("workers"),
                    "automatic_transport_retries": freeze.get(
                        "automatic_transport_retries"),
                    "provider_mode": freeze.get("provider_mode"),
                    "strict_retrieval_attribution": freeze.get(
                        "strict_retrieval_attribution"),
                    "retrieval_attribution_mode": freeze.get(
                        "retrieval_attribution_mode"),
                    "provider": freeze.get("provider"),
                    "prior_artifact_count": len(prior) if isinstance(prior, dict) else None})

    hash_mismatches, hash_skipped = _frozen_hash_audit(
        freeze, repo_root, gold_path, extension_run, hash_mode)
    checks.add("freeze.hashes", not hash_mismatches,
               f"frozen hashes match in {hash_mode!r} mode",
               {"mismatches": hash_mismatches, "unavailable_or_skipped": hash_skipped},
               "Executed mode validates captured runtime files and recorded runtime hashes; "
               "non-executed historical files are disclosed rather than read from a changed workspace.")
    if freeze.get("target_plan_schema") in STRICT_PLAN_SCHEMAS:
        strict_hash_check = ("freeze.v4_complete_current_hashes"
                             if freeze.get("target_plan_schema") == PLAN_SCHEMA_V4
                             else "freeze.v3_complete_current_hashes")
        checks.add(strict_hash_check,
                   hash_mode == "current" and not hash_skipped,
                   "v3/v4 gates require current mode with no skipped source, test, document, input or gold hash",
                   {"hash_mode": hash_mode, "skipped": hash_skipped})

    baseline_mismatches = _frozen_baseline_audit(
        freeze, repo_root, original_run, staged_run)
    checks.add("freeze.baseline_hashes", not baseline_mismatches,
               "the exact preregistered original and staged baseline artifacts",
               baseline_mismatches)
    if "prior_artifacts" in freeze:
        prior_mismatches = _frozen_prior_artifact_audit(freeze, repo_root)
        checks.add("freeze.prior_artifact_hashes", not prior_mismatches,
                   "every preregistered historical comparison and gate artifact",
                   prior_mismatches)

    try:
        relative_run = extension_run.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        relative_run = str(extension_run.resolve())
    checks.add("freeze.next_run", relative_run == freeze.get("next_run"),
               freeze.get("next_run"), relative_run)
    preserved = _preserved_run_declarations(freeze)
    missing_preserved = [relative for relative in preserved if not (repo_root / relative).is_dir()]
    checks.add("freeze.preserved_run_directories", not missing_preserved,
               "every named preserved-run directory still exists", missing_preserved,
               "Existence is checkable; byte-for-byte non-overwrite history is not.")

    secret_findings = _secret_shape_audit(
        _gate_secret_paths(freeze, repo_root, freeze_path, gold_path,
                           original_run, staged_run, extension_run), repo_root)
    checks.add("security.no_secret_shapes", not secret_findings,
               "no recognized API-key or access-token shape occurs in frozen inputs, source, baselines, prior artifacts, or run artifacts",
               secret_findings,
               "Findings disclose only a path and detector name; matched bytes are never retained.")

    required = [extension_run / name for name in
                ("config.json", "inputs.json", "results.json", "status.json")]
    suffixes = ["calls", "report", "result", "target-plan", "verification-history"]
    if freeze.get("target_plan_schema") == PLAN_SCHEMA_V4:
        suffixes.extend(("psi-history", "retrieval"))
    required.extend(extension_run / f"{case_id}-{suffix}.json"
                    for case_id in smoke_cases for suffix in suffixes)
    missing = sorted(path.name for path in required if not path.exists())
    checks.add("execution.required_artifacts", not missing,
               "all aggregate and per-case smoke artifacts exist", missing)
    if missing:
        checks.add("comparison.independent_audit", False,
                   "independent scorer completes without error", comparison_error or "not run")
        return checks.finish(comparison_error)

    config = _read(extension_run / "config.json")
    inputs = _case_inputs(_read(extension_run / "inputs.json"))
    rows = _rows(_read(extension_run / "results.json"))
    status = _read(extension_run / "status.json")
    plans = {case_id: _read(extension_run / f"{case_id}-target-plan.json")
             for case_id in smoke_cases}
    reports = {case_id: _read(extension_run / f"{case_id}-report.json")
               for case_id in smoke_cases}
    retrievals = ({case_id: _read(extension_run / f"{case_id}-retrieval.json")
                   for case_id in smoke_cases}
                  if freeze.get("target_plan_schema") == PLAN_SCHEMA_V4 else {})

    config_ok, actual_config = _exact_config(freeze, config)
    checks.add("execution.frozen_config", config_ok,
               "run configuration equals the frozen smoke contract", actual_config)
    controls_ok, expected_controls, actual_controls = _execution_control_declarations(
        freeze, config, extension_run)
    checks.add("execution.transport_and_repair_declarations", controls_ok,
               expected_controls, actual_controls,
               "The executed SDK retry value is checked when captured model_io.py is available.")
    run_hashes = config.get("source_sha256", {})
    runtime_mismatches = [{"file": path, "expected": expected,
                           "actual": run_hashes.get(path)}
                          for path, expected in freeze.get("files", {}).items()
                          if path in run_hashes and run_hashes.get(path) != expected]
    checks.add("execution.executed_source_hashes", not runtime_mismatches,
               "all frozen files recorded by the run use their frozen hashes",
               runtime_mismatches)
    checks.add("execution.case_set", set(rows) == set(smoke_cases) == set(inputs),
               sorted(smoke_cases), {"results": sorted(rows), "inputs": sorted(inputs)})
    completed_errors = {case_id: {"status": row.get("status"),
                                  "error_type": row.get("error_type"),
                                  "error_stage": row.get("error_stage")}
                        for case_id, row in rows.items()
                        if row.get("status") != "completed" or row.get("error_type") or
                        row.get("error_stage")}
    report_errors = {case_id: report.get("errors") for case_id, report in reports.items()
                     if report.get("errors") or report.get("assessment_valid") is not True}
    checks.add("execution.completed_without_errors", not completed_errors and not report_errors,
               "every smoke case completed with a valid report and no errors",
               {"row_errors": completed_errors, "report_errors": report_errors})
    actual_rounds: list[int] | dict[str, list[int]] = required_rounds
    search_round_counts: dict[str, int] | None = None
    loop_state: dict[str, Any] | None = None
    if freeze.get("target_plan_schema") == PLAN_SCHEMA_V4:
        loop_ok, loop_state, case_rounds, search_round_counts = _outer_loop_audit(
            rows, reports, retrievals, smoke_cases, rounds_count,
            freeze.get("provider_mode"))
        actual_rounds = case_rounds
        loop_check_id = ("execution.adaptive_task_routed_outer_loop"
                         if freeze.get("provider_mode") == "task_routed" else
                         "execution.fixed_reanalysis_outer_loop")
        checks.add(loop_check_id, loop_ok,
                   ("each case has 1..2 real verification cycles, complete task outcomes, "
                    "and every later verification follows a valid material hit"
                    if freeze.get("provider_mode") == "task_routed" else
                    "each case has exactly two forced fixed-reanalysis cycles"),
                   loop_state)
        if freeze.get("provider_mode") == "task_routed":
            demonstrated = bool(loop_state.get(
                "probe_owned_novel_second_pass_cases"))
            checks.add("execution.task_routed_loop_demonstrated", demonstrated,
                       "at least one smoke case has a later frozen-probe task that returns material and triggers a real second verification",
                       {"loop_opportunity_cases": loop_state.get("loop_opportunity_cases"),
                        "second_pass_cases": loop_state.get("second_pass_cases"),
                        "probe_owned_novel_second_pass_cases": loop_state.get(
                            "probe_owned_novel_second_pass_cases")})
    checkpoint_errors = _checkpoint_audit(
        extension_run, rows, smoke_cases, actual_rounds)
    checks.add("execution.retained_rounds", not checkpoint_errors,
               "every accepted verification cycle has one matching retained checkpoint hash and no synthetic checkpoint",
               checkpoint_errors)
    call_errors = _call_audit(extension_run, rows, config, smoke_cases)
    checks.add("execution.calls_models_and_budgets", not call_errors,
               "all calls returned, used the frozen model, stopped cleanly, and stayed in budget",
               call_errors)
    status_errors = _status_audit(status, rows, len(smoke_cases))
    checks.add("execution.aggregate_status", not status_errors,
               "aggregate status and usage exactly reconcile", status_errors)

    checks.add("comparison.independent_audit", comparison_error is None and comparison is not None,
               "independent scorer completes without error", comparison_error)
    if comparison is not None:
        scheduled = len(smoke_cases)
        label_audit = _smoke_label_audit(
            comparison, smoke_cases, expected_labels, freeze.get("provider_mode"),
            (set(loop_state.get("probe_owned_novel_second_pass_cases", []))
             if freeze.get("provider_mode") == "task_routed" and
             isinstance(loop_state, dict) else None))
        checks.add("labels.smoke_accuracy", label_audit["accuracy_ok"],
                   f"extension completes and agrees with {scheduled}/{scheduled} provisional labels",
                   label_audit["accuracy_state"])
        checks.add("labels.completed_baselines", label_audit["baselines_ok"],
                   f"both baselines complete and agree on {scheduled}/{scheduled}",
                   label_audit["baseline_state"])
        pair_id = ("labels.task_routed_zero_baseline_breaks"
                   if label_audit["task_routed"]
                   else "labels.no_baseline_fixes_or_breaks")
        pair_expected = (f"{scheduled} final-label pairs, zero breaks, and fixes allowed"
                         if label_audit["task_routed"] else
                         f"{scheduled} pairs and zero fixes/breaks against each baseline")
        checks.add(pair_id, label_audit["pairs_ok"], pair_expected,
                   label_audit["pair_state"])
        round_id = ("labels.task_routed_round_changes"
                    if label_audit["task_routed"]
                    else "labels.no_round1_to_final_changes")
        round_expected = (
            f"{scheduled} per-case round-1 records, every final correct, fixes allowed, zero breaks"
            if label_audit["task_routed"] else
            f"{scheduled} per-case records with round 1 and final both correct and zero changes")
        checks.add(round_id, label_audit["round_ok"], round_expected,
                   label_audit["round_state"])

        coverage = comparison.get("target_plan_probe_coverage", {})
        totals = coverage.get("totals", {})
        cases = coverage.get("cases", [])
        expected_plan_schema = freeze.get("target_plan_schema")
        checks.add("probe_audit.frozen_plans", totals.get("plans_present") == scheduled and
                   totals.get("plans_valid") == scheduled and len(cases) == scheduled and
                   all(item.get("plan_contract") == expected_plan_schema and
                       item.get("plan_valid") is True for item in cases),
                   f"{scheduled} independently valid {expected_plan_schema} plans",
                   {"plans_present": totals.get("plans_present"),
                    "plans_valid": totals.get("plans_valid"),
                    "contracts": [item.get("plan_contract") for item in cases]})
        coverage_state = {key: totals.get(key) for key in
                          ("required_probe_coverage", "projection_coverage",
                           "executed_stage_projection_coverage")}
        checks.add("probe_audit.structural_coverage",
                   all(value == 1.0 for value in coverage_state.values()),
                   "binding, projection, and executed-stage coverage are each 1.0",
                   coverage_state)
        result_totals = comparison.get("target_probe_result_audit", {}).get("totals", {})
        cycle_state = {key: result_totals.get(key) for key in
                       ("accepted_judgement_cycles", "audited_judgement_cycles")}
        expected_cycles = (sum(len(value) for value in actual_rounds.values())
                           if isinstance(actual_rounds, dict) else
                           scheduled * len(actual_rounds))
        checks.add("probe_audit.accepted_cycles", cycle_state == {
                   "accepted_judgement_cycles": expected_cycles,
                   "audited_judgement_cycles": expected_cycles},
                   f"{expected_cycles} accepted and audited judgement cycles", cycle_state)
        checks.add("probe_audit.result_slot_coverage",
                   result_totals.get("result_slot_coverage") == 1.0,
                   1.0, result_totals.get("result_slot_coverage"))
        checks.add("probe_audit.grounded_conclusive_rate",
                   result_totals.get("grounded_conclusive_rate") == 1.0,
                   1.0, result_totals.get("grounded_conclusive_rate"))
        if freeze.get("target_plan_schema") == PLAN_SCHEMA_V4:
            v4_ok, v4_state = _v4_audit_state(
                comparison, smoke_cases, actual_rounds,
                search_rounds=search_round_counts,
                provider_mode=freeze.get("provider_mode"),
                probe_owned_novel_second_pass_cases=(set(loop_state.get(
                    "probe_owned_novel_second_pass_cases", []))
                    if freeze.get("provider_mode") == "task_routed" and
                    isinstance(loop_state, dict) else set()))
            checks.add("probe_audit.v4_end_to_end_ledgers", v4_ok,
                       "coverage/material/follow-up/retrieval/probe-delta ledgers are complete with zero breaks",
                       v4_state)
    else:
        task_routed = freeze.get("provider_mode") == "task_routed"
        pair_id = ("labels.task_routed_zero_baseline_breaks" if task_routed
                   else "labels.no_baseline_fixes_or_breaks")
        round_id = ("labels.task_routed_round_changes" if task_routed
                    else "labels.no_round1_to_final_changes")
        for identifier in ("labels.smoke_accuracy", "labels.completed_baselines",
                           pair_id, round_id, "probe_audit.frozen_plans",
                           "probe_audit.structural_coverage", "probe_audit.accepted_cycles",
                           "probe_audit.result_slot_coverage",
                           "probe_audit.grounded_conclusive_rate"):
            checks.add(identifier, False, "independent comparison audit data", None,
                       "Not evaluated because the independent scorer failed.")
        if freeze.get("target_plan_schema") == PLAN_SCHEMA_V4:
            checks.add("probe_audit.v4_end_to_end_ledgers", False,
                       "independent v4 scorer audit data", None,
                       "Not evaluated because the independent scorer failed.")

    label_state = {}
    for case_id, expected in expected_labels.items():
        row = rows.get(case_id, {})
        first = next((item.get("decision") for item in row.get("checkpoints", [])
                      if item.get("round") == 1), None)
        label_state[case_id] = {"round1": first, "final": row.get("prediction")}
    task_routed = freeze.get("provider_mode") == "task_routed"
    label_ok = all(
        state["final"] == expected and
        (task_routed and isinstance(state["round1"], str) or
         not task_routed and state["round1"] == expected)
        for case_id, expected in expected_labels.items()
        for state in (label_state[case_id],))
    label_expected = ({case_id: {"round1": "any observed decision", "final": expected}
                       for case_id, expected in expected_labels.items()}
                      if task_routed else
                      {case_id: {"round1": expected, "final": expected}
                       for case_id, expected in expected_labels.items()})
    label_id = ("labels.task_routed_final_case_labels" if task_routed
                else "labels.frozen_case_labels")
    checks.add(label_id, label_ok, label_expected, label_state)

    if "p02" in smoke_cases:
        p02_rounds = (actual_rounds.get("p02", [])
                      if isinstance(actual_rounds, dict) else actual_rounds)
        _p02_checks(checks, plans["p02"], reports["p02"], inputs["p02"]["target"],
                    p02_rounds)
    if ("p04" in smoke_cases and
            plans["p04"].get("schema_version") in STRICT_PLAN_SCHEMAS):
        p04_rounds = (actual_rounds.get("p04", [])
                      if isinstance(actual_rounds, dict) else actual_rounds)
        _p04_checks(checks, reports["p04"], plans["p04"], p04_rounds)
    if "p07" in smoke_cases:
        p07_rounds = (actual_rounds.get("p07", [])
                      if isinstance(actual_rounds, dict) else actual_rounds)
        _p07_checks(checks, plans["p07"], reports["p07"], p07_rounds)
    if "p08" in smoke_cases:
        p08_rounds = (actual_rounds.get("p08", [])
                      if isinstance(actual_rounds, dict) else actual_rounds)
        _p08_checks(checks, plans["p08"], reports["p08"], inputs["p08"],
                    p08_rounds)
    return checks.finish(comparison_error)


def check_smoke_gate(freeze_path: Path | str, gold_path: Path | str,
                     original_run: Path | str, staged_run: Path | str,
                     extension_run: Path | str, hash_mode: str = "current") -> dict[str, Any]:
    freeze_path = Path(freeze_path)
    gold_path = Path(gold_path)
    original_run = Path(original_run)
    staged_run = Path(staged_run)
    extension_run = Path(extension_run)
    freeze = _read(freeze_path)
    comparison = None
    error = None
    try:
        comparison = comparison_score(gold_path, original_run, staged_run, extension_run)
    except Exception as exc:  # The gate reports artifact failures instead of hiding them in a traceback.
        error = f"{type(exc).__name__}: {exc}"
    return _evaluate_gate(freeze, comparison, error, freeze_path, gold_path,
                          original_run, staged_run, extension_run, hash_mode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--original-run", required=True)
    parser.add_argument("--staged-run", required=True)
    parser.add_argument("--extension-run", required=True)
    parser.add_argument("--hash-mode", choices=("current", "executed", "skip"),
                        default="current")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = check_smoke_gate(args.freeze, args.gold, args.original_run,
                              args.staged_run, args.extension_run, args.hash_mode)
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
