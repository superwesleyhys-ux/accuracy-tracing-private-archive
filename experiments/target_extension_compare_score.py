"""Artifact-only comparison of original, staged, and target-extended runs.

The extension run selects the comparison denominator.  Baseline runs may contain
additional cases, but the scorer compares only identical selected target/material
contracts.  No inference adapter, credential, or network service is imported.

Plan coverage is structural: it shows that required checklist items exist in a
checksummed plan and that their projected stages ran.  V2/V3 reports are audited
separately for one grounded result per projected evidence/world probe in every
accepted judgement cycle.  Neither structural delivery nor result retention is
evidence that the answer is correct; both remain separate from labels and
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


PLAN_SCHEMA_V2 = "decision-probe-v2"
PLAN_SCHEMA_V3 = "decision-probe-v3"
VERSIONED_PLAN_SCHEMAS = {PLAN_SCHEMA_V2, PLAN_SCHEMA_V3}
EXTENSION_EXPERIMENTS = {
    "target_extended_psi_development_v1": "legacy-decision-probe-v1",
    "target_extended_psi_development_v2": PLAN_SCHEMA_V2,
    "target_extended_psi_development_v3": PLAN_SCHEMA_V3,
}
PLAN_STAGES = ("atoms", "lineage", "critic", "evidence", "world")
ROUTED_STAGES = ("atoms", "lineage", "evidence", "world")
DIMENSION_PROBES = {
    "negation": "polarity",
    "time": "time_boundary",
    "quantity_unit": "quantity_unit",
    "baseline_scope": "baseline_scope",
    "entity_identity": "entity_identity",
    "exact_designation": "exact_designation",
}

MATCH_POLICIES = {
    "entity_identity": "same_referent",
    "exact_designation": "exact_designation",
}

LEGACY_PROBE_ROUTES_AND_GATES = {
    "semantic_core": (("atoms", "evidence"), "always"),
    "polarity": (("atoms", "evidence"), "always"),
    "time_boundary": (("atoms", "evidence"), "always"),
    "quantity_unit": (("atoms", "evidence"), "always"),
    "baseline_scope": (("atoms", "evidence"), "always"),
    "condition_modality": (("atoms", "evidence"), "always"),
    "entity_identity": (("atoms", "evidence", "world"), "always"),
    "exact_designation": (("atoms", "evidence", "world"), "always"),
    "source_lineage": (("lineage", "world"), "provenance"),
    "source_independence": (("lineage", "world"), "positive_world_only"),
}

V2_PROBE_ROUTES_AND_GATES = {
    **LEGACY_PROBE_ROUTES_AND_GATES,
    **{kind: (("atoms", "evidence", "world"), "always") for kind in (
        "semantic_core", "polarity", "time_boundary", "quantity_unit",
        "baseline_scope", "condition_modality", "designation_relation",
        "conditional_relation", "comparison_relation", "causal_relation")},
}

PROBE_STATUSES = {"supported", "contradicted", "conflicting", "unresolved"}
DIRECT_LINEAGE_KINDS = {"quotes", "cites", "reprints", "translates", "derives"}
REFERENT_RELATIONS = {
    "not_applicable", "exact", "alias", "description", "anaphora",
    "ambiguous", "different", "unresolved",
}

PLAN_FIELDS_V2 = {"schema_version", "target_signature", "logic", "claims",
                  "probes", "notes", "sha256", "projections"}
PROJECTION_FIELDS_V2 = {"schema_version", "target_signature", "plan_sha256",
                        "stage", "logic", "claims", "probes", "notes"}
CLAIM_FIELDS = {"id", "statement", "anchor", "role", "dimensions",
                "parent_claim_id"}
DIMENSION_FIELDS = {"id", "kind", "anchor"}
ANCHOR_FIELDS = {"start", "end", "quote"}
PROBE_FIELDS_V2 = {"id", "claim_id", "kind", "dimension_ids", "question",
                   "decision_impact", "match_policy", "routes", "gate"}
CLAIM_ROLES = {"main", "conjunct", "alternative", "condition", "exception",
               "comparison", "cause", "effect", "attribution",
               "attributed_content"}
DIMENSION_KINDS = {"subject", "predicate", "quantity_unit", "time",
                   "baseline_scope", "negation", "condition", "modality",
                   "entity_identity", "exact_designation"}
PLAN_LOGICS = {"single", "and", "or", "conditional", "comparison", "causal",
               "attribution", "mixed"}
DESIGNATION_CUE = re.compile(
    r"\b(?:nam(?:e|ed|es|ing)|renam(?:e|ed|es|ing)|"
    r"call(?:s|ed|ing)?|titl(?:e|ed|es|ing)|"
    r"designat(?:e|ed|es|ing)|"
    r"label(?:s|ed|ing|led|ling)?|term(?:s|ed|ing)?|known\s+as|"
    r"official\s+(?:name|title|designation))\b", re.IGNORECASE)
ATTRIBUTION_CUE = re.compile(
    r"\b(?:reported|announced|stated|said|wrote|concluded|found)\s+that\b",
    re.IGNORECASE)


def _canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":")).encode()).hexdigest()


def _normalised_exact_contains(text, expected):
    haystack = " ".join(text.casefold().split())
    needle = " ".join(expected.casefold().split())
    if not needle:
        return False
    prefix = r"(?<!\w)" if needle[0].isalnum() else ""
    suffix = r"(?!\w)" if needle[-1].isalnum() else ""
    return re.search(prefix + re.escape(needle) + suffix, haystack) is not None


def _canonical_identity_question(anchor, match_policy):
    """Return the program-owned question for one anchored identity dimension."""
    quote = json.dumps(anchor["quote"], ensure_ascii=False)
    if match_policy == "same_referent":
        return ("Does the evidence identify the same referent as " + quote +
                " by exact mention, alias, unambiguous description or anaphora?")
    if match_policy == "exact_designation":
        return ("Does the evidence establish " + quote +
                " as the exact asserted name, title, label or designation?")
    raise ValueError("identity probe has an unknown match policy")


def _designation_labels(plan, claim_id):
    claim = next((item for item in plan.get("claims", [])
                  if item.get("id") == claim_id), None)
    if claim is None:
        return ()
    return tuple(item["anchor"]["quote"] for item in claim.get("dimensions", [])
                 if item.get("kind") == "exact_designation")


def _has_designation_basis(spans, labels):
    return bool(labels) and any(
        DESIGNATION_CUE.search(unit) and
        all(_normalised_exact_contains(unit, label) for label in labels)
        for span in spans
        for unit in re.split(r"(?<=[.!?;])\s+|\n+", span.get("quote", ""))
        if unit.strip()
    )


def _audit_v2_stable_ids(target, signature, claims, probes):
    """Recompute the planner-owned structural IDs from immutable anchors."""
    claim_ids = {}
    dimension_ids = {}
    for claim in claims:
        predicate = next(item for item in claim["dimensions"]
                         if item["kind"] == "predicate")["anchor"]
        key = (claim["anchor"]["start"], claim["anchor"]["end"], claim["role"],
               predicate["start"], predicate["end"])
        expected_claim = target["id"] + ":claim:" + _canonical_digest(
            [signature, *key])[:20]
        if claim["id"] != expected_claim:
            raise ValueError("v2 target plan claim ID is not planner-stable")
        claim_ids[claim["id"]] = expected_claim
        for dimension in claim["dimensions"]:
            anchor = dimension["anchor"]
            expected_dimension = target["id"] + ":dimension:" + _canonical_digest([
                signature, expected_claim, dimension["kind"],
                anchor["start"], anchor["end"]])[:20]
            if dimension["id"] != expected_dimension:
                raise ValueError("v2 target plan dimension ID is not planner-stable")
            dimension_ids[dimension["id"]] = expected_dimension
    for probe in probes:
        expected_probe = target["id"] + ":probe:" + _canonical_digest([
            signature, claim_ids[probe["claim_id"]], probe["kind"],
            *sorted(dimension_ids[item] for item in probe["dimension_ids"]),
        ])[:20]
        if probe["id"] != expected_probe:
            raise ValueError("v2 target plan probe ID is not planner-stable")


def _nonnegative(value, context, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value < 0 or (integer and int(value) != value)):
        raise ValueError(f"{context} must be a finite nonnegative " +
                         ("integer" if integer else "number"))
    return value


def _extension_schedule(config):
    if config.get("experiment") not in EXTENSION_EXPERIMENTS:
        raise ValueError("Unexpected target-extension experiment")
    experiment = config["experiment"]
    expected_contract = EXTENSION_EXPERIMENTS[experiment]
    if (expected_contract in VERSIONED_PLAN_SCHEMAS and
            config.get("target_plan_schema") != expected_contract):
        raise ValueError("versioned extension config must declare its matching target plan schema")
    if (expected_contract == "legacy-decision-probe-v1" and
            config.get("target_plan_schema") is not None):
        raise ValueError("legacy extension config cannot claim a v2 target plan schema")
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


def _extension_contract(config):
    """Return the artifact contract selected by the experiment identifier.

    The experiment name, rather than the shape of one convenient plan, chooses
    the contract.  This prevents a v2 run from silently downgrading to a legacy
    plan that omits program-owned matching semantics.
    """
    try:
        return EXTENSION_EXPERIMENTS[config.get("experiment")]
    except KeyError:
        raise ValueError("Unexpected target-extension experiment") from None


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
            if set(repair) not in (
                    {"structure", "extension"},
                    {"structure", "extension", "extension_output"}):
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
        return value.get(name, 0) if name else sum(value.values())
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
                ("repair_requested", "structure_repair_requested",
                 "output_repair_requested") for x in planner),
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
            "target_extension_output_repair_requests": sum(x["status"] ==
                "output_repair_requested" for x in planner),
            "target_extension_output_repair_followup_calls": sum(
                _repair_counter(x, "extension_output") > 0 for x in planner
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


def _required_probe_bindings(claim, assessment_mode, logic="single", *,
                             plan_schema=PLAN_SCHEMA_V2):
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
        if dimension in ("entity_identity", "exact_designation"):
            required.update((probe, (identifier,)) for identifier in by_kind.get(dimension, ()))
        elif dimension == "time" and plan_schema == PLAN_SCHEMA_V3:
            required.update((probe, (identifier,)) for identifier in by_kind.get(dimension, ()))
        elif dimension in by_kind:
            required.add((probe, ids(dimension)))
    if set(by_kind) & {"condition", "modality"}:
        required.add(("condition_modality", ids("condition", "modality")))
    if plan_schema in VERSIONED_PLAN_SCHEMAS and by_kind.get("exact_designation"):
        required.add(("designation_relation", tuple(sorted(dimension_ids))))
    relation_probe = {"conditional": "conditional_relation",
                      "comparison": "comparison_relation",
                      "causal": "causal_relation"}.get(logic)
    if relation_probe is not None:
        required.add((relation_probe, tuple(sorted(dimension_ids))))
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


def _audit_anchor(anchor, text, context, parent=None):
    if not isinstance(anchor, dict) or set(anchor) != ANCHOR_FIELDS:
        raise ValueError(context + " has an invalid anchor")
    start, end, quote = anchor["start"], anchor["end"], anchor["quote"]
    if (type(start) is not int or type(end) is not int or not isinstance(quote, str) or
            not 0 <= start < end <= len(text) or text[start:end] != quote):
        raise ValueError(context + " anchor does not match the immutable target")
    if parent is not None and not (parent["start"] <= start < end <= parent["end"]):
        raise ValueError(context + " anchor escapes its claim")


def _audit_v2_claims(claims, target, plan_schema=PLAN_SCHEMA_V2):
    """Independently recheck the target contract retained in a v2 artifact."""
    text = target.get("text")
    if not isinstance(text, str):
        raise ValueError("immutable target text is invalid")
    claim_ids, dimension_ids = set(), set()
    attributions, attributed_content = [], []
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != CLAIM_FIELDS:
            raise ValueError("v2 target plan has malformed claims")
        identifier = claim["id"]
        if not isinstance(identifier, str) or not identifier or identifier in claim_ids:
            raise ValueError("v2 target plan has duplicate or empty claim ids")
        claim_ids.add(identifier)
        if (not isinstance(claim["statement"], str) or not claim["statement"].strip() or
                claim["role"] not in CLAIM_ROLES):
            raise ValueError("v2 target plan has an invalid claim statement or role")
        anchor = claim["anchor"]
        _audit_anchor(anchor, text, "claim")
        dimensions = claim["dimensions"]
        if not isinstance(dimensions, list) or len(dimensions) > 9:
            raise ValueError("v2 target plan claim dimensions must be an array")
        by_kind = {}
        for dimension in dimensions:
            if not isinstance(dimension, dict) or set(dimension) != DIMENSION_FIELDS:
                raise ValueError("v2 target plan has malformed dimensions")
            dimension_id, kind = dimension["id"], dimension["kind"]
            if (not isinstance(dimension_id, str) or not dimension_id or
                    dimension_id in dimension_ids or kind not in DIMENSION_KINDS):
                raise ValueError("v2 target plan has a duplicate or invalid dimension")
            dimension_ids.add(dimension_id)
            _audit_anchor(dimension["anchor"], text, "dimension", parent=anchor)
            by_kind.setdefault(kind, []).append(dimension["anchor"])
        if len(by_kind.get("subject", ())) != 1 or len(by_kind.get("predicate", ())) != 1:
            raise ValueError("v2 target plan claim must have exactly one subject and predicate")
        repeatable = {"entity_identity"}
        if plan_schema == PLAN_SCHEMA_V3:
            repeatable.add("time")
        repeated = [kind for kind, values in by_kind.items()
                    if kind not in repeatable and len(values) > 1]
        if repeated:
            raise ValueError("v2 target plan repeats a non-identity dimension kind")
        if plan_schema == PLAN_SCHEMA_V3:
            times = by_kind.get("time", ())
            for index, left in enumerate(times):
                for right in times[index + 1:]:
                    if max(left["start"], right["start"]) < min(left["end"], right["end"]):
                        raise ValueError("v3 target plan contains overlapping time spans")
        identity = [item for kind in ("entity_identity", "exact_designation")
                    for item in by_kind.get(kind, ())]
        for index, left in enumerate(identity):
            for right in identity[index + 1:]:
                if max(left["start"], right["start"]) < min(left["end"], right["end"]):
                    raise ValueError("v2 target plan contains overlapping identity spans")
        if by_kind.get("exact_designation") and not DESIGNATION_CUE.search(
                by_kind["predicate"][0]["quote"]):
            raise ValueError("v2 target plan exact designation lacks a naming predicate")
        if claim["role"] == "attribution":
            attributions.append(claim)
            cue = ATTRIBUTION_CUE.search(anchor["quote"])
            if cue:
                boundary = anchor["start"] + cue.end()
                if any(item["end"] > boundary for values in by_kind.values()
                       for item in values):
                    raise ValueError("v2 attribution dimensions cross into attributed content")
        elif claim["role"] == "attributed_content":
            attributed_content.append(claim)

    # The caller validates the logic-to-claim cardinality after extracting it
    # from the canonical critic projection. Return role groups for that check.
    return claim_ids, dimension_ids, attributions, attributed_content


def _audit_v2_logic(logic, claims, attributions, attributed_content):
    if logic not in PLAN_LOGICS or logic == "mixed":
        raise ValueError("v2 target plan uses unsupported target logic")
    if logic in {"single", "conditional", "comparison", "causal"} and len(claims) != 1:
        raise ValueError("v2 relational target logic has invalid claim cardinality")
    if logic in {"and", "or"} and len(claims) < 2:
        raise ValueError("v2 Boolean target logic needs multiple claims")
    allowed_boolean_roles = ({"main", "conjunct"} if logic == "and"
                             else {"main", "alternative"})
    if logic in {"and", "or"} and any(
            claim["role"] not in allowed_boolean_roles for claim in claims):
        raise ValueError("v2 Boolean target logic contains a nested relation role")
    if logic == "attribution":
        if (len(attributions) != 1 or not attributed_content or
                len(claims) != 1 + len(attributed_content)):
            raise ValueError("v2 attribution logic lacks one parent and attributed content")
        parent_id = attributions[0]["id"]
        if any(claim["parent_claim_id"] != parent_id for claim in attributed_content):
            raise ValueError("v2 attributed content is not linked to its parent")


def _plan_contract(plan, expected=None):
    declared = plan.get("schema_version")
    if declared is None:
        actual = "legacy-decision-probe-v1"
    elif declared in VERSIONED_PLAN_SCHEMAS:
        actual = declared
    else:
        raise ValueError("target plan has an unknown schema version")
    if expected is not None and actual != expected:
        raise ValueError("target plan schema does not match experiment contract")
    return actual


def _grounded_span(value, materials, allowed_ids, context):
    if not isinstance(value, dict) or set(value) != {"version_id", "start", "end", "quote"}:
        raise ValueError(context + " must be a normal source span")
    version_id, start, end, quote = (value["version_id"], value["start"],
                                     value["end"], value["quote"])
    if (not isinstance(version_id, str) or not version_id or type(start) is not int or
            type(end) is not int or start < 0 or end <= start or not isinstance(quote, str) or
            not quote):
        raise ValueError(context + " has invalid source coordinates")
    if version_id not in materials:
        raise ValueError(context + " references an unknown material")
    if allowed_ids is not None and version_id not in allowed_ids:
        raise ValueError(context + " is outside the stage evidence scope")
    content = materials[version_id]
    if end > len(content) or content[start:end] != quote:
        raise ValueError(context + " is not grounded in material.content")


def _audit_v3_terminal_origins(report, target):
    """Recompute the exact final roots from raw candidates and the eligible graph.

    The runner retains model-proposed origins in each current analysis, then
    projects only reachable terminal candidates into ``report["origins"]``.
    Requiring merely that the exported roots do not reach each other misses an
    omitted terminal candidate: an intermediate-only export can look terminal
    when the real terminal candidate has been removed from the export.  This
    audit therefore starts again from the retained raw candidate declarations.
    """
    if not isinstance(report, dict):
        raise ValueError("v3 terminal-origin audit requires the final report")
    origins, relations = report.get("origins"), report.get("relations")
    materials = report.get("materials")
    eligible = report.get("eligible_version_ids")
    analyses = report.get("analyses")
    if (not isinstance(origins, list) or not isinstance(relations, list) or
            not isinstance(materials, list) or not isinstance(eligible, list) or
            not isinstance(analyses, dict)):
        raise ValueError(
            "v3 final report must retain origins, relations, materials, eligible versions and analyses")

    material_ids = []
    for material in materials:
        version_id = material.get("version_id") if isinstance(material, dict) else None
        if not isinstance(version_id, str) or not version_id:
            raise ValueError("v3 final report contains an invalid material version")
        material_ids.append(version_id)
    if len(material_ids) != len(set(material_ids)):
        raise ValueError("v3 final report contains duplicate material versions")
    if (any(not isinstance(version_id, str) or not version_id for version_id in eligible) or
            len(eligible) != len(set(eligible)) or not set(eligible) <= set(material_ids)):
        raise ValueError("v3 final report contains invalid eligible version ids")
    eligible_ids = set(eligible)

    # ``current_analyses`` contains exactly one accepted latest analysis for
    # every eligible version.  Ineligible archive decompositions live only in
    # analysis_history and must never create a candidate root.
    if (any(not isinstance(version_id, str) or not version_id for version_id in analyses) or
            set(analyses) != eligible_ids):
        raise ValueError("v3 current analyses do not exactly match eligible versions")
    candidate_ids = set()
    for owner, analysis in analyses.items():
        if not isinstance(analysis, dict) or not isinstance(analysis.get("origins"), list):
            raise ValueError("v3 current analysis has no valid origin array")
        for candidate in analysis["origins"]:
            if not isinstance(candidate, dict):
                raise ValueError("v3 current analysis contains an invalid origin candidate")
            target_id, version_id = candidate.get("target_id"), candidate.get("version_id")
            if (target_id != target.get("id") or not isinstance(version_id, str) or
                    not version_id or version_id not in eligible_ids):
                raise ValueError("v3 current analysis contains an ineligible origin candidate")
            candidate_ids.add(version_id)

    root_ids = []
    for origin in origins:
        if not isinstance(origin, dict):
            raise ValueError("v3 final report contains an invalid origin")
        target_id, version_id = origin.get("target_id"), origin.get("version_id")
        if (target_id != target.get("id") or not isinstance(version_id, str) or
                not version_id or version_id not in eligible_ids):
            raise ValueError("v3 final report contains an invalid origin")
        root_ids.append(version_id)
    if len(root_ids) != len(set(root_ids)):
        raise ValueError("v3 final report contains duplicate origins")
    root_ids = set(root_ids)

    adjacency = {}
    direct_edges = 0
    for relation in relations:
        if not isinstance(relation, dict):
            raise ValueError("v3 final report contains an invalid relation")
        if (relation.get("status") != "direct" or
                relation.get("kind") not in DIRECT_LINEAGE_KINDS):
            continue
        source, upstream = relation.get("from_version"), relation.get("to_version")
        if (not isinstance(source, str) or not source or
                not isinstance(upstream, str) or not upstream):
            raise ValueError("v3 final report contains an invalid direct lineage edge")
        # This is the scorer-side equivalent of passing ``eligible.values()``
        # into the runtime graph projection: exact version IDs outside that set
        # do not participate in reachability.
        if source in eligible_ids and upstream in eligible_ids:
            adjacency.setdefault(source, set()).add(upstream)
            direct_edges += 1
    source_version_id = target.get("source_version_id") if isinstance(target, dict) else None
    if not isinstance(source_version_id, str) or not source_version_id:
        raise ValueError("v3 terminal-origin audit requires the target source version")

    def reached_from(start):
        if start not in eligible_ids:
            return set()
        reached, pending = set(), [start]
        while pending:
            version_id = pending.pop()
            if version_id in reached:
                continue
            reached.add(version_id)
            pending.extend(adjacency.get(version_id, ()))
        return reached

    reachable = reached_from(source_version_id)
    reachable_candidates = candidate_ids & reachable
    downstream = {candidate: reached_from(candidate)
                  for candidate in reachable_candidates}
    terminal_candidates = {candidate for candidate, reached in downstream.items()
        if not ((reached - {candidate}) & reachable_candidates)}
    if root_ids != terminal_candidates:
        raise ValueError(
            "v3 final origins violate the terminal-root exact-set invariant")

    # Mirror the runtime's second return value as well as its terminal set.  A
    # disconnected raw candidate, or a reachable candidate component with no
    # path to any terminal candidate (for example a candidate cycle), is not a
    # clean empty-root result: the runner must retain an active lineage gap.
    disconnected = len(reachable_candidates) != len(candidate_ids)
    nonterminating = any(not (reached & terminal_candidates)
                         for reached in downstream.values())
    incomplete = disconnected or nonterminating
    gaps = report.get("gaps")
    if not isinstance(gaps, list):
        raise ValueError("v3 final report must retain an active gap array")
    lineage_id = "lineage:" + target["id"]
    lineage_gaps = [gap for gap in gaps
                    if isinstance(gap, dict) and gap.get("id") == lineage_id]
    if incomplete and (len(lineage_gaps) != 1 or
            lineage_gaps[0].get("stage") != "provenance" or
            lineage_gaps[0].get("blocking") is not True):
        raise ValueError(
            "v3 incomplete origin chain lacks its active blocking lineage gap")
    return {"applicable": True, "origin_count": len(root_ids),
            "direct_lineage_edges": direct_edges,
            "reachable_origin_count": len(root_ids), "terminal_roots": True}


def _probe_result_audit(plan, verification, report_history, materials, target,
                        required=False):
    """Audit persisted per-probe answers, separately from structural routing.

    ``verification`` is the verifier call ledger. ``report_history`` is the
    final provenance report's VerificationResult history.  Legacy v1 reports
    did not persist per-probe answers, so absence is represented as unavailable
    rather than as zero coverage.
    """
    probes = {item["id"]: item for item in plan["probes"]}
    expected_by_stage = {stage: {identifier for identifier, probe in probes.items()
        if stage in probe["routes"]} for stage in ("evidence", "world")}
    accepted_rounds = []
    for item in verification:
        if item.get("stage") != "judgement_critic" or item.get("status") != "accepted":
            continue
        round_number = item.get("round")
        if type(round_number) is not int or round_number <= 0:
            if required:
                raise ValueError("accepted v2 judgement cycle has no valid round")
            continue
        accepted_rounds.append(round_number)
    if len(accepted_rounds) != len(set(accepted_rounds)):
        raise ValueError("duplicate accepted judgement cycle")
    accepted_rounds = sorted(accepted_rounds)
    expected_by_stage_total = {stage: len(ids) * len(accepted_rounds)
                               for stage, ids in expected_by_stage.items()}
    expected_total = sum(expected_by_stage_total.values())

    history = report_history if isinstance(report_history, list) else []
    result_fields = ("evidence_probe_results", "world_probe_results")
    presence = [any(field in item for field in result_fields) for item in history
                if isinstance(item, dict)]
    available = bool(presence) and all(presence)
    if any(presence) and not all(presence):
        raise ValueError("per-probe result history is only partially retained")
    if required and not available:
        raise ValueError("completed v2 case has no retained per-probe results")
    empty = {
        "available": False,
        "accepted_judgement_cycles": len(accepted_rounds),
        "audited_judgement_cycles": 0,
        "expected_result_slots": expected_total,
        "result_slots": None,
        "result_slot_coverage": None,
        "results_with_basis": None,
        "conclusive_results": None,
        "grounded_conclusive_results": None,
        "grounded_conclusive_rate": None,
        "basis_spans": None,
        "status_counts": {},
        "by_stage": {stage: {"expected_result_slots": count,
                             "result_slots": None,
                             "result_slot_coverage": None}
                     for stage, count in expected_by_stage_total.items()},
    }
    if not available:
        return empty
    if materials is None:
        raise ValueError("materials are required to audit per-probe basis")
    material_content = {}
    for item in materials:
        if (not isinstance(item, dict) or not isinstance(item.get("version_id"), str) or
                not isinstance(item.get("content"), str) or
                item["version_id"] in material_content):
            raise ValueError("case materials are invalid for per-probe audit")
        material_content[item["version_id"]] = item["content"]
    evidence_scope = target.get("evidence_scope")
    if not isinstance(evidence_scope, list) or any(not isinstance(x, str) for x in evidence_scope):
        raise ValueError("target evidence_scope is invalid for per-probe audit")
    allowed_by_stage = {
        "evidence": set(evidence_scope) if evidence_scope else set(material_content),
        "world": set(material_content),
    }

    by_round = {}
    for item in history:
        if not isinstance(item, dict) or type(item.get("round")) is not int or item["round"] <= 0:
            raise ValueError("final verification history has an invalid round")
        if item["round"] in by_round:
            raise ValueError("final verification history has duplicate rounds")
        by_round[item["round"]] = item
    if set(by_round) != set(accepted_rounds):
        raise ValueError("per-probe result rounds do not match accepted judgement cycles")

    counters = Counter()
    stage_counters = {stage: Counter() for stage in expected_by_stage}
    for round_number in accepted_rounds:
        record = by_round[round_number]
        for stage in ("evidence", "world"):
            key = stage + "_probe_results"
            results = record.get(key)
            if not isinstance(results, list):
                raise ValueError(key + " must be an array")
            seen = set()
            for index, result in enumerate(results):
                context = f"round {round_number} {stage} probe result {index}"
                if not isinstance(result, dict) or set(result) != {
                        "probe_id", "claim_id", "stage", "status", "basis",
                        "rationale", "referent_relation"}:
                    raise ValueError(context + " has invalid fields")
                probe_id = result["probe_id"]
                if not isinstance(probe_id, str) or probe_id in seen:
                    raise ValueError(context + " has a missing or duplicate probe id")
                seen.add(probe_id)
                if probe_id not in expected_by_stage[stage]:
                    raise ValueError(context + " is not projected to this stage")
                probe = probes[probe_id]
                if result["claim_id"] != probe["claim_id"] or result["stage"] != stage:
                    raise ValueError(context + " disagrees with its program-owned binding")
                if result["status"] not in PROBE_STATUSES:
                    raise ValueError(context + " has an invalid status")
                if (not isinstance(result["rationale"], str) or not result["rationale"].strip() or
                        result["referent_relation"] not in REFERENT_RELATIONS):
                    raise ValueError(context + " has invalid rationale or referent relation")
                basis = result["basis"]
                if not isinstance(basis, list):
                    raise ValueError(context + " basis must be an array")
                if result["status"] != "unresolved" and not basis:
                    raise ValueError(context + " conclusive status requires source basis")
                if result["status"] == "conflicting" and (len(basis) < 2 or
                        len({_canonical_digest(item) for item in basis}) < 2):
                    raise ValueError(context + " conflicting status requires two distinct source passages")
                policy = probe.get("match_policy",
                    MATCH_POLICIES.get(probe.get("kind"), "semantic_constraint"))
                relation = result["referent_relation"]
                if policy == "semantic_constraint":
                    if relation != "not_applicable":
                        raise ValueError(context + " semantic constraint has a referent relation")
                elif policy in ("same_referent", "exact_designation"):
                    permitted = {
                        "supported": ({"exact", "alias", "description", "anaphora"}
                                      if policy == "same_referent" else {"exact"}),
                        "contradicted": {"different"},
                        "conflicting": {"ambiguous"},
                        "unresolved": {"ambiguous", "unresolved"},
                    }[result["status"]]
                    if relation not in permitted:
                        raise ValueError(context + " status conflicts with its match policy relation")
                else:
                    raise ValueError(context + " has an unknown match policy")
                for span_index, span in enumerate(basis):
                    _grounded_span(span, material_content, allowed_by_stage[stage],
                                   context + f" basis {span_index}")
                if (probe.get("kind") == "designation_relation" and
                        result["status"] != "unresolved" and
                        not _has_designation_basis(
                            basis, _designation_labels(plan, probe["claim_id"]))):
                    raise ValueError(
                        context + " needs one naming-predicate basis span containing "
                        "every asserted designation"
                    )
                counters["result_slots"] += 1
                counters["basis_spans"] += len(basis)
                counters["results_with_basis"] += bool(basis)
                counters["conclusive_results"] += result["status"] != "unresolved"
                counters["grounded_conclusive_results"] += (
                    result["status"] != "unresolved" and bool(basis))
                counters["status_" + result["status"]] += 1
                stage_counters[stage]["result_slots"] += 1
            if seen != expected_by_stage[stage]:
                raise ValueError(f"round {round_number} {stage} does not have exactly one "
                                 "result per projected probe")

    by_stage = {}
    for stage, expected in expected_by_stage_total.items():
        actual = stage_counters[stage]["result_slots"]
        by_stage[stage] = {"expected_result_slots": expected,
                           "result_slots": actual,
                           "result_slot_coverage": _ratio(actual, expected)}
    conclusive = counters["conclusive_results"]
    return {
        "available": True,
        "accepted_judgement_cycles": len(accepted_rounds),
        "audited_judgement_cycles": len(by_round),
        "expected_result_slots": expected_total,
        "result_slots": counters["result_slots"],
        "result_slot_coverage": _ratio(counters["result_slots"], expected_total),
        "results_with_basis": counters["results_with_basis"],
        "conclusive_results": conclusive,
        "grounded_conclusive_results": counters["grounded_conclusive_results"],
        "grounded_conclusive_rate": _ratio(counters["grounded_conclusive_results"], conclusive),
        "basis_spans": counters["basis_spans"],
        "status_counts": {status: counters["status_" + status]
                          for status in sorted(PROBE_STATUSES)},
        "by_stage": by_stage,
    }


def _audit_plan(path, target, row, psi, verification, expected_contract=None,
                report_history=None, materials=None, report=None):
    plan = _read(path)
    if not isinstance(plan, dict):
        raise ValueError("target plan must be an object")
    contract = _plan_contract(plan, expected_contract)
    target_signature = _canonical_digest(target)
    projections = plan.get("projections")
    if not isinstance(projections, dict) or set(projections) != set(PLAN_STAGES):
        raise ValueError("target plan must retain every stage projection")
    if contract in VERSIONED_PLAN_SCHEMAS:
        if set(plan) != PLAN_FIELDS_V2:
            raise ValueError("v2 target plan has missing or unexpected fields")
        critic = projections["critic"]
        if not isinstance(critic, dict) or set(critic) != PROJECTION_FIELDS_V2:
            raise ValueError("v2 critic projection has missing or unexpected fields")
        if (critic.get("schema_version") != contract or
                critic.get("stage") != "critic" or
                critic.get("target_signature") != target_signature or
                not isinstance(critic.get("notes"), str)):
            raise ValueError("v2 critic projection metadata mismatch")
        canonical = {key: critic[key] for key in
            ("schema_version", "target_signature", "logic", "claims", "probes", "notes")}
        expected_sha = _canonical_digest(canonical)
        if critic["plan_sha256"] != expected_sha or plan["sha256"] != expected_sha:
            raise ValueError("target plan checksum differs from canonical critic projection")
        if any(plan[key] != canonical[key] for key in canonical):
            raise ValueError("top-level target plan differs from canonical critic projection")
    else:
        required_top = {"target_signature", "logic", "claims", "probes", "notes",
                        "sha256", "projections"}
        if not required_top <= set(plan):
            raise ValueError("target plan is missing required fields")
        payload = {key: plan[key] for key in
                   ("target_signature", "logic", "claims", "probes", "notes")}
        if plan["sha256"] != _canonical_digest(payload):
            raise ValueError("target plan checksum mismatch")
    if plan["target_signature"] != target_signature:
        raise ValueError("target plan signature differs from immutable target")
    if row.get("target_plan_sha256") != plan["sha256"]:
        raise ValueError("result row target plan checksum mismatch")
    claims, probes = plan["claims"], plan["probes"]
    if contract in VERSIONED_PLAN_SCHEMAS and not (
            isinstance(claims, list) and isinstance(probes, list) and
            1 <= len(claims) <= 4 and 1 <= len(probes) <= 44):
        raise ValueError("v2 target plan exceeds its planner schema bounds")
    claim_ids, probe_ids = _ids(claims, "target claims"), _ids(probes, "decision probes")
    if contract in VERSIONED_PLAN_SCHEMAS:
        _, _, attributions, attributed_content = _audit_v2_claims(
            claims, target, plan_schema=contract)
        _audit_v2_logic(plan["logic"], claims, attributions, attributed_content)
    known_claims = set(claim_ids)
    claims_by_id = {item["id"]: item for item in claims}
    by_id = {}
    actual_bindings = set()
    routing_contract = (V2_PROBE_ROUTES_AND_GATES if contract in VERSIONED_PLAN_SCHEMAS
                        else LEGACY_PROBE_ROUTES_AND_GATES)
    for probe in probes:
        if contract in VERSIONED_PLAN_SCHEMAS and (not isinstance(probe, dict) or
                                                   set(probe) != PROBE_FIELDS_V2):
            raise ValueError("v2 decision probe has missing or unexpected fields")
        kind = probe.get("kind")
        if probe.get("claim_id") not in known_claims or kind not in routing_contract:
            raise ValueError("decision probe references an unknown claim or kind")
        if (not isinstance(probe.get("question"), str) or not probe["question"].strip() or
                not isinstance(probe.get("decision_impact"), str) or
                not probe["decision_impact"].strip()):
            raise ValueError("decision probe has an empty question or impact")
        dimension_ids = probe.get("dimension_ids")
        if (not isinstance(dimension_ids, list) or
                any(not isinstance(item, str) or not item for item in dimension_ids) or
                len(dimension_ids) != len(set(dimension_ids))):
            raise ValueError("decision probe has invalid dimension bindings")
        owned = {item["id"] for item in claims_by_id[probe["claim_id"]].get("dimensions", [])}
        if not set(dimension_ids) <= owned:
            raise ValueError("decision probe binds a dimension outside its claim")
        expected_routes, expected_gate = routing_contract[kind]
        if probe.get("routes") != list(expected_routes) or probe.get("gate") != expected_gate:
            raise ValueError("decision probe differs from program-owned routes or gate")
        expected_policy = MATCH_POLICIES.get(kind, "semantic_constraint")
        if contract in VERSIONED_PLAN_SCHEMAS and probe.get("match_policy") != expected_policy:
            raise ValueError("decision probe has invalid program-owned match policy")
        if contract == PLAN_SCHEMA_V3 and kind in MATCH_POLICIES:
            if len(dimension_ids) != 1:
                raise ValueError("identity probe must bind exactly one dimension")
            dimensions_by_id = {item["id"]: item for item in
                                claims_by_id[probe["claim_id"]]["dimensions"]}
            expected_question = _canonical_identity_question(
                dimensions_by_id[dimension_ids[0]]["anchor"], expected_policy)
            if probe["question"] != expected_question:
                raise ValueError("identity probe question is not program-owned canonical text")
        if (contract not in VERSIONED_PLAN_SCHEMAS and "match_policy" in probe and
                probe["match_policy"] != expected_policy):
            raise ValueError("legacy decision probe has an invalid optional match policy")
        by_id[probe["id"]] = probe
        actual_bindings.add((probe["claim_id"], kind,
                             tuple(sorted(dimension_ids))))
    required_bindings = {(claim["id"], kind, dimension_ids) for claim in claims
                         for kind, dimension_ids in _required_probe_bindings(
                             claim, target["assessment_mode"],
                             plan["logic"] if contract in VERSIONED_PLAN_SCHEMAS else "single",
                             plan_schema=contract)}
    if actual_bindings - required_bindings:
        raise ValueError("decision probe binding does not match target dimensions")
    if contract in VERSIONED_PLAN_SCHEMAS and actual_bindings != required_bindings:
        raise ValueError("v2 target plan does not exactly cover required probe bindings")
    if contract in VERSIONED_PLAN_SCHEMAS:
        _audit_v2_stable_ids(target, target_signature, claims, probes)
    covered_bindings = required_bindings & actual_bindings

    projection_slots = covered_projection_slots = 0
    projected_by_stage = {}
    for stage in PLAN_STAGES:
        view = projections[stage]
        if (not isinstance(view, dict) or view.get("target_signature") != plan["target_signature"] or
                view.get("plan_sha256") != plan["sha256"] or view.get("stage") != stage):
            raise ValueError("target plan projection metadata mismatch")
        if contract in VERSIONED_PLAN_SCHEMAS and (set(view) != PROJECTION_FIELDS_V2 or
                view.get("schema_version") != contract or
                view.get("logic") != plan["logic"] or view.get("notes") != plan["notes"]):
            raise ValueError("v2 target plan projection contract mismatch")
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
        expected_probes = probes if stage == "critic" else [item for item in probes
                                                             if stage in item["routes"]]
        expected_claims = claims if stage == "critic" else [item for item in claims
            if item["id"] in {probe["claim_id"] for probe in expected_probes}]
        if view["probes"] != expected_probes or view["claims"] != expected_claims:
            raise ValueError("target plan projection content differs from frozen plan")
        projected_by_stage[stage] = len(expected_probe_ids)

    observed = ({item["stage"] for item in psi} |
                {item["stage"] for item in verification})
    at_executed_stage = sum(count for stage, count in projected_by_stage.items()
                            if stage != "critic" and stage in observed)
    routed_total = sum(projected_by_stage[stage] for stage in ROUTED_STAGES)
    terminal_origin_audit = (_audit_v3_terminal_origins(report, target)
                             if contract == PLAN_SCHEMA_V3 and report is not None else None)
    probe_results = _probe_result_audit(plan, verification, report_history, materials, target,
        required=(contract in VERSIONED_PLAN_SCHEMAS and row.get("status") == "completed"))
    return {
        "plan_contract": contract,
        "claims": len(claims),
        "dimensions": sum(len(item.get("dimensions", [])) for item in claims),
        "probes": len(probes),
        "required_probe_slots": len(required_bindings),
        "required_probe_slots_covered": len(covered_bindings),
        "projection_slots": projection_slots,
        "projection_slots_covered": covered_projection_slots,
        "routed_stage_slots": routed_total,
        "projected_slots_at_executed_stages": at_executed_stage,
        "projected_probes_by_stage": projected_by_stage,
        "terminal_origin_audit": terminal_origin_audit,
        "probe_result_audit": probe_results,
    }


def _plan_coverage(run, ids, inputs, rows, config=None):
    by_id = {row["id"]: row for row in rows}
    totals = Counter()
    result_totals = Counter()
    result_statuses = Counter()
    result_stages = {stage: Counter() for stage in ("evidence", "world")}
    cases = []
    expected_contract = _extension_contract(config) if config is not None else None
    for case_id in ids:
        row = by_id[case_id]
        path = run / f"{case_id}-target-plan.json"
        planner = _history(run, case_id, "target-planner-history")
        psi = _history(run, case_id, "psi-history")
        verification = _history(run, case_id, "verification-history")
        item = {"id": case_id, "status": row["status"], "plan_present": path.exists(),
                "plan_valid": False}
        if path.exists():
            report_path = run / f"{case_id}-report.json"
            report = None
            report_history = None
            if report_path.exists():
                report = _read(report_path)
                if not isinstance(report, dict):
                    raise ValueError("case report must be an object")
                report_history = report.get("verification_history")
                if report_history is not None and not isinstance(report_history, list):
                    raise ValueError("final verification_history must be an array")
            audit = _audit_plan(path, inputs[case_id]["target"], row, psi, verification,
                expected_contract=expected_contract, report_history=report_history,
                materials=inputs[case_id].get("materials"), report=report)
            item.update(plan_valid=True, **audit)
            totals.update({key: value for key, value in audit.items()
                           if isinstance(value, int) and not isinstance(value, bool)})
            for stage, count in audit["projected_probes_by_stage"].items():
                totals["projected_probes_" + stage] += count
            result = audit["probe_result_audit"]
            result_totals["cases_with_results"] += result["available"]
            for key in ("accepted_judgement_cycles", "audited_judgement_cycles",
                        "expected_result_slots"):
                result_totals[key] += result[key]
            if result["available"]:
                for key in ("result_slots", "results_with_basis", "conclusive_results",
                            "grounded_conclusive_results", "basis_spans"):
                    result_totals[key] += result[key]
                result_statuses.update(result["status_counts"])
                for stage, values in result["by_stage"].items():
                    result_stages[stage]["result_slots"] += values["result_slots"]
            for stage, values in result["by_stage"].items():
                result_stages[stage]["expected_result_slots"] += values["expected_result_slots"]
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
            ("projected_slots_at_executed_stages", "routed_stage_slots",
             "executed_stage_projection_coverage")):
        totals[name] = _ratio(totals[numerator], totals[denominator])
    any_results = result_totals["cases_with_results"] > 0
    result_summary = {
        **{key: result_totals[key] for key in (
            "cases_with_results", "accepted_judgement_cycles",
            "audited_judgement_cycles", "expected_result_slots", "result_slots",
            "results_with_basis", "conclusive_results",
            "grounded_conclusive_results", "basis_spans")},
        "scheduled_cases": len(cases),
        "result_slot_coverage": (_ratio(result_totals["result_slots"],
            result_totals["expected_result_slots"]) if any_results else None),
        "grounded_conclusive_rate": (_ratio(result_totals["grounded_conclusive_results"],
            result_totals["conclusive_results"]) if any_results else None),
        "status_counts": {status: result_statuses[status]
                          for status in sorted(PROBE_STATUSES)} if any_results else {},
        "by_stage": {stage: {
            "expected_result_slots": values["expected_result_slots"],
            "result_slots": values["result_slots"] if any_results else None,
            "result_slot_coverage": (_ratio(values["result_slots"],
                values["expected_result_slots"]) if any_results else None),
        } for stage, values in result_stages.items()},
    }
    structural_cases, result_cases = [], []
    for item in cases:
        structural = dict(item)
        result = structural.pop("probe_result_audit", None)
        structural_cases.append(structural)
        if result is not None:
            result_cases.append({"id": item["id"], "status": item["status"], **result})
    return {"totals": dict(totals), "cases": structural_cases,
            "probe_results": {"totals": result_summary, "cases": result_cases,
                "meaning": "Persisted evidence/world answers only. Coverage requires exactly one grounded, scope-valid result per projected probe in every accepted judgement cycle; grounded-conclusive is descriptive and is not accuracy."},
            "meaning": "Structural checklist, deterministic projection, and executed-stage presence only; it is not per-probe answer coverage or semantic correctness.",
            }


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
        [raw_by_id["extension"][case_id] for case_id in selected_ids],
        configs["extension"])
    probe_results = coverage.pop("probe_results")
    label_pairs = {}
    for baseline, candidate in (("original", "staged"), ("original", "extension"),
                                ("staged", "extension")):
        name = candidate + "_vs_" + baseline
        label_pairs[name] = {"all_scheduled": _paired(cases, matrix, baseline, candidate),
                             "shared_completed": _paired(cases, matrix, baseline, candidate, True)}

    return {
        "schema_version": "target-extension-compare-score-v2",
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
        "target_probe_result_audit": probe_results,
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
            "Target-plan coverage is structural checklist/projection coverage only. Persisted per-probe answers are audited separately and neither measure establishes label accuracy.",
            "A grounded conclusive probe result means it has at least one scope-valid exact source span; it does not by itself establish source independence, correctness, or sufficient evidence.",
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
