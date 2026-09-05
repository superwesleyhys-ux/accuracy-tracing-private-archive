"""Artifact-only comparison of original, staged, and target-extended runs.

The extension run selects the comparison denominator.  Baseline runs may contain
additional cases, but the scorer compares only identical selected target/material
contracts.  No inference adapter, credential, or network service is imported.

Plan coverage is structural: it shows that required checklist items exist in a
checksummed plan and that their projected stages ran.  V2+ reports are audited
separately for one grounded result per projected evidence/world probe in every
accepted judgement cycle.  V4 additionally recomputes the target-segment ledger,
material probe ledgers, strict follow-ups, and task-to-return attribution from raw
artifacts.  None of these audits is evidence that the answer is correct; all
remain separate from labels and provenance accuracy.
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
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from proof_score import USAGE_FIELDS, _arm_score, _label_evaluable, _ratio
from loop_receipt_artifact_score import audit_loop_receipt_artifacts
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
PLAN_SCHEMA_V4 = "decision-probe-v4"
V4_FIXED_EXPERIMENT = "target_extended_psi_development_v4_fixed_reanalysis"
V4_TASK_ROUTED_EXPERIMENT = "target_extended_psi_development_v4_task_routed"
TASK_ROUTED_PROVIDER = "task-routed fixed eligible snapshots, no open-web collection"
VERSIONED_PLAN_SCHEMAS = {PLAN_SCHEMA_V2, PLAN_SCHEMA_V3, PLAN_SCHEMA_V4}
EXTENSION_EXPERIMENTS = {
    "target_extended_psi_development_v1": "legacy-decision-probe-v1",
    "target_extended_psi_development_v2": PLAN_SCHEMA_V2,
    "target_extended_psi_development_v3": PLAN_SCHEMA_V3,
    V4_FIXED_EXPERIMENT: PLAN_SCHEMA_V4,
    V4_TASK_ROUTED_EXPERIMENT: PLAN_SCHEMA_V4,
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

V4_REPEATABLE_DIMENSIONS = {
    "entity_identity", "actor_role", "quantity_unit", "time", "location",
    "baseline_scope", "condition", "modality",
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

V4_PROBE_ROUTES_AND_GATES = {
    **{kind: (("atoms", "evidence", "world"), "always") for kind in (
        "predicate_core", "claim_composition", "polarity", "time_boundary",
        "quantity_unit", "location", "baseline_scope", "condition_modality",
        "actor_role", "entity_identity", "exact_designation",
        "designation_relation", "attribution_relation", "conditional_relation",
        "comparison_relation", "causal_relation")},
    "source_lineage": (("lineage", "world"), "provenance"),
    "source_independence": (("lineage", "world"), "positive_world_only"),
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
PLAN_FIELDS_V4 = PLAN_FIELDS_V2 | {"coverage_ledger"}
PROJECTION_FIELDS_V4 = PROJECTION_FIELDS_V2 | {"coverage_ledger"}
CLAIM_FIELDS = {"id", "statement", "anchor", "role", "dimensions",
                "parent_claim_id"}
DIMENSION_FIELDS = {"id", "kind", "anchor"}
ANCHOR_FIELDS = {"start", "end", "quote"}
PROBE_FIELDS_V2 = {"id", "claim_id", "kind", "dimension_ids", "question",
                   "decision_impact", "match_policy", "routes", "gate"}
RETRIEVAL_TASK_FIELDS = {"id", "question", "stage", "dimension", "blocking",
                         "target_id", "basis", "decision_impact", "action",
                         "locator", "probe_id"}
CLAIM_ROLES = {"main", "conjunct", "alternative", "condition", "exception",
               "comparison", "cause", "effect", "attribution",
               "attributed_content"}
DIMENSION_KINDS = {"subject", "predicate", "quantity_unit", "time",
                   "baseline_scope", "negation", "condition", "modality",
                   "entity_identity", "exact_designation"}
DIMENSION_KINDS_V4 = DIMENSION_KINDS | {"actor_role", "location"}
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


def _canonical_probe_question(claim, kind, dimension_ids, owned_dimensions,
                              known_claims):
    """Independently derive the v4 question from frozen offsets and bindings."""
    bound = sorted((owned_dimensions[identifier] for identifier in dimension_ids),
                   key=lambda item: (item["anchor"]["start"], item["anchor"]["end"],
                                     item["kind"], item["id"]))
    anchors = [json.dumps(item["anchor"]["quote"], ensure_ascii=False) for item in bound]
    claim_ref = f'target[{claim["anchor"]["start"]}:{claim["anchor"]["end"]}]'
    if claim["parent_claim_id"] is not None:
        parent = known_claims[claim["parent_claim_id"]]
        claim_ref += (" as content attributed by target[" +
                      str(parent["anchor"]["start"]) + ":" +
                      str(parent["anchor"]["end"]) + "]")

    if kind == "entity_identity":
        return _canonical_identity_question(bound[0]["anchor"], "same_referent")
    if kind == "exact_designation":
        return _canonical_identity_question(bound[0]["anchor"], "exact_designation")
    if kind == "predicate_core":
        predicate = json.dumps(bound[0]["anchor"]["quote"], ensure_ascii=False)
        return "Does the evidence establish predicate " + predicate + " for " + claim_ref + "?"
    if kind == "semantic_core":
        subject = next(json.dumps(item["anchor"]["quote"], ensure_ascii=False)
                       for item in bound if item["kind"] == "subject")
        predicate = next(json.dumps(item["anchor"]["quote"], ensure_ascii=False)
                         for item in bound if item["kind"] == "predicate")
        return ("Does the evidence establish " + claim_ref + " with subject " + subject +
                " and predicate " + predicate + " in the asserted relation?")
    if kind == "claim_composition":
        return ("Does the evidence establish every anchored dimension of " + claim_ref +
                " together as the single asserted claim?")
    if kind == "polarity":
        return ("Does the evidence preserve the polarity asserted by " + claim_ref +
                ", including " + anchors[0] + "?")
    if kind == "time_boundary":
        return ("Does the evidence establish the independently anchored time qualifier " +
                anchors[0] + " for " + claim_ref + "?")
    if kind == "quantity_unit":
        return ("Does the evidence establish the quantity and unit " + anchors[0] +
                " together for " + claim_ref + "?")
    if kind == "location":
        return ("Does the evidence establish " + anchors[0] +
                " as the location where " + claim_ref + " applies?")
    if kind == "baseline_scope":
        return ("Does the evidence establish the comparison baseline or scope " +
                anchors[0] + " for " + claim_ref + "?")
    if kind == "condition_modality":
        return ("Does the evidence preserve the condition or modality " +
                ", ".join(anchors) + " for " + claim_ref + "?")
    if kind == "actor_role":
        return ("Does the evidence establish " + anchors[0] +
                " in the actor or agent role asserted by " + claim_ref + "?")
    relation = {
        "designation_relation": "directed naming or renaming relation",
        "attribution_relation": "directed parent-to-content attribution relation",
        "conditional_relation": "complete conditional relation",
        "comparison_relation": "complete directed comparison",
        "causal_relation": "complete directed causal relation",
    }.get(kind)
    if relation is not None:
        return ("Does the evidence establish the " + relation + " in " + claim_ref +
                " as one relation rather than separately true parts?")
    if kind == "source_lineage":
        return ("Can " + claim_ref +
                " be traced through an explicit source lineage to a terminal source?")
    if kind == "source_independence":
        return ("For world assessment, are sources supporting " + claim_ref +
                " independent rather than copied or derivative?")
    raise ValueError("unknown v4 probe kind")


def _canonical_decision_impact(kind):
    impacts = {
        "predicate_core": "Without the bound predicate, the claim cannot be supported.",
        "semantic_core": "Without the bound subject-predicate relation, the claim cannot be supported.",
        "claim_composition": "Separately matched dimensions cannot support the claim unless their composition is established.",
        "polarity": "A polarity mismatch can reverse or defeat the claim.",
        "time_boundary": "A mismatched time qualifier prevents support for the time-bounded claim.",
        "quantity_unit": "A mismatched quantity or unit prevents support for the quantified claim.",
        "location": "A mismatched location prevents support for the location-bounded claim.",
        "baseline_scope": "A mismatched baseline or scope prevents support for the comparison.",
        "condition_modality": "A missing condition or modality can change the claim's force or applicability.",
        "actor_role": "A mismatched actor or agent role prevents support for who performed the event.",
        "entity_identity": "An unresolved referent prevents support for the bound entity claim.",
        "exact_designation": "A different name, title, label or designation prevents exact-designation support.",
        "designation_relation": "Separately true naming facts cannot support the asserted directed designation relation.",
        "attribution_relation": "Content truth cannot substitute for evidence that the parent actually attributed that content.",
        "conditional_relation": "Separately true components cannot support the asserted conditional relation.",
        "comparison_relation": "Separately true components cannot support the asserted directed comparison.",
        "causal_relation": "Separately true components cannot support the asserted directed causal relation.",
        "source_lineage": "Without adequate lineage, provenance remains unresolved even if wording is similar.",
        "source_independence": "Derivative sources cannot satisfy independent corroboration for a positive world claim.",
    }
    try:
        return impacts[kind]
    except KeyError:
        raise ValueError("unknown v4 probe kind") from None


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

PLAN_NOTES_V4 = ("Program-owned v4 contract. Segment coverage audits deterministic high-signal "
                 "cues and residual context; it does not prove complete natural-language "
                 "semantic coverage.")
SEGMENT_CUE_KINDS = {"number", "date", "negation", "modality", "condition",
                     "designation", "attribution", "actor", "location",
                     "logic_connector", "lexical_content", "context"}
COVERAGE_STATUSES = {"covered_by_dimension", "covered_by_relation",
                     "logic_connector", "context_only", "suspected_missing"}
CUE_DIMENSION_KINDS = {
    "number": {"quantity_unit", "time", "baseline_scope", "entity_identity",
               "exact_designation"},
    "date": {"time", "baseline_scope", "entity_identity", "exact_designation"},
    "negation": {"negation"},
    "modality": {"modality"},
    "condition": {"condition"},
    "designation": {"predicate", "exact_designation"},
    "attribution": {"predicate"},
    "actor": {"actor_role"},
    "location": {"location"},
}
_DATE_CUE = re.compile(
    r"\b(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?|"
    r"\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)(?:\s+\d{4})?|\d{4}-\d{2}-\d{2}|(?:19|20)\d{2})\b",
    re.IGNORECASE)
_NUMBER_CUE = re.compile(
    r"(?<![\w-])(?:[+-]?\d[\d,]*(?:\.\d+)?(?:\s?(?:%|°[CF]?|kg|g|km|m|"
    r"million|billion|trillion))?|one|two|three|four|five|six|seven|eight|nine|"
    r"ten|eleven|twelve)(?![\w-])", re.IGNORECASE)
_NEGATION_CUE = re.compile(
    r"\b(?:not(?!\s+only\b)|no(?!\s+later\s+than\b)|never|neither|nor|without|"
    r"hardly|scarcely|rarely|failed\s+to|denied|\w+n['’]t)\b", re.IGNORECASE)
_MODALITY_CUE = re.compile(
    r"\b(?:may|might|could|can|must|should|would|will|likely|unlikely|"
    r"reportedly|allegedly)\b", re.IGNORECASE)
_CONDITION_CUE = re.compile(
    r"\b(?:if|when|unless|provided\s+that|subject\s+to|in\s+case\s+of)\b", re.IGNORECASE)
_BROAD_ATTRIBUTION_CUE = re.compile(
    r"\b(?:according\s+to|reported|announced|stated|said|wrote|concluded|found)\b",
    re.IGNORECASE)
_LOCATION_CUE = re.compile(
    r"\b(?:aboard|near|across|within|outside|inside|throughout|in|at)\s+"
    r"(?:the\s+)?[a-z][\w'’-]*(?:\s+[a-z][\w'’-]*){0,3}\b", re.IGNORECASE)
_PASSIVE_ACTOR_CUE = re.compile(
    r"\bby\s+(?:the\s+)?[a-z][\w'’-]*(?:\s+[a-z][\w'’-]*){0,3}\b",
    re.IGNORECASE)
_LOGIC_CONNECTOR_CUE = re.compile(
    r"\b(?:and|or|but|while|whereas|than|because|therefore|so\s+that)\b",
    re.IGNORECASE)
_MONTH_NAME = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\.?\b", re.IGNORECASE)
_MONTH_WORDS = {"january", "february", "march", "april", "may", "june", "july",
                "august", "september", "october", "november", "december", "jan",
                "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
                "nov", "dec"}
_CONTEXT_STOPWORDS = {"a", "an", "the", "to", "of", "for", "from", "with", "as",
                      "that", "this", "these", "those", "which", "who", "whom",
                      "whose", "and", "or", "but", "also", "then", "therefore", "so",
                      "by", "on", "at", "in", "into", "onto", "via"}
_ACCORDING_TO_CUE = re.compile(
    r"\baccording\s+to\s+(?P<speaker>[^,;:]{1,160}?)(?P<separator>\s*[,;:])\s*",
    re.IGNORECASE)
_REPORTING_ATTRIBUTION = re.compile(
    r"\b(?P<verb>reported|announced|stated|said|wrote|concluded|found)\s+that\b",
    re.IGNORECASE)
_SEARCH_TOKEN = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)
_SEARCH_STOPWORDS = {"about", "after", "and", "evidence", "find", "for", "from",
                     "into", "material", "one", "record", "search", "source", "that",
                     "the", "this", "through", "version", "what", "when", "where",
                     "which", "with"}


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
                            ("dataset_status", DATASET)):
        if config.get(field) != expected or (isinstance(expected, bool) and
                                              type(config.get(field)) is not bool):
            raise ValueError(f"Invalid source/execution assumption: {field}")
    task_routed = experiment == V4_TASK_ROUTED_EXPERIMENT
    expected_provider = TASK_ROUTED_PROVIDER if task_routed else PROVIDER
    if config.get("provider") != expected_provider:
        raise ValueError("Invalid source/execution assumption: provider")
    if task_routed:
        if (config.get("provider_mode") != "task_routed" or
                config.get("strict_retrieval_attribution") is not True or
                config.get("retrieval_attribution_mode") not in {None, "strict"}):
            raise ValueError("task-routed v4 experiment lacks its preregistered provider contract")
    elif experiment == V4_FIXED_EXPERIMENT:
        if (config.get("provider_mode") != "fixed_reanalysis" or
                config.get("strict_retrieval_attribution") is not False or
                config.get("retrieval_attribution_mode") not in {None, "legacy"}):
            raise ValueError("fixed-reanalysis v4 experiment lacks its preregistered provider contract")
    trace = config.get("trace_config")
    v4_fixed = experiment == V4_FIXED_EXPERIMENT
    if (not isinstance(trace, dict) or type(trace.get("max_rounds")) is not int or
            (trace["max_rounds"] != 2 if task_routed or v4_fixed else
             not 1 <= trace["max_rounds"] <= 3) or
            trace.get("experimental_force_rounds") is not (not task_routed)):
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

    if plan_schema == PLAN_SCHEMA_V4:
        required = {("predicate_core", ids("predicate")),
                    ("source_lineage", ())}
        for dimension, probe in (
                ("negation", "polarity"),
                ("quantity_unit", "quantity_unit"),
                ("time", "time_boundary"),
                ("location", "location"),
                ("baseline_scope", "baseline_scope"),
                ("condition", "condition_modality"),
                ("modality", "condition_modality"),
                ("actor_role", "actor_role"),
                ("entity_identity", "entity_identity"),
                ("exact_designation", "exact_designation")):
            required.update((probe, (identifier,))
                            for identifier in by_kind.get(dimension, ()))
        all_dimensions = tuple(sorted(dimension_ids))
        if claim.get("role") == "attributed_content":
            composition = ("attribution_relation", ())
        elif by_kind.get("exact_designation"):
            composition = ("designation_relation", all_dimensions)
        else:
            composition = ({
                "conditional": "conditional_relation",
                "comparison": "comparison_relation",
                "causal": "causal_relation",
            }.get(logic, "claim_composition"), all_dimensions)
        required.add(composition)
        if assessment_mode == "world":
            required.add(("source_independence", ()))
        return required

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
    """Independently recheck a retained versioned target contract."""
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
        if plan_schema == PLAN_SCHEMA_V4 and claim["statement"] != anchor["quote"]:
            raise ValueError("v4 claim statement must equal its immutable anchor quote")
        dimensions = claim["dimensions"]
        max_dimensions = 14 if plan_schema == PLAN_SCHEMA_V4 else 9
        allowed_kinds = DIMENSION_KINDS_V4 if plan_schema == PLAN_SCHEMA_V4 else DIMENSION_KINDS
        if not isinstance(dimensions, list) or len(dimensions) > max_dimensions:
            raise ValueError("v2 target plan claim dimensions must be an array")
        by_kind = {}
        for dimension in dimensions:
            if not isinstance(dimension, dict) or set(dimension) != DIMENSION_FIELDS:
                raise ValueError("v2 target plan has malformed dimensions")
            dimension_id, kind = dimension["id"], dimension["kind"]
            if (not isinstance(dimension_id, str) or not dimension_id or
                    dimension_id in dimension_ids or kind not in allowed_kinds):
                raise ValueError("v2 target plan has a duplicate or invalid dimension")
            dimension_ids.add(dimension_id)
            _audit_anchor(dimension["anchor"], text, "dimension", parent=anchor)
            by_kind.setdefault(kind, []).append(dimension["anchor"])
        if plan_schema == PLAN_SCHEMA_V4 and dimensions != sorted(dimensions, key=lambda item: (
                item["anchor"]["start"], item["anchor"]["end"], item["kind"], item["id"])):
            raise ValueError("v4 target plan dimensions are not in canonical order")
        if len(by_kind.get("subject", ())) != 1 or len(by_kind.get("predicate", ())) != 1:
            raise ValueError("v2 target plan claim must have exactly one subject and predicate")
        repeatable = ({"entity_identity"} if plan_schema != PLAN_SCHEMA_V4
                      else set(V4_REPEATABLE_DIMENSIONS))
        if plan_schema == PLAN_SCHEMA_V3:
            repeatable.add("time")
        repeated = [kind for kind, values in by_kind.items()
                    if kind not in repeatable and len(values) > 1]
        if repeated:
            raise ValueError("v2 target plan repeats a non-identity dimension kind")
        overlap_kinds = ({"time"} if plan_schema == PLAN_SCHEMA_V3 else
                         (V4_REPEATABLE_DIMENSIONS - {"entity_identity"}
                          if plan_schema == PLAN_SCHEMA_V4 else set()))
        for repeated_kind in overlap_kinds:
            anchors = by_kind.get(repeated_kind, ())
            for index, left in enumerate(anchors):
                for right in anchors[index + 1:]:
                    if max(left["start"], right["start"]) < min(left["end"], right["end"]):
                        raise ValueError("versioned target plan contains overlapping " +
                                         repeated_kind + " repeated dimensions")
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
    if plan_schema == PLAN_SCHEMA_V4:
        if claims != sorted(claims, key=lambda item: (
                item["anchor"]["start"], item["anchor"]["end"], item["role"], item["id"])):
            raise ValueError("v4 target plan claims are not in canonical order")
        if any(claim["parent_claim_id"] is not None and
               claim["role"] != "attributed_content" for claim in claims):
            raise ValueError("only v4 attributed content may have a parent claim")
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


def _audit_v4_logic(logic, claims, attributions, attributed_content, target):
    """Apply v4 cardinality and one-composition-gate representability rules."""
    _audit_v2_logic(logic, claims, attributions, attributed_content)
    explicit_nested = (_REPORTING_ATTRIBUTION.search(target["text"]) is not None or
                       _ACCORDING_TO_CUE.search(target["text"]) is not None)
    has_attribution_roles = bool(attributions or attributed_content)
    if explicit_nested or has_attribution_roles:
        if logic != "attribution" or len(attributions) != 1 or not attributed_content:
            raise ValueError("v4 explicit attribution lacks typed parent/content claims")
    if logic == "attribution" and any(
            claim["role"] not in {"attribution", "attributed_content"} for claim in claims):
        raise ValueError("v4 attribution contains an unsupported nested role")
    if logic in {"conditional", "comparison", "causal"} and any(
            dimension["kind"] == "exact_designation"
            for claim in claims for dimension in claim["dimensions"]):
        raise ValueError("v4 target needs more than one specialized composition gate")
    if any(claim["role"] == "attributed_content" and any(
            dimension["kind"] == "exact_designation"
            for dimension in claim["dimensions"]) for claim in claims):
        raise ValueError("v4 attributed designation cannot use one composition gate")
    _audit_v4_obvious_conjuncts(logic, claims, target["text"])
    _audit_v4_attribution_orientation(logic, claims, target["text"])


def _audit_v4_obvious_conjuncts(logic, claims, text):
    """Fail closed on a bounded, obvious two-clause ``and``/``or`` split.

    This is deliberately narrower than natural-language parsing.  It catches a
    capitalized three-token clause on each side of a conjunction, the concrete
    omission class the coverage ledger's generic ``context_only`` label cannot
    be allowed to hide.
    """
    word = re.compile(r"[A-Za-z][\w'’-]*")
    predicates = [dimension["anchor"] for claim in claims
                  for dimension in claim["dimensions"]
                  if dimension["kind"] == "predicate"]
    for match in re.finditer(r"\b(?:and|or)\b", text, re.IGNORECASE):
        left_start = max(text.rfind(mark, 0, match.start()) for mark in ".!?;") + 1
        right_marks = [position for mark in ".!?;"
                       for position in [text.find(mark, match.end())] if position >= 0]
        right_end = min(right_marks) if right_marks else len(text)
        left_tokens = word.findall(text[left_start:match.start()])
        right_tokens = word.findall(text[match.end():right_end])
        left_capitalized = bool(left_tokens and left_tokens[0][0].isupper())
        right_capitalized = bool(right_tokens and right_tokens[0][0].isupper())
        if not (len(left_tokens) >= 3 and len(right_tokens) >= 3 and
                left_capitalized and right_capitalized):
            continue
        has_left = any(left_start <= anchor["start"] and anchor["end"] <= match.start()
                       for anchor in predicates)
        has_right = any(match.end() <= anchor["start"] and anchor["end"] <= right_end
                        for anchor in predicates)
        expected_logic = match.group().casefold()
        if not has_left or not has_right or logic != expected_logic:
            raise ValueError(
                "v4 obvious conjunct clause is missing a separately anchored predicate")


def _audit_v4_attribution_orientation(logic, claims, text):
    """Recompute reporting cue, speaker, content boundary and parent direction."""
    nested = _REPORTING_ATTRIBUTION.search(text)
    according = _ACCORDING_TO_CUE.search(text)
    if nested is None and according is None:
        if logic == "attribution":
            raise ValueError("v4 attribution lacks a program-recognized reporting boundary")
        return
    if according is not None and (nested is None or according.start() < nested.start()):
        cue_start, cue_end = according.start(), according.start("speaker")
        while cue_end > cue_start and text[cue_end - 1].isspace():
            cue_end -= 1
        predicate_end = cue_end
        speaker_start, speaker_end = according.start("speaker"), according.end("speaker")
        content_start = according.end()
    else:
        cue_start, cue_end = nested.start("verb"), nested.end("verb")
        predicate_end = cue_end
        boundary = max(text.rfind(mark, 0, nested.start()) for mark in ".!?;") + 1
        while boundary < nested.start() and text[boundary].isspace():
            boundary += 1
        speaker_start, speaker_end = boundary, nested.start()
        while speaker_end > speaker_start and text[speaker_end - 1].isspace():
            speaker_end -= 1
        content_start = nested.end()
    while content_start < len(text) and text[content_start].isspace():
        content_start += 1
    parents = [claim for claim in claims if claim["role"] == "attribution"]
    contents = [claim for claim in claims if claim["role"] == "attributed_content"]
    if logic != "attribution" or len(parents) != 1 or not contents:
        raise ValueError("v4 explicit attribution cannot be flattened into single logic")
    parent = parents[0]
    if not (parent["anchor"]["start"] <= speaker_start < speaker_end <=
            parent["anchor"]["end"] and parent["anchor"]["start"] <= cue_start <
            cue_end <= parent["anchor"]["end"] and
            parent["anchor"]["end"] <= content_start):
        raise ValueError("v4 attribution/content roles reverse the reporting cue owner")
    predicates = [item["anchor"] for item in parent["dimensions"]
                  if item["kind"] == "predicate"]
    subjects = [item["anchor"] for item in parent["dimensions"]
                if item["kind"] in {"subject", "actor_role"}]
    if (not any(max(cue_start, anchor["start"]) < min(predicate_end, anchor["end"])
                for anchor in predicates) or
            not any(max(speaker_start, anchor["start"]) < min(speaker_end, anchor["end"])
                    for anchor in subjects)):
        raise ValueError("v4 attribution parent does not bind reporting predicate and speaker")
    if any(dimension["anchor"]["end"] > content_start
           for dimension in parent["dimensions"]):
        raise ValueError("v4 attribution parent dimensions cross the content boundary")
    if any(claim["parent_claim_id"] != parent["id"] or
           claim["anchor"]["start"] < content_start or
           any(item["anchor"]["start"] < content_start
               for item in claim["dimensions"]) for claim in contents):
        raise ValueError("v4 attributed content has reversed parent orientation")


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


def _target_segments_v4(target, signature, claims):
    """Recompute the v4 lexical segmentation without importing planner code."""
    patterns = (
        ("date", _DATE_CUE),
        ("number", _NUMBER_CUE),
        ("negation", _NEGATION_CUE),
        ("modality", _MODALITY_CUE),
        ("condition", _CONDITION_CUE),
        ("designation", DESIGNATION_CUE),
        ("attribution", _BROAD_ATTRIBUTION_CUE),
        ("logic_connector", _LOGIC_CONNECTOR_CUE),
    )
    text = target["text"]
    candidates = []
    for priority, (cue_kind, pattern) in enumerate(patterns):
        for match in pattern.finditer(text):
            candidates.append((match.start(), match.end(), priority, cue_kind))
    # A bare month is still a time expression (``in March``), even though the
    # more specific date expression above intentionally also handles full
    # dates.  Insert it at date priority; longest-first selection preserves the
    # full-date span when both expressions overlap.
    for match in _MONTH_NAME.finditer(text):
        candidates.append((match.start(), match.end(), 0, "date"))
    for match in _PASSIVE_ACTOR_CUE.finditer(text):
        candidates.append((match.start(), match.end(), 7, "actor"))
    for match in _LOCATION_CUE.finditer(text):
        words = tuple(re.findall(r"[^\W_]+(?:['’][^\W_]+)?",
                                 match.group().casefold(), re.UNICODE))
        objects = tuple(word for word in words[1:] if word != "the")
        if not objects or objects[0] in _MONTH_WORDS or any(
                char.isdigit() for char in match.group()):
            continue
        candidates.append((match.start(), match.end(), 8, "location"))
    chosen = []
    for start, end, priority, cue_kind in sorted(
            candidates, key=lambda item: (item[2], item[0], -(item[1] - item[0]))):
        if any(max(start, left) < min(end, right) for left, right, _, _ in chosen):
            continue
        chosen.append((start, end, priority, cue_kind))
    chosen.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

    def overlap(left_start, left_end, right_start, right_end):
        return max(left_start, right_start) < min(left_end, right_end)

    def owner_for(start, end, cue_kind):
        scored = []
        allowed = CUE_DIMENSION_KINDS.get(cue_kind)
        for order, claim in enumerate(claims):
            overlapping = [dimension for dimension in claim["dimensions"]
                           if overlap(start, end, dimension["anchor"]["start"],
                                      dimension["anchor"]["end"])]
            compatible = [dimension for dimension in overlapping
                          if allowed is None or dimension["kind"] in allowed]
            useful = compatible if compatible else overlapping
            overlap_size = sum(
                min(end, dimension["anchor"]["end"]) -
                max(start, dimension["anchor"]["start"])
                for dimension in useful)
            predicate_overlap = any(item["kind"] == "predicate" for item in useful)
            attribution_owner = cue_kind == "attribution" and claim["role"] == "attribution"
            if useful:
                distance = 0
            else:
                predicates = [item["anchor"] for item in claim["dimensions"]
                              if item["kind"] == "predicate"]
                distance = min(min(abs(start - anchor["end"]),
                                   abs(end - anchor["start"])) for anchor in predicates)
            anchor = claim["anchor"]
            contains = anchor["start"] <= start and end <= anchor["end"]
            scored.append(((bool(compatible), attribution_owner, overlap_size,
                            len(useful), predicate_overlap, contains, -distance, -order),
                           claim))
        return max(scored, key=lambda item: item[0])[1]

    pieces = []
    boundaries = sorted({position for claim in claims
                         for dimension in claim["dimensions"]
                         for position in (dimension["anchor"]["start"],
                                          dimension["anchor"]["end"])})

    def add_piece(start, end, cue_kind):
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start >= end or not text[start:end].strip():
            return
        if cue_kind == "context":
            words = tuple(re.findall(r"[^\W_]+(?:['’][^\W_]+)?",
                                     text[start:end].casefold(), re.UNICODE))
            if words and not all(word in _CONTEXT_STOPWORDS for word in words):
                cue_kind = "lexical_content"
        owner = owner_for(start, end, cue_kind)
        identifier = target["id"] + ":segment:" + _canonical_digest(
            [signature, owner["id"], start, end, cue_kind])[:20]
        pieces.append({
            "segment_id": identifier,
            "claim_id": owner["id"],
            "anchor": {"start": start, "end": end, "quote": text[start:end]},
            "cue_kind": cue_kind,
            "high_signal": cue_kind not in {"logic_connector", "context"},
        })

    def add_context(start, end):
        split = [start, *(position for position in boundaries if start < position < end), end]
        for left, right in zip(split, split[1:]):
            add_piece(left, right, "context")

    cursor = 0
    for start, end, _, cue_kind in chosen:
        add_context(cursor, start)
        add_piece(start, end, cue_kind)
        cursor = end
    add_context(cursor, len(text))
    return sorted(pieces, key=lambda item: (
        item["anchor"]["start"], item["anchor"]["end"], item["cue_kind"],
        item["claim_id"], item["segment_id"]))


def _audit_v4_coverage_ledger(plan, target):
    ledger = plan.get("coverage_ledger")
    if not isinstance(ledger, list) or len(ledger) > 192:
        raise ValueError("v4 coverage ledger must be a bounded array")
    expected_list = _target_segments_v4(target, plan["target_signature"], plan["claims"])
    expected = {item["segment_id"]: item for item in expected_list}
    dimensions = {dimension["id"]: dimension for claim in plan["claims"]
                  for dimension in claim["dimensions"]}
    if len(dimensions) != sum(len(claim["dimensions"]) for claim in plan["claims"]):
        raise ValueError("v4 coverage audit found duplicate dimension IDs")
    seen = set()
    covered = set()
    normalized = []
    fields = {"segment_id", "claim_id", "anchor", "cue_kind", "high_signal",
              "status", "dimension_ids"}
    for entry in ledger:
        if not isinstance(entry, dict) or set(entry) != fields:
            raise ValueError("v4 coverage ledger entry has invalid fields")
        segment_id = entry["segment_id"]
        if segment_id in seen or segment_id not in expected:
            raise ValueError("v4 coverage ledger has duplicate or unknown segment")
        seen.add(segment_id)
        segment = expected[segment_id]
        for key in ("claim_id", "anchor", "cue_kind", "high_signal"):
            if entry[key] != segment[key]:
                raise ValueError("v4 coverage ledger differs from recomputed target segment")
        status = entry["status"]
        dimension_ids = entry["dimension_ids"]
        if (status not in COVERAGE_STATUSES or not isinstance(dimension_ids, list) or
                any(not isinstance(item, str) or not item for item in dimension_ids) or
                len(dimension_ids) > 56 or dimension_ids != sorted(dimension_ids) or
                len(dimension_ids) != len(set(dimension_ids)) or
                not set(dimension_ids) <= set(dimensions)):
            raise ValueError("v4 coverage ledger has invalid classification or dimensions")
        if status == "covered_by_dimension":
            if not dimension_ids:
                raise ValueError("v4 covered segment lacks a dimension")
            bound = [dimensions[identifier] for identifier in dimension_ids]
            if any(max(entry["anchor"]["start"], item["anchor"]["start"]) >=
                   min(entry["anchor"]["end"], item["anchor"]["end"])
                   for item in bound):
                raise ValueError("v4 coverage dimension does not overlap its segment")
            allowed = CUE_DIMENSION_KINDS.get(entry["cue_kind"])
            if allowed is not None and not any(item["kind"] in allowed for item in bound):
                raise ValueError("v4 high-signal segment lacks a compatible dimension")
            covered.update(dimension_ids)
        elif status == "covered_by_relation":
            owner = next((claim for claim in plan["claims"]
                          if claim["id"] == entry["claim_id"]), None)
            relation_owned = bool(owner and
                owner["anchor"]["start"] <= entry["anchor"]["start"] and
                entry["anchor"]["end"] <= owner["anchor"]["end"] and
                (owner["role"] == "attributed_content" or
                 any(item["kind"] == "exact_designation"
                     for item in owner["dimensions"]) or
                 plan["logic"] in {"conditional", "comparison", "causal"}))
            if dimension_ids or entry["cue_kind"] != "lexical_content" or not relation_owned:
                raise ValueError("v4 lexical residue is not owned by a typed relation gate")
        elif status == "logic_connector":
            if dimension_ids or entry["cue_kind"] != "logic_connector":
                raise ValueError("v4 logic-connector classification is invalid")
        elif status == "context_only":
            if dimension_ids or entry["high_signal"] or entry["cue_kind"] != "context":
                raise ValueError("v4 high-signal segment cannot be context-only")
        else:
            if dimension_ids:
                raise ValueError("v4 suspected-missing segment cannot bind a dimension")
            raise ValueError("accepted v4 plan retains a suspected-missing segment")
        normalized.append(entry)
    if seen != set(expected):
        raise ValueError("v4 coverage ledger does not exactly cover recomputed segments")
    if covered != set(dimensions):
        raise ValueError("v4 coverage ledger does not cover every target dimension")
    canonical_order = sorted(normalized, key=lambda item: (
        item["claim_id"], item["anchor"]["start"], item["anchor"]["end"],
        item["cue_kind"], item["segment_id"]))
    if ledger != canonical_order:
        raise ValueError("v4 coverage ledger is not in deterministic order")
    return {"expected_segments": len(expected), "ledger_entries": len(ledger),
            "required_dimensions": len(dimensions), "covered_dimensions": len(covered),
            "breaks": 0}


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
    if not incomplete and lineage_gaps:
        raise ValueError(
            "v3 complete origin chain retains a spurious active lineage gap")
    return {"applicable": True, "origin_count": len(root_ids),
            "direct_lineage_edges": direct_edges,
            "reachable_origin_count": len(root_ids), "terminal_roots": True}


def _visible_ids_by_round(retrieval, report):
    """Rebuild cumulative accepted visibility from provider returns, not input corpus."""
    if not isinstance(retrieval, list) or not isinstance(report, dict) or not isinstance(
            report.get("analysis_history"), list):
        return None
    accepted_at = {}
    for item in report["analysis_history"]:
        if (not isinstance(item, dict) or item.get("accepted") is not True or
                type(item.get("round")) is not int or
                not isinstance(item.get("version_id"), str)):
            continue
        accepted_at[item["version_id"]] = min(
            item["round"], accepted_at.get(item["version_id"], item["round"]))
    returned = set()
    visible = {}
    for record in sorted(retrieval, key=lambda item: item.get("round", -1)
                         if isinstance(item, dict) else -1):
        if not isinstance(record, dict) or type(record.get("round")) is not int or not isinstance(
                record.get("returned"), list):
            raise ValueError("v4 cannot reconstruct round visibility from retrieval history")
        round_number = record["round"]
        for version_id in record["returned"]:
            if not isinstance(version_id, str) or not version_id:
                raise ValueError("v4 retrieval visibility contains an invalid version")
            returned.add(version_id)
        visible[round_number] = {version_id for version_id in returned
                                 if accepted_at.get(version_id, round_number + 1) <= round_number}
    return visible


def _audit_world_provenance_probe(probe, result, report, target, context):
    """Recompute supported lineage/independence from the retained direct graph."""
    if result["status"] != "supported" or probe.get("kind") not in {
            "source_lineage", "source_independence"}:
        return
    if not isinstance(report, dict):
        raise ValueError(context + " needs the retained provenance graph")
    materials = report.get("materials")
    eligible = report.get("eligible_version_ids")
    origins = report.get("origins")
    relations = report.get("relations")
    if not all(isinstance(value, list) for value in
               (materials, eligible, origins, relations)):
        raise ValueError(context + " has no auditable provenance graph")
    material_ids = {item.get("version_id") for item in materials
                    if isinstance(item, dict) and isinstance(item.get("version_id"), str)}
    eligible_ids = set(eligible)
    if len(material_ids) != len(materials) or not eligible_ids <= material_ids:
        raise ValueError(context + " has malformed eligible provenance materials")
    roots = {item.get("version_id") for item in origins
             if isinstance(item, dict) and item.get("target_id") == target.get("id") and
             isinstance(item.get("version_id"), str)}
    adjacency = {}
    for edge in relations:
        if (not isinstance(edge, dict) or edge.get("status") != "direct" or
                edge.get("kind") not in DIRECT_LINEAGE_KINDS):
            continue
        source, upstream = edge.get("from_version"), edge.get("to_version")
        if source in eligible_ids and upstream in eligible_ids:
            adjacency.setdefault(source, set()).add(upstream)

    def reachable(start):
        reached, pending = set(), [start]
        while pending:
            version_id = pending.pop()
            if version_id in reached or version_id not in eligible_ids:
                continue
            reached.add(version_id)
            pending.extend(adjacency.get(version_id, ()))
        return reached

    source_version = target.get("source_version_id")
    target_roots = reachable(source_version) & roots
    if not target_roots:
        raise ValueError(context + " claims supported lineage without a confirmed terminal path")
    if any((reachable(root) - {root}) & roots for root in roots):
        raise ValueError(context + " relies on a non-terminal exported origin")
    if probe["kind"] == "source_lineage":
        return
    supporting = sorted({span["version_id"] for span in result["basis"]})
    if len(supporting) < 2 or len(target_roots) < 2:
        raise ValueError(context + " claims source independence from fewer than two sources")
    components = [reachable(version_id) & target_roots for version_id in supporting]
    if any(not component for component in components) or not any(
            left.isdisjoint(right) for index, left in enumerate(components)
            for right in components[index + 1:]):
        raise ValueError(context + " source-independence basis shares one derivation component")


def _probe_result_audit(plan, verification, report_history, materials, target,
                        required=False, visible_by_round=None, report=None):
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
            allowed_versions = allowed_by_stage[stage]
            if visible_by_round is not None:
                if round_number not in visible_by_round:
                    raise ValueError("probe-result round has no reconstructed material visibility")
                allowed_versions = allowed_versions & visible_by_round[round_number]
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
                    _grounded_span(span, material_content, allowed_versions,
                                   context + f" basis {span_index}")
                if (probe.get("kind") == "designation_relation" and
                        result["status"] != "unresolved" and
                        not _has_designation_basis(
                            basis, _designation_labels(plan, probe["claim_id"]))):
                    raise ValueError(
                        context + " needs one naming-predicate basis span containing "
                        "every asserted designation"
                    )
                if stage == "world" and plan.get("schema_version") == PLAN_SCHEMA_V4:
                    _audit_world_provenance_probe(probe, result, report, target, context)
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


def _audit_material_probe_ledgers(plan, psi, report, required=False):
    """Join accepted PSI ledgers to retained decomposition findings.

    A stage-call count is never treated as probe coverage.  Coverage comes only
    from an exact projected-ID ledger whose finding indexes can be reconciled
    with the accepted analysis retained by the provenance engine.
    """
    zero = {"stage_calls": 0, "audited_stage_calls": 0,
            "expected_probe_checks": 0, "probe_checks": 0,
            "findings": 0, "referenced_findings": 0, "breaks": 0}
    if not required and report is None:
        return zero, []
    if not isinstance(report, dict) or not isinstance(report.get("analysis_history"), list):
        raise ValueError("v4 material-ledger audit requires report analysis_history")

    expected = {stage: {probe["id"] for probe in plan["probes"]
                        if stage in probe["routes"]}
                for stage in ("atoms", "lineage")}
    accepted_calls = [item for item in psi
                      if item.get("stage") in expected and item.get("status") == "accepted"]
    transactions = []
    pending = []
    for item in psi:
        stage, status = item.get("stage"), item.get("status")
        if stage in expected and status == "accepted":
            pending.append(item)
        elif stage == "critic" and status == "accepted":
            selected = {}
            for call in pending:
                selected[call["stage"]] = call
            if set(selected) != set(expected):
                raise ValueError("v4 accepted decomposition lacks final atoms or lineage call")
            if len({selected[stage].get("material") for stage in expected}) != 1 or len(
                    {selected[stage].get("round") for stage in expected}) != 1:
                raise ValueError("v4 decomposition stages disagree on material or round")
            if item.get("material") != selected["atoms"].get("material") or item.get(
                    "round") != selected["atoms"].get("round"):
                raise ValueError("v4 critic does not close its material transaction")
            transactions.append(selected)
            pending = []
        elif stage == "critic" and status in {"rejected", "repair_exhausted", "failed"}:
            pending = []
    if pending and required:
        raise ValueError("v4 PSI history ends with an unreviewed material ledger")

    revisions = [item for item in report["analysis_history"]
                 if isinstance(item, dict) and item.get("accepted") is True]
    if len(transactions) != len(revisions):
        raise ValueError("v4 accepted material transactions do not match analysis history")

    selected_sequences = {call["sequence"] for transaction in transactions
                          for call in transaction.values()}
    revision_by_sequence = {}
    decomposition_keys = []
    for transaction, revision in zip(transactions, revisions):
        if (revision.get("version_id") != transaction["atoms"].get("material") or
                revision.get("round") != transaction["atoms"].get("round")):
            raise ValueError("v4 PSI transaction does not match retained analysis revision")
        analysis = revision.get("analysis")
        if not isinstance(analysis, dict):
            raise ValueError("v4 accepted analysis revision has no analysis")
        for key in ("fragments", "relations", "origins"):
            if not isinstance(analysis.get(key), list):
                raise ValueError("v4 accepted analysis lacks material findings")
        for stage, call in transaction.items():
            revision_by_sequence[call["sequence"]] = analysis
        decomposition_keys.append((revision["round"], revision["version_id"]))

    counters = Counter(stage_calls=len(accepted_calls))
    for call in accepted_calls:
        stage = call["stage"]
        checks = call.get("probe_checks")
        if not isinstance(checks, list):
            raise ValueError("v4 accepted material stage lacks probe_checks")
        seen, used = set(), set()
        origin_used = False
        for check in checks:
            if not isinstance(check, dict) or set(check) != {
                    "probe_id", "status", "finding_indexes", "origin_used", "rationale"}:
                raise ValueError("v4 material probe check has invalid fields")
            probe_id = check["probe_id"]
            if probe_id not in expected[stage] or probe_id in seen:
                raise ValueError("v4 material ledger has unknown or duplicate probe")
            seen.add(probe_id)
            indexes = check["finding_indexes"]
            if (not isinstance(indexes, list) or
                    any(type(index) is not int or not 0 <= index <= 5 for index in indexes) or
                    len(indexes) != len(set(indexes))):
                raise ValueError("v4 material ledger has invalid finding indexes")
            used.update(indexes)
            uses_origin = check["origin_used"]
            if type(uses_origin) is not bool:
                raise ValueError("v4 material ledger origin_used must be boolean")
            if stage == "atoms" and uses_origin:
                raise ValueError("v4 atoms ledger cannot use a lineage origin")
            origin_used |= uses_origin
            if (check["status"] not in {"addressed", "absent", "ambiguous"} or
                    not isinstance(check["rationale"], str) or
                    not check["rationale"].strip()):
                raise ValueError("v4 material ledger has invalid status or rationale")
            has_finding = bool(indexes) or uses_origin
            if check["status"] == "absent" and has_finding:
                raise ValueError("v4 absent material check cites a finding")
            if check["status"] != "absent" and not has_finding:
                raise ValueError("v4 non-absent material check lacks a finding")
        if seen != expected[stage]:
            raise ValueError("v4 material ledger does not exactly cover projected probes")
        # Even superseded valid drafts must use a contiguous zero-based finding
        # index set; their raw findings are not persisted after full replacement.
        if used and used != set(range(max(used) + 1)):
            raise ValueError("v4 material ledger skips a finding index")
        counters["audited_stage_calls"] += 1
        counters["expected_probe_checks"] += len(expected[stage])
        counters["probe_checks"] += len(checks)

        analysis = revision_by_sequence.get(call["sequence"])
        if analysis is None:
            continue
        findings = analysis["fragments" if stage == "atoms" else "relations"]
        available_origin = stage == "lineage" and bool(analysis["origins"])
        if any(index >= len(findings) for index in used):
            raise ValueError("v4 material ledger finding index escapes retained findings")
        if origin_used and not available_origin:
            raise ValueError("v4 lineage ledger uses an absent retained origin")
        if used != set(range(len(findings))):
            raise ValueError("v4 retained material finding is outside its probe ledger")
        counters["findings"] += len(findings)
        counters["referenced_findings"] += len(used)

    return ({key: counters[key] for key in (
        "stage_calls", "audited_stage_calls", "expected_probe_checks", "probe_checks",
        "findings", "referenced_findings")} | {"breaks": 0}, decomposition_keys)


def _audit_strict_followups(plan, report_history, materials, target, required=False,
                            visible_by_round=None):
    metrics = {"unresolved_probe_slots": 0, "task_followups": 0,
               "allowed_stops": 0, "covered_slots": 0, "breaks": 0}
    if not required and not isinstance(report_history, list):
        return metrics
    if not isinstance(report_history, list):
        raise ValueError("v4 strict-followup audit requires verification history")
    probes = {item["id"]: item for item in plan["probes"]}
    material_content = {}
    material_metadata = {}
    for item in materials or []:
        if (not isinstance(item, dict) or not isinstance(item.get("version_id"), str) or
                not isinstance(item.get("content"), str) or
                item["version_id"] in material_content):
            raise ValueError("v4 strict-followup materials are invalid")
        material_content[item["version_id"]] = item["content"]
        material_metadata[item["version_id"]] = item
    allowed_stops = {"no_source_lead"}
    for record in report_history:
        if not isinstance(record, dict) or record.get("strict_probe_followups") is not True:
            raise ValueError("v4 verification cycle lacks strict_probe_followups=true")
        results = []
        for stage in ("evidence", "world"):
            values = record.get(stage + "_probe_results")
            if not isinstance(values, list):
                raise ValueError("v4 strict-followup cycle lacks probe results")
            results.extend((stage, item) for item in values)
        unresolved = {(stage, item.get("probe_id")) for stage, item in results
                      if isinstance(item, dict) and item.get("status") == "unresolved"}
        conclusive = {(stage, item.get("probe_id")) for stage, item in results
                      if isinstance(item, dict) and item.get("status") != "unresolved"}
        gaps = record.get("gaps")
        stops = record.get("probe_stops")
        if not isinstance(gaps, list) or not isinstance(stops, list):
            raise ValueError("v4 strict-followup cycle lacks gaps or probe_stops")
        task_keys = []
        for gap in gaps:
            if not isinstance(gap, dict):
                raise ValueError("v4 strict follow-up task is malformed")
            key = (gap.get("dimension"), gap.get("probe_id"))
            aggregate = record.get(key[0] + "_verdict")
            if aggregate is None:
                aggregate = record.get("verdict")
            if aggregate not in PROBE_STATUSES:
                raise ValueError("v4 strict follow-up lacks its aggregate layer verdict")
            expected_blocking = aggregate == "unresolved"
            probe = probes.get(key[1])
            if (gap.get("stage") != "verification" or key not in unresolved or
                    gap.get("target_id") != target.get("id") or
                    gap.get("action") not in {"fetch", "search", "reanalyse"} or
                    not isinstance(gap.get("locator"), str) or not gap["locator"].strip() or
                    not isinstance(gap.get("question"), str) or not gap["question"].strip() or
                    not isinstance(gap.get("decision_impact"), str) or
                    not gap["decision_impact"].strip() or
                    gap.get("blocking") is not expected_blocking):
                raise ValueError("v4 strict follow-up task has invalid probe ownership")
            if (probe is None or gap["question"] != probe["question"] or
                    gap["decision_impact"] != probe["decision_impact"]):
                raise ValueError("v4 strict follow-up changes its canonical probe question or impact")
            basis = gap.get("basis")
            if not isinstance(basis, list) or not basis:
                raise ValueError("v4 strict follow-up task lacks a source-backed lead")
            allowed_ids = ((set(target.get("evidence_scope", ())) or set(material_content))
                           if key[0] == "evidence" else set(material_content))
            if visible_by_round is not None:
                round_number = record.get("round")
                if round_number not in visible_by_round:
                    raise ValueError("v4 follow-up round has no reconstructed material visibility")
                allowed_ids &= visible_by_round[round_number]
            for index, span in enumerate(basis):
                _grounded_span(span, material_content, allowed_ids,
                               "v4 strict follow-up basis " + str(index))
            action, locator = gap["action"], gap["locator"]
            if action == "fetch":
                if not (urlsplit(locator).scheme in {"http", "https"} and
                        urlsplit(locator).netloc):
                    raise ValueError("v4 fetch follow-up lacks an explicit HTTP locator")
                declared = any(locator in span["quote"] for span in basis)
                visible_metadata = any(locator in {
                    material_metadata[span["version_id"]].get("url"),
                    material_metadata[span["version_id"]].get("version_id"),
                } for span in basis)
                if not declared and not visible_metadata:
                    raise ValueError("v4 fetch follow-up locator is not traceable to its basis")
            if action == "reanalyse" and (locator not in allowed_ids or
                    locator not in {span["version_id"] for span in basis}):
                raise ValueError("v4 reanalyse follow-up does not name its visible basis version")
            if action == "search":
                claim = next(item for item in plan["claims"]
                             if item["id"] == probe["claim_id"])

                def terms(value):
                    return {token.casefold() for token in _SEARCH_TOKEN.findall(value)
                            if (len(token) >= 2 or token.isdigit()) and
                            token.casefold() not in _SEARCH_STOPWORDS}

                trace_text = " ".join([probe["question"], claim["statement"],
                                       *(span["quote"] for span in basis)])
                if not terms(locator) or not (terms(locator) & terms(trace_text)):
                    raise ValueError("v4 search follow-up locator is unrelated to its probe and basis")
            task_keys.append(key)
        stop_keys = []
        for stop in stops:
            if (not isinstance(stop, dict) or set(stop) != {
                    "probe_id", "stage", "reason", "rationale"}):
                raise ValueError("v4 probe stop is malformed")
            key = (stop["stage"], stop["probe_id"])
            if (key not in unresolved or stop["reason"] not in allowed_stops or
                    not isinstance(stop["rationale"], str) or not stop["rationale"].strip()):
                raise ValueError("v4 probe stop is not an allowed unresolved-probe stop")
            stop_keys.append(key)
        if len(task_keys) != len(set(task_keys)) or len(stop_keys) != len(set(stop_keys)):
            raise ValueError("v4 strict follow-up repeats a probe slot")
        task_set, stop_set = set(task_keys), set(stop_keys)
        if task_set & stop_set or task_set | stop_set != unresolved:
            raise ValueError("v4 unresolved probe does not have exactly one task or allowed stop")
        if (task_set | stop_set) & conclusive:
            raise ValueError("v4 conclusive probe has a follow-up")
        metrics["unresolved_probe_slots"] += len(unresolved)
        metrics["task_followups"] += len(task_set)
        metrics["allowed_stops"] += len(stop_set)
        metrics["covered_slots"] += len(task_set | stop_set)
    return metrics


def _registered_retrieval_tasks(report, target):
    """Rebuild immutable task definitions and the first round they can be issued.

    Material and verification gaps are registered only after their producing
    round has completed, so they may first drive retrieval in the next round.
    The initial origin task and the no-gap runner fallback are program-owned.
    """
    initial = {
        "id": "origin:" + target["id"],
        "question": "Find the producing record and evidenced lineage for: " + target["text"],
        "stage": "provenance", "dimension": "auto", "blocking": True,
        "target_id": None, "basis": [], "decision_impact": "",
        "action": "search", "locator": None, "probe_id": None,
    }
    inspect = {
        "id": "inspect-lineage", "question": "Inspect unresolved upstream lineage",
        "stage": "provenance", "dimension": "auto", "blocking": True,
        "target_id": None, "basis": [], "decision_impact": "",
        "action": "search", "locator": None, "probe_id": None,
    }
    sources = {initial["id"]: [(1, initial)]}

    def retain(gap, available_round):
        if (not isinstance(gap, dict) or set(gap) != RETRIEVAL_TASK_FIELDS or
                not isinstance(gap.get("id"), str) or not gap["id"]):
            raise ValueError("strict retrieval report contains a malformed registered gap")
        sources.setdefault(gap["id"], []).append((available_round, gap))

    history = report.get("analysis_history")
    verification = report.get("verification_history")
    if not isinstance(history, list) or not isinstance(verification, list):
        raise ValueError("strict retrieval report lacks task registry histories")
    for revision in history:
        if not isinstance(revision, dict) or revision.get("accepted") is not True:
            continue
        round_number, analysis = revision.get("round"), revision.get("analysis")
        if type(round_number) is not int or not isinstance(analysis, dict) or not isinstance(
                analysis.get("gaps"), list):
            raise ValueError("strict retrieval accepted analysis has no auditable gap registry")
        relations = analysis.get("relations")
        if not isinstance(relations, list):
            raise ValueError("strict retrieval accepted analysis has no relation array")
        for gap in analysis["gaps"]:
            if isinstance(gap, dict) and gap.get("probe_id") is None:
                locator = gap.get("locator")
                action = ("fetch" if isinstance(locator, str) and
                          urlsplit(locator).scheme in {"http", "https"} and
                          urlsplit(locator).netloc else "search")
                matching_relations = [edge for edge in relations
                    if (isinstance(edge, dict) and
                        edge.get("from_version") == revision.get("version_id") and
                        edge.get("to_version") is None and
                        edge.get("status") == "declared" and
                        edge.get("kind") in DIRECT_LINEAGE_KINDS and
                        edge.get("upstream_locator") == locator and
                        edge.get("basis") == gap.get("basis"))]
                digest = hashlib.sha256(json.dumps(
                    [target["id"], action, locator], sort_keys=True,
                    ensure_ascii=False).encode()).hexdigest()[:20]
                expected_id = revision["version_id"] + ":upstream:" + digest
                if (len(matching_relations) != 1 or gap.get("id") != expected_id or
                        gap.get("question") != "Locate the explicitly cited upstream: " +
                        str(locator) or gap.get("stage") != "provenance" or
                        gap.get("dimension") != "provenance" or
                        gap.get("blocking") is not True or
                        gap.get("target_id") != target["id"] or
                        gap.get("action") != action or
                        not isinstance(gap.get("decision_impact"), str) or
                        not gap["decision_impact"].strip()):
                    raise ValueError(
                        "strict retrieval provenance task is not derived from its citation")
            retain(gap, round_number + 1)
    for record in verification:
        if not isinstance(record, dict) or type(record.get("round")) is not int or not isinstance(
                record.get("gaps"), list):
            raise ValueError("strict retrieval verification history has no auditable gap registry")
        for gap in record["gaps"]:
            retain(gap, record["round"] + 1)

    # The engine persists the canonical registry independently of provider
    # history.  Join to it, but do not trust it as the only source: every
    # model-created entry must also be present in an accepted material or
    # verification history.  The remaining entries are narrowly bounded
    # program-owned runtime tasks.
    retained = report.get("gap_registry")
    if not isinstance(retained, list):
        raise ValueError("strict retrieval report lacks its canonical gap registry")
    registry = {}
    lineage = {
        "id": "lineage:" + target["id"],
        "question": ("Provide an evidenced direct citation/derivation path from the "
                     "target source to a terminal original material"),
        "stage": "provenance", "dimension": "auto", "blocking": True,
        "target_id": None, "basis": [], "decision_impact": "",
        "action": "search", "locator": None, "probe_id": None,
    }
    for gap in retained:
        if (not isinstance(gap, dict) or set(gap) != RETRIEVAL_TASK_FIELDS or
                not isinstance(gap.get("id"), str) or not gap["id"] or
                gap["id"] in registry):
            raise ValueError("strict retrieval report has a malformed canonical gap registry")
        candidates = sources.get(gap["id"], ())
        if gap == initial:
            candidates = ((1, initial),)
        elif gap == inspect:
            candidates = ((1, inspect),)
        elif gap == lineage:
            candidates = ((2, lineage),)
        if not any(_retrieval_task_lifecycle(gap, target) ==
                   _retrieval_task_lifecycle(definition, target)
                   for _, definition in candidates):
            raise ValueError("strict retrieval registry contains an unproduced task definition")
        registry[gap["id"]] = gap
    if initial["id"] not in registry:
        raise ValueError("strict retrieval registry omits the program-owned origin task")
    return registry


def _retrieval_task_lifecycle(gap, target):
    dimension = ("provenance" if gap["stage"] == "provenance" else "world") \
        if gap["dimension"] == "auto" else gap["dimension"]
    return (gap["stage"], dimension, gap["target_id"] or target["id"],
            gap["action"], gap["locator"], gap["probe_id"])


def _active_retrieval_tasks_by_round(report, target, rounds):
    """Replay full-replacement analyses and verifier task lifecycle per round."""
    initial = {
        "id": "origin:" + target["id"],
        "question": "Find the producing record and evidenced lineage for: " + target["text"],
        "stage": "provenance", "dimension": "auto", "blocking": True,
        "target_id": None, "basis": [], "decision_impact": "",
        "action": "search", "locator": None, "probe_id": None,
    }
    inspect = {
        "id": "inspect-lineage", "question": "Inspect unresolved upstream lineage",
        "stage": "provenance", "dimension": "auto", "blocking": True,
        "target_id": None, "basis": [], "decision_impact": "",
        "action": "search", "locator": None, "probe_id": None,
    }
    lineage = {
        "id": "lineage:" + target["id"],
        "question": ("Provide an evidenced direct citation/derivation path from the "
                     "target source to a terminal original material"),
        "stage": "provenance", "dimension": "auto", "blocking": True,
        "target_id": None, "basis": [], "decision_impact": "",
        "action": "search", "locator": None, "probe_id": None,
    }
    revisions = report.get("analysis_history")
    verification = report.get("verification_history")
    if not isinstance(revisions, list) or not isinstance(verification, list):
        raise ValueError("strict retrieval cannot replay task lifecycle histories")
    revisions_by_round = {}
    for item in revisions:
        if not isinstance(item, dict) or type(item.get("round")) is not int:
            raise ValueError("strict retrieval analysis history has an invalid round")
        if item.get("accepted") is True:
            revisions_by_round.setdefault(item["round"], []).append(item)
    verification_by_round = {}
    for item in verification:
        if (not isinstance(item, dict) or type(item.get("round")) is not int or
                item["round"] in verification_by_round):
            raise ValueError("strict retrieval verification history has invalid rounds")
        verification_by_round[item["round"]] = item

    operations = report.get("operations")
    if not isinstance(operations, list):
        raise ValueError("strict retrieval cannot audit verifier supersession events")
    followup_events = {}
    for item in operations:
        if not isinstance(item, dict) or item.get("action") != "probe_followups_validated":
            continue
        round_number = item.get("round")
        if (set(item) != {"sequence", "round", "action", "active", "stops",
                          "stopped_prior_gap_ids", "superseded_prior_gaps"} or
                type(round_number) is not int or round_number in followup_events or
                not all(isinstance(item.get(field), list) for field in (
                    "active", "stops", "stopped_prior_gap_ids",
                    "superseded_prior_gaps"))):
            raise ValueError("strict retrieval has malformed verifier supersession events")
        followup_events[round_number] = item
    if set(followup_events) != set(verification_by_round):
        raise ValueError("strict retrieval lacks exact verifier supersession events")

    current = {}
    current_round = {}
    verify_active = {}
    reopened_round = {}
    explicit_verify_resolved = set()
    superseded_gap_ids = set()

    def current_resolution_ids():
        values = set(explicit_verify_resolved)
        for owner, analysis in current.items():
            for resolution in analysis.get("resolutions", ()):
                gap_id = resolution.get("gap_id") if isinstance(resolution, dict) else None
                if not isinstance(gap_id, str):
                    raise ValueError("strict retrieval current analysis has malformed resolutions")
                if gap_id not in verify_active or current_round[owner] > reopened_round.get(gap_id, 0):
                    values.add(gap_id)
        return values

    def incomplete_origin_chain():
        eligible = set(current)
        adjacency = {}
        candidates = set()
        for analysis in current.values():
            for edge in analysis.get("relations", ()):
                if (isinstance(edge, dict) and edge.get("status") == "direct" and
                        edge.get("kind") in DIRECT_LINEAGE_KINDS and
                        edge.get("from_version") in eligible and
                        edge.get("to_version") in eligible):
                    adjacency.setdefault(edge["from_version"], set()).add(edge["to_version"])
            for origin in analysis.get("origins", ()):
                if (isinstance(origin, dict) and origin.get("target_id") == target["id"] and
                        origin.get("version_id") in eligible):
                    candidates.add(origin["version_id"])

        def reachable(start):
            reached, pending = set(), [start]
            while pending:
                version_id = pending.pop()
                if version_id in reached or version_id not in eligible:
                    continue
                reached.add(version_id)
                pending.extend(adjacency.get(version_id, ()))
            return reached

        target_reachable = reachable(target.get("source_version_id"))
        reachable_candidates = candidates & target_reachable
        downstream = {item: reachable(item) for item in reachable_candidates}
        terminals = {item for item, reached in downstream.items()
                     if not ((reached - {item}) & reachable_candidates)}
        return (len(reachable_candidates) != len(candidates) or
                any(not (reached & terminals) for reached in downstream.values()))

    def projection():
        active = {}

        def add(gap):
            if (not isinstance(gap, dict) or set(gap) != RETRIEVAL_TASK_FIELDS or
                    not isinstance(gap.get("id"), str) or not gap["id"]):
                raise ValueError("strict retrieval task replay found a malformed gap")
            if gap["id"] in active and active[gap["id"]] != gap:
                raise ValueError("strict retrieval task replay found conflicting gap definitions")
            active[gap["id"]] = gap

        add(initial)
        for analysis in current.values():
            gaps = analysis.get("gaps")
            if not isinstance(gaps, list):
                raise ValueError("strict retrieval current analysis lacks a gap array")
            for gap in gaps:
                add(gap)
        for gap in verify_active.values():
            add(gap)
        if incomplete_origin_chain():
            add(lineage)
        resolved = current_resolution_ids()
        active = {identifier: gap for identifier, gap in active.items()
                  if identifier not in resolved}
        return list(active.values()) or [inspect]

    expected = {}
    for round_number in range(1, rounds + 1):
        expected[round_number] = projection()
        for revision in revisions_by_round.get(round_number, ()):
            analysis = revision.get("analysis")
            owner = revision.get("version_id")
            if not isinstance(owner, str) or not isinstance(analysis, dict):
                raise ValueError("strict retrieval accepted revision is malformed")
            for name in ("gaps", "resolutions", "relations", "origins"):
                if not isinstance(analysis.get(name), list):
                    raise ValueError("strict retrieval accepted revision lacks replay fields")
            current[owner] = analysis
            current_round[owner] = round_number
        record = verification_by_round.get(round_number)
        if record is None:
            continue
        gaps, resolutions, stops = (record.get("gaps"), record.get("resolutions"),
                                    record.get("probe_stops"))
        if not all(isinstance(value, list) for value in (gaps, resolutions, stops)):
            raise ValueError("strict retrieval verification lifecycle fields are malformed")
        new_gaps_by_slot = {}
        for gap in gaps:
            if not isinstance(gap, dict) or set(gap) != RETRIEVAL_TASK_FIELDS:
                raise ValueError("strict retrieval verification gap is malformed")
            if gap["id"] in superseded_gap_ids:
                raise ValueError("strict retrieval reissues a superseded verification gap")
            key = (gap.get("dimension"), gap.get("probe_id"))
            if key in new_gaps_by_slot:
                raise ValueError("strict retrieval verification repeats a probe slot")
            new_gaps_by_slot[key] = gap
            verify_active[gap["id"]] = gap
            explicit_verify_resolved.discard(gap["id"])
            reopened_round[gap["id"]] = round_number
        for resolution in resolutions:
            gap_id = resolution.get("gap_id") if isinstance(resolution, dict) else None
            if not isinstance(gap_id, str):
                raise ValueError("strict retrieval verification resolution is malformed")
            verify_active.pop(gap_id, None)
            explicit_verify_resolved.add(gap_id)
        results = {}
        for stage in ("evidence", "world"):
            values = record.get(stage + "_probe_results")
            if not isinstance(values, list):
                raise ValueError("strict retrieval verification replay lacks probe results")
            results.update({(stage, item.get("probe_id")): item for item in values
                            if isinstance(item, dict)})
        verdicts = {stage: record.get(stage + "_verdict", record.get("verdict"))
                    for stage in ("evidence", "world")}
        for gap_id, gap in list(verify_active.items()):
            key = (gap["dimension"], gap.get("probe_id"))
            result = results.get(key)
            if result is None:
                continue
            if result.get("status") != "unresolved":
                verify_active.pop(gap_id)
                explicit_verify_resolved.add(gap_id)
            else:
                expected_blocking = verdicts[key[0]] == "unresolved"
                if gap.get("blocking") is not expected_blocking:
                    gap = dict(gap)
                    gap["blocking"] = expected_blocking
                    verify_active[gap_id] = gap
        superseded = []
        for gap_id, gap in list(verify_active.items()):
            key = (gap.get("dimension"), gap.get("probe_id"))
            replacement = new_gaps_by_slot.get(key)
            if replacement is not None and replacement["id"] != gap_id:
                verify_active.pop(gap_id)
                superseded_gap_ids.add(gap_id)
                superseded.append({
                    "stage": key[0], "probe_id": key[1],
                    "prior_gap_id": gap_id,
                    "replacement_gap_id": replacement["id"],
                })
        superseded.sort(key=lambda item: (
            item["stage"], item["probe_id"], item["prior_gap_id"],
            item["replacement_gap_id"]))
        stopped_prior_gap_ids = []
        for stop in stops:
            if (not isinstance(stop, dict) or set(stop) != {
                    "probe_id", "stage", "reason", "rationale"}):
                raise ValueError("strict retrieval probe stop is malformed")
            for gap_id, gap in list(verify_active.items()):
                if (gap.get("dimension"), gap.get("probe_id")) == (
                        stop.get("stage"), stop.get("probe_id")):
                    verify_active.pop(gap_id)
                    stopped_prior_gap_ids.append(gap_id)
        active_rows = []
        active_slots = set()
        for gap_id, gap in verify_active.items():
            key = (gap.get("dimension"), gap.get("probe_id"))
            if key in active_slots:
                raise ValueError("strict retrieval replay retains duplicate active probe slots")
            active_slots.add(key)
            active_rows.append({
                "stage": key[0], "probe_id": key[1], "gap_id": gap_id,
                "blocking": gap.get("blocking"),
            })
        active_rows.sort(key=lambda item: (item["stage"], item["probe_id"]))
        sorted_stops = sorted(stops, key=lambda item: (item["stage"], item["probe_id"]))
        event = followup_events[round_number]
        if (event["active"] != active_rows or event["stops"] != sorted_stops or
                event["stopped_prior_gap_ids"] != sorted(stopped_prior_gap_ids) or
                event["superseded_prior_gaps"] != superseded):
            raise ValueError("strict retrieval verifier supersession ledger disagrees with replay")
    final_active = projection()
    if len(final_active) == 1 and final_active[0]["id"] == "inspect-lineage":
        final_active = []
    return expected, final_active


def _audit_retrieval_attribution(plan, retrieval, report, decomposition_keys,
                                 strict, required=False):
    metrics = {"search_rounds": 0, "issued_tasks": 0, "provider_returns": 0,
               "attributed_returns": 0, "task_links": 0,
               "valid_task_links": 0, "later_probe_owned_tasks": 0,
               "later_probe_owned_hit_tasks": 0, "breaks": 0}
    if retrieval is None:
        if required:
            raise ValueError("v4 retrieval audit requires retained provider history")
        return metrics
    if not isinstance(retrieval, list):
        raise ValueError("v4 retrieval history must be an array")
    metrics["search_rounds"] = len(retrieval)
    if not strict:
        for record in retrieval:
            if (not isinstance(record, dict) or not isinstance(record.get("tasks"), list) or
                    not isinstance(record.get("returned"), list) or
                    "attribution" in record):
                raise ValueError("fixed reanalysis retrieval history is malformed")
            metrics["issued_tasks"] += len(record["tasks"])
            metrics["provider_returns"] += len(record["returned"])
        if report is not None and report.get("retrieval_attribution_mode") != "legacy-compatible":
            raise ValueError("fixed reanalysis report incorrectly claims strict attribution")
        return metrics

    if not isinstance(report, dict) or report.get("retrieval_attribution_mode") != "strict":
        raise ValueError("task-routed v4 report must declare strict retrieval attribution")
    probe_ids = {item["id"] for item in plan["probes"]}
    probes = {item["id"]: item for item in plan["probes"]}
    target = report.get("target")
    if (not isinstance(target, dict) or
            _canonical_digest(target) != plan.get("target_signature")):
        raise ValueError("strict retrieval report lacks its immutable target")
    registry = _registered_retrieval_tasks(report, target)
    active_by_round, final_active = _active_retrieval_tasks_by_round(
        report, target, len(retrieval))
    expected_returns = []
    expected_feedback = []
    expected_rounds = []
    tasks_by_round = {}
    seen_rounds = set()
    returned_before = set()
    for record in retrieval:
        if not isinstance(record, dict) or record.get("kind") != "task_routed_fixed_corpus":
            raise ValueError("strict retrieval round is malformed")
        round_number = record.get("round")
        if type(round_number) is not int or round_number <= 0 or round_number in seen_rounds:
            raise ValueError("strict retrieval history has an invalid or duplicate round")
        seen_rounds.add(round_number)
        expected_rounds.append(round_number)
        tasks, returned, attribution = (record.get("tasks"), record.get("returned"),
                                        record.get("attribution"))
        feedback = record.get("feedback")
        if (not isinstance(tasks, list) or not isinstance(returned, list) or
                not isinstance(attribution, list) or not isinstance(feedback, list) or
                len(attribution) != len(returned) or len(returned) != len(set(returned))):
            raise ValueError("strict retrieval round lacks exact task/hit ledgers")
        if tasks != active_by_round.get(round_number):
            raise ValueError("strict retrieval tasks are not the replayed active gap set")
        task_map = {}
        for task in tasks:
            if (not isinstance(task, dict) or set(task) != RETRIEVAL_TASK_FIELDS or
                    not isinstance(task.get("id"), str) or not task["id"]):
                raise ValueError("strict retrieval round contains an invalid issued task")
            if task["id"] in task_map:
                raise ValueError("strict retrieval round repeats an issued task")
            probe_id = task.get("probe_id")
            retained = registry.get(task["id"])
            if (retained is None or _retrieval_task_lifecycle(task, target) !=
                    _retrieval_task_lifecycle(retained, target)):
                raise ValueError("strict retrieval task was not in the prior registered gap set")
            if task["id"] == "inspect-lineage" and len(tasks) != 1:
                raise ValueError("inspect-lineage is only valid as the sole no-gap fallback")
            if probe_id is None:
                if task["stage"] != "provenance":
                    raise ValueError("non-probe retrieval task is not a provenance task")
            else:
                probe = probes.get(probe_id)
                if (probe is None or task["stage"] != "verification" or
                        task["dimension"] not in {"evidence", "world"} or
                        task["target_id"] != target.get("id") or
                        task["question"] != probe["question"] or
                        task["decision_impact"] != probe["decision_impact"]):
                    raise ValueError("strict retrieval task is not bound to its frozen target probe")
                if round_number > 1:
                    metrics["later_probe_owned_tasks"] += 1
            task_map[task["id"]] = task
        tasks_by_round[round_number] = tasks
        metrics["issued_tasks"] += len(task_map)
        hit_tasks = set()
        for version_id, hit in zip(returned, attribution):
            if (not isinstance(version_id, str) or not version_id or not isinstance(hit, dict) or
                    set(hit) != {"version_id", "task_ids"} or
                    hit["version_id"] != version_id or
                    not isinstance(hit["task_ids"], list) or not hit["task_ids"] or
                    any(task_id not in task_map for task_id in hit["task_ids"]) or
                    len(hit["task_ids"]) != len(set(hit["task_ids"]))):
                raise ValueError("strict retrieval return lacks valid issued-task attribution")
            task_positions = [list(task_map).index(task_id) for task_id in hit["task_ids"]]
            if task_positions != sorted(task_positions):
                raise ValueError("strict retrieval attribution changes issued-task order")
            if version_id in returned_before and any(
                    task_map[task_id].get("action") != "reanalyse"
                    for task_id in hit["task_ids"]):
                raise ValueError("strict retrieval may repeat a version only for reanalyse")
            hit_tasks.update(hit["task_ids"])
            trigger_probes = list(dict.fromkeys(
                task_map[task_id].get("probe_id") for task_id in hit["task_ids"]
                if task_map[task_id].get("probe_id") is not None))
            expected_returns.append((round_number, version_id,
                                     tuple(hit["task_ids"]), tuple(trigger_probes)))
            metrics["attributed_returns"] += 1
            metrics["task_links"] += len(hit["task_ids"])
            metrics["valid_task_links"] += len(hit["task_ids"])
        metrics["provider_returns"] += len(returned)
        returned_before.update(returned)
        exhausted = set()
        for item in feedback:
            if (not isinstance(item, dict) or set(item) != {
                    "task_id", "probe_id", "status", "detail"} or
                    item.get("status") != "corpus_exhausted" or
                    item.get("task_id") not in task_map or
                    item.get("probe_id") != task_map[item["task_id"]].get("probe_id") or
                    not isinstance(item.get("detail"), str) or not item["detail"].strip()):
                raise ValueError("strict retrieval feedback is not an issued-task outcome")
            if item["task_id"] in exhausted:
                raise ValueError("strict retrieval repeats a task exhaustion outcome")
            exhausted.add(item["task_id"])
            expected_feedback.append((round_number, item))
        if hit_tasks & exhausted or hit_tasks | exhausted != set(task_map):
            raise ValueError("each issued retrieval task needs exactly one hit-or-exhausted outcome")
        metrics["later_probe_owned_hit_tasks"] += sum(
            round_number > 1 and task_map[task_id].get("probe_id") in probe_ids
            for task_id in hit_tasks)
    if (expected_rounds != list(range(1, len(expected_rounds) + 1)) or
            len(expected_rounds) > 2):
        raise ValueError("strict retrieval rounds are not consecutive adaptive rounds")
    final_gaps = report.get("gaps")
    if (not isinstance(final_gaps, list) or
            {item.get("id"): item for item in final_gaps if isinstance(item, dict)} !=
            {item["id"]: item for item in final_active} or
            len(final_gaps) != len(final_active)):
        raise ValueError("strict retrieval replay disagrees with the report's final active gaps")

    operations = report.get("operations")
    observations = report.get("observations")
    history = report.get("analysis_history")
    if not all(isinstance(value, list) for value in (operations, observations, history)):
        raise ValueError("strict retrieval report lacks attribution audit collections")
    if any(not isinstance(item, dict) or item.get("sequence") != index
           for index, item in enumerate(operations, 1)):
        raise ValueError("strict retrieval operations do not retain canonical sequence order")
    search_operations = [item for item in operations
                         if isinstance(item, dict) and item.get("action") == "search"]
    if (len(search_operations) != len(tasks_by_round) or
            {item.get("round") for item in search_operations} != set(tasks_by_round) or
            any(item.get("tasks") != tasks_by_round[item.get("round")]
                for item in search_operations)):
        raise ValueError("strict retrieval issued tasks disagree with report search operations")
    operation_returns = [(item.get("round"), item.get("version_id"),
                          tuple(item.get("trigger_task_ids", ())),
                          tuple(item.get("trigger_probe_ids", ())))
                         for item in operations if isinstance(item, dict) and
                         item.get("action") == "retrieval_attribution_validated"]
    observation_returns = [(item.get("round"), item.get("version_id"),
                            tuple(item.get("trigger_task_ids", ())),
                            tuple(item.get("trigger_probe_ids", ())))
                           for item in observations if isinstance(item, dict) and
                           "trigger_task_ids" in item]
    # Observations do not historically store round; pair their immutable return
    # fields to provider history in encounter order and supply that round here.
    observation_returns = [
        (expected_returns[index][0], version_id, task_ids, probe_ids_value)
        for index, (_, version_id, task_ids, probe_ids_value) in enumerate(observation_returns)
    ] if len(observation_returns) == len(expected_returns) else observation_returns
    history_returns = [(item.get("round"), item.get("version_id"),
                        tuple(item.get("trigger_task_ids", ())),
                        tuple(item.get("trigger_probe_ids", ())))
                       for item in history if isinstance(item, dict) and
                       "trigger_task_ids" in item]
    if any(("trigger_task_ids" in item) != ("trigger_probe_ids" in item) or
           ("trigger_task_ids" in item and item.get("revisit") is not False)
           for item in history if isinstance(item, dict)):
        raise ValueError("strict direct-return and dependency-revisit histories are conflated")
    if any(item.get("accepted") is not True for item in history
           if isinstance(item, dict) and "trigger_task_ids" in item):
        raise ValueError("strict provider return lacks an accepted analysis history entry")
    if (operation_returns != expected_returns or observation_returns != expected_returns or
            history_returns != expected_returns):
        raise ValueError("strict retrieval attribution disagrees across provider and report")
    feedback_operations = [(item.get("round"), item.get("feedback"))
                           for item in operations if isinstance(item, dict) and
                           item.get("action") == "retrieval_feedback"]
    if feedback_operations != expected_feedback:
        raise ValueError("strict retrieval feedback disagrees with report operations")
    expected_decompositions = Counter((round_number, version_id)
                                      for round_number, version_id, _, _ in expected_returns)
    retained_decompositions = Counter(decomposition_keys)
    if any(retained_decompositions[key] < count
           for key, count in expected_decompositions.items()):
        raise ValueError("strict provider returns do not all have accepted atoms and lineage histories")
    for round_number, version_id, task_ids, trigger_probes in expected_returns:
        def matching(action, require_triggers=False):
            values = [item for item in operations
                      if item.get("round") == round_number and
                      item.get("version_id") == version_id and
                      item.get("action") == action]
            if require_triggers:
                values = [item for item in values
                          if tuple(item.get("trigger_task_ids", ())) == task_ids and
                          tuple(item.get("trigger_probe_ids", ())) == trigger_probes]
            return values

        validated = matching("retrieval_attribution_validated", True)
        direct_history = [item for item in history
                          if item.get("round") == round_number and
                          item.get("version_id") == version_id and
                          tuple(item.get("trigger_task_ids", ())) == task_ids and
                          tuple(item.get("trigger_probe_ids", ())) == trigger_probes]
        if len(direct_history) != 1 or type(direct_history[0].get("duplicate")) is not bool:
            raise ValueError("strict retrieval return lacks one direct accepted revision")
        lifecycle = ("duplicate_observed" if direct_history[0]["duplicate"]
                     else "snapshot_saved")
        snapshots = matching(lifecycle)
        wrong_lifecycle = matching("snapshot_saved" if lifecycle == "duplicate_observed"
                                   else "duplicate_observed")
        started = matching("decompose_started", True)
        completed = matching("decompose_completed", True)
        if (wrong_lifecycle or not all(len(values) == 1 for values in
                                      (validated, snapshots, started, completed))):
            raise ValueError("strict retrieval return lacks one save/decompose operation chain")
        if snapshots[0].get("eligible") is not True:
            raise ValueError("strict retrieval return was not saved as an eligible snapshot")
        search_sequence = next(item["sequence"] for item in search_operations
                               if item.get("round") == round_number)
        sequences = [search_sequence, validated[0]["sequence"], snapshots[0]["sequence"],
                     started[0]["sequence"], completed[0]["sequence"]]
        if sequences != sorted(sequences) or len(sequences) != len(set(sequences)):
            raise ValueError("strict retrieval attribution must precede save and decomposition")
    for round_number, _ in expected_feedback:
        feedback_sequences = [item["sequence"] for item in operations
                              if item.get("round") == round_number and
                              item.get("action") == "retrieval_feedback"]
        completed_sequences = [item["sequence"] for item in operations
                               if item.get("round") == round_number and
                               item.get("action") == "decompose_completed"]
        verification_sequences = [item["sequence"] for item in operations
                                   if item.get("round") == round_number and
                                   item.get("action") == "verification_started"]
        if ((completed_sequences and min(feedback_sequences) <= max(completed_sequences)) or
                (verification_sequences and max(feedback_sequences) >=
                 min(verification_sequences))):
            raise ValueError("strict retrieval feedback is out of lifecycle order")
    return metrics


def _graph_snapshot_at_round(report, target, round_number):
    """Rebuild the target-connected lineage subgraph after one round.

    ``edges`` retains the complete eligible direct graph for diagnostics.  Causal
    delta attribution must use ``path_edges``/``path_nodes``/``roots`` instead:
    an unrelated side edge is a graph change, but it cannot resolve a frozen
    source probe about this target.
    """
    current = {}
    for revision in report.get("analysis_history", ()):
        if (isinstance(revision, dict) and revision.get("accepted") is True and
                type(revision.get("round")) is int and revision["round"] <= round_number and
                isinstance(revision.get("version_id"), str) and
                isinstance(revision.get("analysis"), dict)):
            current[revision["version_id"]] = revision["analysis"]
    eligible = set(current)
    edges = set()
    adjacency = {}
    candidates = set()
    for analysis in current.values():
        for edge in analysis.get("relations", ()):
            if (isinstance(edge, dict) and edge.get("status") == "direct" and
                    edge.get("kind") in DIRECT_LINEAGE_KINDS and
                    edge.get("from_version") in eligible and edge.get("to_version") in eligible):
                key = (edge["from_version"], edge["to_version"], edge["kind"])
                edges.add(key)
                adjacency.setdefault(key[0], set()).add(key[1])
        for origin in analysis.get("origins", ()):
            if (isinstance(origin, dict) and origin.get("target_id") == target.get("id") and
                    origin.get("version_id") in eligible):
                candidates.add(origin["version_id"])

    def reachable(start):
        reached, pending = set(), [start]
        while pending:
            version_id = pending.pop()
            if version_id in reached or version_id not in eligible:
                continue
            reached.add(version_id)
            pending.extend(adjacency.get(version_id, ()))
        return reached

    from_target = reachable(target.get("source_version_id"))
    reachable_candidates = candidates & from_target
    roots = {candidate for candidate in reachable_candidates
             if not ((reachable(candidate) - {candidate}) & reachable_candidates)}
    path_nodes = {version_id for version_id in from_target
                  if reachable(version_id) & roots}
    path_edges = {edge for edge in edges
                  if edge[0] in path_nodes and edge[1] in path_nodes}
    return {"edges": edges, "path_edges": path_edges,
            "roots": roots, "path_nodes": path_nodes}


def _aggregate_frozen_probe_statuses(plan, keyed_results, stage, target, visible_ids):
    """Independently evaluate one layer using only the frozen flat plan logic."""
    probes = {item["id"]: item for item in plan["probes"]}
    by_claim = {}
    for (result_stage, probe_id), result in keyed_results.items():
        if result_stage != stage or probe_id not in probes:
            continue
        probe = probes[probe_id]
        by_claim.setdefault(probe["claim_id"], []).append(
            (probe, result.get("status")))

    def and_status(values):
        values = set(values)
        if not values:
            return "unresolved"
        for status in ("contradicted", "conflicting", "unresolved"):
            if status in values:
                return status
        return "supported"

    claim_statuses = []
    for claim in plan["claims"]:
        checks = by_claim.get(claim["id"], ())
        base = and_status(status for probe, status in checks
                          if probe.get("gate") == "always")
        for probe, status in checks:
            if (probe.get("gate") in {"provenance", "positive_world_only"} and
                    base == "supported" and status != "supported"):
                base = "unresolved"
        claim_statuses.append(base)
    logic = plan.get("logic")
    if logic in {"single", "conditional", "comparison", "causal"}:
        aggregate = claim_statuses[0] if len(claim_statuses) == 1 else "unresolved"
    elif logic in {"and", "attribution"}:
        aggregate = and_status(claim_statuses)
    elif logic == "or":
        values = set(claim_statuses)
        if not values:
            aggregate = "unresolved"
        elif "supported" in values:
            aggregate = "supported"
        elif "conflicting" in values:
            aggregate = "conflicting"
        elif "unresolved" in values:
            aggregate = "unresolved"
        else:
            aggregate = "contradicted"
    else:
        aggregate = "unresolved"
    if (stage == "evidence" and target.get("evidence_scope") and
            not set(target["evidence_scope"]) <= set(visible_ids)):
        return "unresolved"
    return aggregate


def _decision_probe_status(value):
    """Map public decision labels and internal statuses to one status alphabet."""
    return {"true": "supported", "false": "contradicted",
            "mixed": "conflicting", "unverifiable": "unresolved"}.get(value, value)


def _audit_probe_delta_attribution(plan, report_history, retrieval, report, target, row,
                                   strict, required=False, psi=None):
    """Attribute round>1 semantic probe changes to routed material receipts."""
    metrics = {"round_transitions": 0, "probe_slots_compared": 0,
               "semantic_deltas": 0, "basis_drifts": 0,
               "traced_semantic_deltas": 0,
               "graph_traced_semantic_deltas": 0,
               "probe_owned_novel_second_pass_cases": 0,
               "label_changes": 0, "label_changes_with_decisive_delta": 0,
               "breaks": 0}
    if not strict:
        return metrics
    if not isinstance(report_history, list) or not isinstance(retrieval, list) or not isinstance(
            report, dict):
        if required:
            raise ValueError("v4 probe-delta audit lacks routed histories")
        return metrics
    records = sorted(report_history, key=lambda item: item.get("round", -1))
    if len(records) < 2:
        return metrics
    if not isinstance(psi, list):
        raise ValueError("v4 probe-delta audit lacks material probe ledgers")
    probes = {item["id"]: item for item in plan["probes"]}
    retrieval_by_round = {item.get("round"): item for item in retrieval
                          if isinstance(item, dict)}
    returned_before = set()
    visible_by_round = _visible_ids_by_round(retrieval, report)
    if not isinstance(visible_by_round, dict):
        raise ValueError("v4 probe-delta audit cannot rebuild accepted material visibility")
    novel_by_round = {}
    triggered_by_round = {}
    novel_triggered_by_round = {}
    reanalysed_by_round = {}
    probe_owned_novel_rounds = set()
    for record in sorted(retrieval, key=lambda item: item.get("round", -1)):
        round_number = record["round"]
        tasks = {item["id"]: item for item in record["tasks"]}
        novel_by_round[round_number] = set(record["returned"]) - returned_before
        trigger_map = {}
        novel_trigger_map = {}
        reanalysis_map = {}
        for hit in record["attribution"]:
            version_id = hit["version_id"]
            for task_id in hit["task_ids"]:
                task = tasks[task_id]
                probe_id = task.get("probe_id")
                if probe_id is not None:
                    trigger_map.setdefault((task.get("dimension"), probe_id), set()).add(
                        version_id)
                    if (round_number > 1 and version_id in novel_by_round[round_number] and
                            task.get("action") in {"fetch", "search"}):
                        probe_owned_novel_rounds.add(round_number)
                        novel_trigger_map.setdefault(
                            (task.get("dimension"), probe_id), set()).add(version_id)
                    if task.get("action") == "reanalyse":
                        reanalysis_map.setdefault((task.get("dimension"), probe_id), set()).add(
                            version_id)
        triggered_by_round[round_number] = trigger_map
        novel_triggered_by_round[round_number] = novel_trigger_map
        reanalysed_by_round[round_number] = reanalysis_map
        returned_before.update(record["returned"])
    verified_rounds = {item.get("round") for item in records if isinstance(item, dict)}
    metrics["probe_owned_novel_second_pass_cases"] = int(bool(
        probe_owned_novel_rounds & verified_rounds))

    changed_reanalyses = {}
    latest_analysis = {}
    for revision in report.get("analysis_history", ()):
        if not isinstance(revision, dict) or revision.get("accepted") is not True:
            continue
        version_id, round_number = revision.get("version_id"), revision.get("round")
        analysis = revision.get("analysis")
        if not isinstance(version_id, str) or type(round_number) is not int or not isinstance(
                analysis, dict):
            continue
        if (revision.get("revisit") is False and "trigger_task_ids" in revision and
                version_id in latest_analysis and latest_analysis[version_id] != analysis):
            changed_reanalyses.setdefault(round_number, set()).add(version_id)
        latest_analysis[version_id] = analysis

    material_checks = {}
    for call in psi:
        if (not isinstance(call, dict) or call.get("status") != "accepted" or
                call.get("stage") not in {"atoms", "lineage"}):
            continue
        checks = call.get("probe_checks")
        if not isinstance(checks, list):
            continue
        for check in checks:
            if isinstance(check, dict):
                material_checks[(call.get("round"), call.get("material"),
                                 call["stage"], check.get("probe_id"))] = check.get("status")

    counterfactual_by_transition = {}

    def result_map(record):
        result = {}
        for stage in ("evidence", "world"):
            for item in record.get(stage + "_probe_results", ()):
                if isinstance(item, dict):
                    result[(stage, item.get("probe_id"))] = item
        return result

    for previous, current in zip(records, records[1:]):
        round_number = current.get("round")
        if round_number != previous.get("round") + 1 or round_number not in retrieval_by_round:
            raise ValueError("v4 probe-delta audit cannot align consecutive routed rounds")
        before, after = result_map(previous), result_map(current)
        if set(before) != set(after):
            raise ValueError("v4 probe-delta audit sees changing projected probe slots")
        metrics["round_transitions"] += 1
        metrics["probe_slots_compared"] += len(after)
        before_graph = _graph_snapshot_at_round(report, target, previous["round"])
        after_graph = _graph_snapshot_at_round(report, target, round_number)
        target_graph_fields = ("path_edges", "path_nodes", "roots")
        target_graph_changed = any(
            before_graph[field] != after_graph[field]
            for field in target_graph_fields)
        changed_path_edges = (before_graph["path_edges"] ^
                              after_graph["path_edges"])
        target_delta_participants = (
            before_graph["path_nodes"] ^ after_graph["path_nodes"] |
            before_graph["roots"] ^ after_graph["roots"] |
            {version_id for edge in changed_path_edges
             for version_id in edge[:2]})
        newly_target_connected = (
            after_graph["path_nodes"] - before_graph["path_nodes"] |
            after_graph["roots"] - before_graph["roots"] |
            {version_id
             for edge in after_graph["path_edges"] - before_graph["path_edges"]
             for version_id in edge[:2]})
        novel_delta_keys = set()
        for key in sorted(after):
            old, new = before[key], after[key]
            if old.get("basis") != new.get("basis"):
                metrics["basis_drifts"] += 1
            semantic = (old.get("status"), old.get("referent_relation")) != (
                new.get("status"), new.get("referent_relation"))
            if not semantic:
                continue
            metrics["semantic_deltas"] += 1
            stage, probe_id = key
            probe = probes.get(probe_id)
            if probe is None:
                raise ValueError("v4 probe-delta references an unknown frozen probe")
            triggered_versions = triggered_by_round.get(round_number, {}).get(key, set())
            novel_triggered = novel_triggered_by_round.get(round_number, {}).get(key, set())
            basis_versions = {item.get("version_id") for item in new.get("basis", ())
                              if isinstance(item, dict)}
            novel = novel_by_round.get(round_number, set())
            reanalysed = (reanalysed_by_round.get(round_number, {}).get(key, set()) &
                          changed_reanalyses.get(round_number, set()))
            routed_material_stages = ({"lineage"} if probe["kind"] in {
                "source_lineage", "source_independence"} else
                set(probe.get("routes", ())) & {"atoms", "lineage"})

            def psi_addressed(version_id):
                return bool(routed_material_stages) and all(
                    material_checks.get((round_number, version_id, material_stage, probe_id))
                    in {"addressed", "ambiguous"}
                    for material_stage in routed_material_stages)

            traced = False
            graph_trace = False
            novel_trace = False
            if probe["kind"] in {"source_lineage", "source_independence"}:
                involved = (novel | reanalysed) & triggered_versions
                delta_involved = involved & target_delta_participants
                graph_trace = bool(target_graph_changed and delta_involved and
                    any(psi_addressed(version_id)
                        for version_id in delta_involved))
                novel_involved = novel_triggered & newly_target_connected
                novel_trace = bool(target_graph_changed and novel_involved and
                                   any(psi_addressed(version_id)
                                       for version_id in novel_involved))
                traced = graph_trace
            elif triggered_versions:
                addressed = {version_id for version_id in triggered_versions
                             if psi_addressed(version_id)}
                if new.get("status") == "unresolved":
                    traced = bool(addressed & (novel | reanalysed))
                    novel_trace = bool(addressed & novel_triggered)
                else:
                    traced = bool(basis_versions & addressed & (novel | reanalysed))
                    novel_trace = bool(basis_versions & addressed & novel_triggered)
            if not traced:
                raise ValueError(
                    "v4 semantic probe delta is not attributable to its routed material")
            metrics["traced_semantic_deltas"] += 1
            metrics["graph_traced_semantic_deltas"] += graph_trace
            if novel_trace:
                novel_delta_keys.add(key)

        assessment_stage = target.get("assessment_mode")
        hybrid = dict(before)
        for key in novel_delta_keys:
            if key[0] == assessment_stage:
                hybrid[key] = after[key]
        prior_aggregate = _aggregate_frozen_probe_statuses(
            plan, before, assessment_stage, target,
            visible_by_round.get(previous["round"], set()))
        final_aggregate = _aggregate_frozen_probe_statuses(
            plan, after, assessment_stage, target,
            visible_by_round.get(round_number, set()))
        hybrid_aggregate = _aggregate_frozen_probe_statuses(
            plan, hybrid, assessment_stage, target,
            visible_by_round.get(round_number, set()))
        reported_prior = previous.get(assessment_stage + "_verdict",
                                      previous.get("verdict"))
        reported_final = current.get(assessment_stage + "_verdict",
                                     current.get("verdict"))
        if (reported_prior in PROBE_STATUSES and reported_prior != prior_aggregate) or (
                reported_final in PROBE_STATUSES and reported_final != final_aggregate):
            raise ValueError("v4 probe-delta frozen-plan aggregate disagrees with report")
        counterfactual_by_transition[round_number] = (
            bool(novel_delta_keys) and hybrid_aggregate == final_aggregate and
            hybrid_aggregate != prior_aggregate)

    checkpoints = row.get("checkpoints") if isinstance(row, dict) else None
    if isinstance(checkpoints, list) and checkpoints:
        first = next((item.get("decision") for item in checkpoints
                      if item.get("round") == records[0].get("round")), None)
        final_prediction = row.get("prediction")
        if isinstance(final_prediction, dict):
            final_prediction = final_prediction.get("decision")
        if first is not None and final_prediction is not None and first != final_prediction:
            metrics["label_changes"] = 1
            if not metrics["probe_owned_novel_second_pass_cases"]:
                raise ValueError(
                    "v4 label change did not occur in a probe-owned novel second pass")
            first_status = _decision_probe_status(first)
            final_status = _decision_probe_status(final_prediction)
            assessment_stage = target.get("assessment_mode")
            first_aggregate = _aggregate_frozen_probe_statuses(
                plan, result_map(records[0]), assessment_stage, target,
                visible_by_round.get(records[0]["round"], set()))
            final_aggregate = _aggregate_frozen_probe_statuses(
                plan, result_map(records[-1]), assessment_stage, target,
                visible_by_round.get(records[-1]["round"], set()))
            if first_status != first_aggregate or final_status != final_aggregate:
                raise ValueError("v4 public label change disagrees with frozen-plan aggregate")
            if not counterfactual_by_transition.get(records[-1]["round"]):
                raise ValueError(
                    "v4 final label change lacks a novel-receipt counterfactual delta")
            metrics["label_changes_with_decisive_delta"] = 1
    return metrics


def _audit_plan(path, target, row, psi, verification, expected_contract=None,
                report_history=None, materials=None, report=None, retrieval=None,
                strict_retrieval=None, calls=None):
    plan = _read(path)
    if not isinstance(plan, dict):
        raise ValueError("target plan must be an object")
    contract = _plan_contract(plan, expected_contract)
    target_signature = _canonical_digest(target)
    projections = plan.get("projections")
    if not isinstance(projections, dict) or set(projections) != set(PLAN_STAGES):
        raise ValueError("target plan must retain every stage projection")
    if contract in VERSIONED_PLAN_SCHEMAS:
        plan_fields = PLAN_FIELDS_V4 if contract == PLAN_SCHEMA_V4 else PLAN_FIELDS_V2
        projection_fields = (PROJECTION_FIELDS_V4 if contract == PLAN_SCHEMA_V4
                             else PROJECTION_FIELDS_V2)
        if set(plan) != plan_fields:
            raise ValueError("v2 target plan has missing or unexpected fields")
        critic = projections["critic"]
        if not isinstance(critic, dict) or set(critic) != projection_fields:
            raise ValueError("v2 critic projection has missing or unexpected fields")
        if (critic.get("schema_version") != contract or
                critic.get("stage") != "critic" or
                critic.get("target_signature") != target_signature or
                not isinstance(critic.get("notes"), str)):
            raise ValueError("v2 critic projection metadata mismatch")
        canonical_fields = ["schema_version", "target_signature", "logic", "claims", "probes"]
        if contract == PLAN_SCHEMA_V4:
            canonical_fields.append("coverage_ledger")
        canonical_fields.append("notes")
        canonical = {key: critic[key] for key in canonical_fields}
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
    max_probes = 64 if contract == PLAN_SCHEMA_V4 else 44
    if contract in VERSIONED_PLAN_SCHEMAS and not (
            isinstance(claims, list) and isinstance(probes, list) and
            1 <= len(claims) <= 4 and 1 <= len(probes) <= max_probes):
        raise ValueError("v2 target plan exceeds its planner schema bounds")
    claim_ids, probe_ids = _ids(claims, "target claims"), _ids(probes, "decision probes")
    if contract in VERSIONED_PLAN_SCHEMAS:
        _, _, attributions, attributed_content = _audit_v2_claims(
            claims, target, plan_schema=contract)
        if contract == PLAN_SCHEMA_V4:
            _audit_v4_logic(plan["logic"], claims, attributions,
                            attributed_content, target)
            if plan["notes"] != PLAN_NOTES_V4:
                raise ValueError("v4 target plan notes are not program canonical")
        else:
            _audit_v2_logic(plan["logic"], claims, attributions, attributed_content)
    known_claims = set(claim_ids)
    claims_by_id = {item["id"]: item for item in claims}
    by_id = {}
    actual_bindings = set()
    routing_contract = (V4_PROBE_ROUTES_AND_GATES if contract == PLAN_SCHEMA_V4 else
                        V2_PROBE_ROUTES_AND_GATES if contract in VERSIONED_PLAN_SCHEMAS
                        else LEGACY_PROBE_ROUTES_AND_GATES)
    full_gate_kinds = {"claim_composition", "designation_relation",
                       "attribution_relation", "conditional_relation",
                       "comparison_relation", "causal_relation"}
    full_gate_counts = Counter()
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
        if contract == PLAN_SCHEMA_V4 and dimension_ids != sorted(dimension_ids):
            raise ValueError("v4 probe dimension bindings are not canonical")
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
        if contract == PLAN_SCHEMA_V4:
            dimensions_by_id = {item["id"]: item for item in
                                claims_by_id[probe["claim_id"]]["dimensions"]}
            expected_question = _canonical_probe_question(
                claims_by_id[probe["claim_id"]], kind, tuple(dimension_ids),
                dimensions_by_id, claims_by_id)
            if probe["question"] != expected_question:
                raise ValueError("v4 probe question is not program-owned canonical text")
            if probe["decision_impact"] != _canonical_decision_impact(kind):
                raise ValueError("v4 probe decision impact is not program-owned canonical text")
            if kind in full_gate_kinds:
                full_gate_counts[probe["claim_id"]] += 1
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
    if contract == PLAN_SCHEMA_V4 and any(
            full_gate_counts[claim_id] != 1 for claim_id in claim_ids):
        raise ValueError("v4 requires exactly one composition gate per claim")
    if contract in VERSIONED_PLAN_SCHEMAS:
        _audit_v2_stable_ids(target, target_signature, claims, probes)
    if contract == PLAN_SCHEMA_V4:
        probe_order = [(probe["claim_id"], probe["kind"],
                        tuple(probe["dimension_ids"]), probe["id"]) for probe in probes]
        if probe_order != sorted(probe_order):
            raise ValueError("v4 decision probes are not in canonical order")
        coverage_ledger_audit = _audit_v4_coverage_ledger(plan, target)
    else:
        coverage_ledger_audit = None
    covered_bindings = required_bindings & actual_bindings

    projection_slots = covered_projection_slots = 0
    projected_by_stage = {}
    for stage in PLAN_STAGES:
        view = projections[stage]
        if (not isinstance(view, dict) or view.get("target_signature") != plan["target_signature"] or
                view.get("plan_sha256") != plan["sha256"] or view.get("stage") != stage):
            raise ValueError("target plan projection metadata mismatch")
        expected_projection_fields = (PROJECTION_FIELDS_V4 if contract == PLAN_SCHEMA_V4
                                      else PROJECTION_FIELDS_V2)
        if contract in VERSIONED_PLAN_SCHEMAS and (set(view) != expected_projection_fields or
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
        if contract == PLAN_SCHEMA_V4:
            expected_dimension_ids = {dimension["id"] for claim in expected_claims
                                      for dimension in claim["dimensions"]}
            expected_ledger = (plan["coverage_ledger"] if stage == "critic" else [
                entry for entry in plan["coverage_ledger"]
                if (entry["claim_id"] in expected_claim_ids or
                    set(entry["dimension_ids"]) & expected_dimension_ids)])
            if view["coverage_ledger"] != expected_ledger:
                raise ValueError("v4 projection coverage ledger differs from frozen plan")
        projected_by_stage[stage] = len(expected_probe_ids)

    observed = ({item["stage"] for item in psi} |
                {item["stage"] for item in verification})
    at_executed_stage = sum(count for stage, count in projected_by_stage.items()
                            if stage != "critic" and stage in observed)
    routed_total = sum(projected_by_stage[stage] for stage in ROUTED_STAGES)
    terminal_origin_audit = (_audit_v3_terminal_origins(report, target)
                             if contract in {PLAN_SCHEMA_V3, PLAN_SCHEMA_V4} and
                             report is not None else None)
    visible_by_round = (_visible_ids_by_round(retrieval, report)
                        if contract == PLAN_SCHEMA_V4 else None)
    probe_results = _probe_result_audit(plan, verification, report_history, materials, target,
        required=(contract in VERSIONED_PLAN_SCHEMAS and row.get("status") == "completed"),
        visible_by_round=visible_by_round, report=report)
    v4_audit = None
    if contract == PLAN_SCHEMA_V4:
        completed = row.get("status") == "completed"
        material_audit, decomposition_keys = _audit_material_probe_ledgers(
            plan, psi, report, required=completed)
        followup_audit = _audit_strict_followups(
            plan, report_history, materials, target, required=completed,
            visible_by_round=visible_by_round)
        retrieval_audit = _audit_retrieval_attribution(
            plan, retrieval, report, decomposition_keys,
            strict=bool(strict_retrieval), required=completed)
        delta_audit = _audit_probe_delta_attribution(
            plan, report_history, retrieval, report, target, row,
            strict=bool(strict_retrieval), required=completed, psi=psi)
        receipt_audit = (audit_loop_receipt_artifacts(
            calls, psi, verification, report, retrieval, target["id"],
            required=True)
            if completed and strict_retrieval and calls is not None else
            audit_loop_receipt_artifacts(
                [], [], [], None, [], target["id"], required=False))
        v4_audit = {"coverage_ledger": coverage_ledger_audit,
                    "material_probe_ledger": material_audit,
                    "strict_followups": followup_audit,
                    "retrieval_attribution": retrieval_audit,
                    "probe_delta_attribution": delta_audit,
                    "live_artifact_receipts": receipt_audit}
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
        "v4_audit": v4_audit,
    }


def _plan_coverage(run, ids, inputs, rows, config=None):
    by_id = {row["id"]: row for row in rows}
    totals = Counter()
    result_totals = Counter()
    result_statuses = Counter()
    result_stages = {stage: Counter() for stage in ("evidence", "world")}
    cases = []
    v4_cases = []
    expected_contract = _extension_contract(config) if config is not None else None
    strict_retrieval = bool(config and str(config.get("experiment", "")).endswith(
        "_v4_task_routed"))
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
            retrieval_path = run / f"{case_id}-retrieval.json"
            retrieval = _read(retrieval_path) if retrieval_path.exists() else None
            calls_path = run / f"{case_id}-calls.json"
            if (expected_contract == PLAN_SCHEMA_V4 and strict_retrieval and
                    row.get("status") == "completed" and not calls_path.exists()):
                raise ValueError(
                    "completed task-routed v4 case lacks retained model calls")
            calls = _read(calls_path) if calls_path.exists() else None
            audit = _audit_plan(path, inputs[case_id]["target"], row, psi, verification,
                expected_contract=expected_contract, report_history=report_history,
                materials=inputs[case_id].get("materials"), report=report,
                retrieval=retrieval, strict_retrieval=strict_retrieval,
                calls=calls)
            if expected_contract == PLAN_SCHEMA_V4 and row.get("status") == "completed":
                accepted_cycles = audit["probe_result_audit"]["accepted_judgement_cycles"]
                if strict_retrieval:
                    if not 1 <= accepted_cycles <= config["trace_config"]["max_rounds"]:
                        raise ValueError(
                            "task-routed v4 completed case must retain one or two accepted cycles")
                elif accepted_cycles != config["trace_config"]["max_rounds"]:
                    raise ValueError(
                        "fixed-reanalysis v4 completed case does not retain every forced cycle")
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
            if audit["v4_audit"] is not None:
                v4_cases.append({"id": case_id, **audit["v4_audit"]})
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
        structural.pop("v4_audit", None)
        structural_cases.append(structural)
        if result is not None:
            result_cases.append({"id": item["id"], "status": item["status"], **result})
    v4_totals = {name: Counter() for name in (
        "coverage_ledger", "material_probe_ledger", "strict_followups",
        "retrieval_attribution", "probe_delta_attribution",
        "live_artifact_receipts")}
    for case in v4_cases:
        for name, counter in v4_totals.items():
            counter.update(case[name])
    return {"totals": dict(totals), "cases": structural_cases,
            "v4_audit": {"applicable_cases": len(v4_cases),
                "totals": {name: dict(values) for name, values in v4_totals.items()},
                "cases": v4_cases,
                "meaning": ("Independent artifact joins: recomputed target segmentation, "
                            "material finding ledgers, unresolved-probe follow-ups, and "
                            "provider task/probe-delta attribution. For strict v4 runs, retained "
                            "request digests and raw user JSON additionally prove that exact loop "
                            "receipts reached each bounded model stage. Complete ledgers are not evidence "
                            "that a claim is true.")},
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
    task_routed = configs["extension"].get("experiment") == V4_TASK_ROUTED_EXPERIMENT
    for key in ("model", "reasoning_effort", "budget_per_case", "input_sha256",
                "dataset_status"):
        values = [configs[arm].get(key) for arm in runs]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"Runs have unequal {key}")
    provider_values = {arm: configs[arm].get("provider") for arm in runs}
    if task_routed:
        if (provider_values["original"] != PROVIDER or
                provider_values["staged"] != PROVIDER or
                provider_values["extension"] != TASK_ROUTED_PROVIDER):
            raise ValueError("Task-routed provider difference is not preregistered")
    elif len(set(provider_values.values())) != 1:
        raise ValueError("Runs have unequal provider")
    material_exposure_comparable = not task_routed
    if (not task_routed and
            configs["extension"]["trace_config"] != configs["staged"]["trace_config"]):
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
                staged_report = _read(runs[arm] / f"{case_id}-report.json")
                if task_routed and arm == "extension":
                    if (staged_report.get("target") != selected_inputs[case_id]["target"] or
                            staged_report.get("config") != configs[arm]["trace_config"]):
                        raise ValueError("Task-routed report changes the frozen task or trace contract")
                    verification_calls = staged_report.get("usage", {}).get(
                        "verification_calls")
                    if (type(verification_calls) is not int or
                            not 1 <= verification_calls <=
                            configs[arm]["trace_config"]["max_rounds"]):
                        raise ValueError("Task-routed report violates adaptive verification bounds")
                    # Reuse the common native-report checks on copies with the
                    # old forced-round scalar normalized to the observed count.
                    # The immutable config equality and adaptive bounds were
                    # checked above against the original artifacts.
                    audit_config = deepcopy(configs[arm])
                    audit_report = deepcopy(staged_report)
                    audit_config["trace_config"]["max_rounds"] = verification_calls
                    audit_report["config"]["max_rounds"] = verification_calls
                    normalized = _staged(row, audit_report, audit_config,
                                         selected_inputs[case_id], eligible[case_id])
                else:
                    normalized = _staged(row, staged_report, configs[arm],
                                         selected_inputs[case_id], eligible[case_id])
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
        extension_cross_arm = task_routed and "extension" in {baseline, candidate}
        label_pairs[name]["material_exposure_comparable"] = not extension_cross_arm
        label_pairs[name]["accuracy_interpretation"] = (
            "descriptive_counts_only_not_a_fair_or_causal_accuracy_comparison"
            if extension_cross_arm else "paired_descriptive_development_comparison")
        if extension_cross_arm:
            for denominator in ("all_scheduled", "shared_completed"):
                label_pairs[name][denominator]["candidate_minus_baseline"] = None
                label_pairs[name][denominator]["delta_withheld_reason"] = (
                    "different_preregistered_material_exposure")

    return {
        "schema_version": "target-extension-compare-score-v2",
        "comparison": "target_extension_vs_staged_and_original_development",
        "selected_case_ids": selected_ids,
        "scheduled_cases_per_arm": len(selected_ids),
        "model": configs["extension"]["model"],
        "reasoning_effort": configs["extension"]["reasoning_effort"],
        "material_exposure_comparable": material_exposure_comparable,
        "cross_arm_extension_accuracy_comparable": material_exposure_comparable,
        "primary_comparison": ("within_extension_round1_to_final_and_v4_ledgers"
                               if task_routed else
                               "paired_cross_arm_development_diagnostic"),
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
        ] + ([
            "The task-routed extension and fixed-reanalysis baselines have different material exposure. Cross-arm fix/break counts are descriptive only; no fair or causal accuracy delta is reported. The primary evaluation is the extension's within-run round-1-to-final change plus independently audited v4 ledgers."
        ] if task_routed else []),
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
