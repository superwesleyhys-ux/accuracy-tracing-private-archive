"""Offline target-extension comparison tests; no model or credentials are used."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import extended_semantic as runtime_contract
from target_extension_compare_score import (CUE_DIMENSION_KINDS, PLAN_NOTES_V4,
                                             PLAN_SCHEMA_V2, PLAN_SCHEMA_V3, PLAN_SCHEMA_V4,
                                             TASK_ROUTED_PROVIDER,
                                             V4_FIXED_EXPERIMENT,
                                             V4_TASK_ROUTED_EXPERIMENT,
                                             V4_PROBE_ROUTES_AND_GATES,
                                             _active_retrieval_tasks_by_round,
                                             _audit_plan,
                                             _audit_probe_delta_attribution,
                                             _audit_retrieval_attribution,
                                             _audit_world_provenance_probe,
                                             _canonical_decision_impact,
                                             _canonical_digest, _extension_schedule,
                                             _canonical_probe_question,
                                             _history, _loop_counts, _plan_coverage,
                                             _required_probe_bindings,
                                             _target_segments_v4, score)


class TargetExtensionCompareTests(unittest.TestCase):
    GOLD = ROOT / "experiments" / "proof_pilot" / "gold.json"
    ORIGINAL = ROOT / "reports" / "staged-baseline-01"
    STAGED = ROOT / "reports" / "staged-dev-04"
    EXTENSION_SMOKE = ROOT / "reports" / "target-extension-smoke-01"

    @staticmethod
    def _stabilize_v2_ids(target, claims, probes):
        """Give hand-built fixtures the same structural IDs as TargetPlanner."""
        signature = _canonical_digest(target)
        claim_map = {}
        for claim in claims:
            old = claim["id"]
            predicate = next(item for item in claim["dimensions"]
                             if item["kind"] == "predicate")["anchor"]
            key = (claim["anchor"]["start"], claim["anchor"]["end"], claim["role"],
                   predicate["start"], predicate["end"])
            claim["id"] = target["id"] + ":claim:" + _canonical_digest(
                [signature, *key])[:20]
            claim_map[old] = claim["id"]
        dimension_map = {}
        for claim in claims:
            if claim["parent_claim_id"] is not None:
                claim["parent_claim_id"] = claim_map[claim["parent_claim_id"]]
            for dimension in claim["dimensions"]:
                old = dimension["id"]
                anchor = dimension["anchor"]
                dimension["id"] = target["id"] + ":dimension:" + _canonical_digest([
                    signature, claim["id"], dimension["kind"],
                    anchor["start"], anchor["end"]])[:20]
                dimension_map[old] = dimension["id"]
        for probe in probes:
            probe["claim_id"] = claim_map[probe["claim_id"]]
            probe["dimension_ids"] = [dimension_map[item]
                                      for item in probe["dimension_ids"]]
            probe["id"] = target["id"] + ":probe:" + _canonical_digest([
                signature, probe["claim_id"], probe["kind"],
                *sorted(probe["dimension_ids"]),
            ])[:20]

    @staticmethod
    def _plan(version=1):
        plan_schema = {2: PLAN_SCHEMA_V2, 3: PLAN_SCHEMA_V3}.get(version)
        target = {"id": "c1", "text": "Alpha reported 3 items.",
                  "as_of": "2026-09-05T00:00:00Z", "source_version_id": "m1",
                  "assessment_mode": "evidence", "evidence_scope": ["m1"]}
        claim = {"id": "c1:claim:one", "statement": target["text"],
                 "anchor": {"start": 0, "end": len(target["text"]), "quote": target["text"]},
                 "role": "main", "dimensions": [
                     {"id": "c1:dimension:subject", "kind": "subject",
                      "anchor": {"start": 0, "end": 5, "quote": "Alpha"}},
                     {"id": "c1:dimension:predicate", "kind": "predicate",
                      "anchor": {"start": 6, "end": 14, "quote": "reported"}},
                     {"id": "c1:dimension:quantity", "kind": "quantity_unit",
                      "anchor": {"start": 15, "end": 22, "quote": "3 items"}},
                 ], "parent_claim_id": None}
        probe_specs = (
            ("semantic", "semantic_core", ["c1:dimension:subject", "c1:dimension:predicate"],
             ["atoms", "evidence", "world"] if plan_schema else ["atoms", "evidence"]),
            ("quantity", "quantity_unit", ["c1:dimension:quantity"],
             ["atoms", "evidence", "world"] if plan_schema else ["atoms", "evidence"]),
            ("lineage", "source_lineage", [], ["lineage", "world"]),
        )
        probes = []
        for name, kind, dimension_ids, routes in probe_specs:
            probe = {"id": "c1:probe:" + name, "claim_id": claim["id"], "kind": kind,
                     "dimension_ids": dimension_ids,
                     "question": name + "?", "decision_impact": name + " impact",
                     "routes": routes, "gate": ("provenance" if kind == "source_lineage"
                                                 else "always")}
            if plan_schema:
                probe["match_policy"] = "semantic_constraint"
            probes.append(probe)
        if plan_schema:
            TargetExtensionCompareTests._stabilize_v2_ids(target, [claim], probes)
        base = {"target_signature": _canonical_digest(target), "logic": "single",
                "claims": [claim], "probes": probes, "notes": ""}
        if plan_schema:
            base = {"schema_version": plan_schema, **base}
        checksum = _canonical_digest(base)
        projections = {}
        for stage in ("atoms", "lineage", "critic", "evidence", "world"):
            selected = probes if stage == "critic" else [p for p in probes if stage in p["routes"]]
            projections[stage] = {"target_signature": base["target_signature"],
                "plan_sha256": checksum, "stage": stage,
                "claims": [claim] if selected else [], "probes": selected}
            if plan_schema:
                projections[stage] = {"schema_version": plan_schema,
                    **projections[stage], "logic": "single", "notes": base["notes"]}
        return target, {**base, "sha256": checksum, "projections": projections}

    @staticmethod
    def _v4_plan():
        """Build one canonical offline v4 artifact with every new scalar kind."""
        target = {"id": "v4", "text":
                  "Agency delivered 3 parcels in Paris on 5 September 2025.",
                  "as_of": "2026-09-05T00:00:00Z", "source_version_id": "m1",
                  "assessment_mode": "world", "evidence_scope": ["m1"]}

        class Client:
            def call(self, system, user, schema):
                payload = json.loads(user)
                if system == runtime_contract.CLAIM_CONTRACT_PROMPT:
                    return {"claims": [{"statement": target["text"],
                        "quote": target["text"], "role": "main", "dimensions": [
                            {"kind": "subject", "quote": "Agency"},
                            {"kind": "predicate", "quote": "delivered"},
                            {"kind": "actor_role", "quote": "Agency"},
                            {"kind": "quantity_unit", "quote": "3 parcels"},
                            {"kind": "entity_identity", "quote": "parcels"},
                            {"kind": "location", "quote": "in Paris"},
                            {"kind": "time", "quote": "on 5 September 2025"},
                        ]}], "logic": "single", "notes": ""}
                claim = payload["claim_contract"]["claims"][0]
                converted_dimensions = tuple(runtime_contract.ClaimDimension(
                    item["id"], item["kind"],
                    runtime_contract.TextAnchor(**item["anchor"]))
                    for item in claim["dimensions"])
                converted_claim = runtime_contract.TargetClaim(
                    claim["id"], claim["statement"],
                    runtime_contract.TextAnchor(**claim["anchor"]), claim["role"],
                    converted_dimensions, claim["parent_claim_id"])
                bindings = runtime_contract._required_probe_bindings(
                    converted_claim, target["assessment_mode"], "single")
                probes = [{"claim_id": claim["id"], "kind": kind,
                    "dimension_ids": list(dimension_ids),
                    "question": "placeholder?", "decision_impact": "placeholder"}
                    for kind, dimension_ids in bindings]

                dimensions = {item["id"]: item for item in claim["dimensions"]}
                ledger = []
                for segment in payload["coverage_segments"]:
                    overlapping = [item for item in dimensions.values()
                        if max(segment["anchor"]["start"], item["anchor"]["start"]) <
                        min(segment["anchor"]["end"], item["anchor"]["end"])]
                    allowed = runtime_contract.CUE_DIMENSION_KINDS.get(
                        segment["cue_kind"])
                    compatible = [item for item in overlapping
                                  if allowed is None or item["kind"] in allowed]
                    if segment["high_signal"]:
                        status = ("covered_by_dimension" if compatible
                                  else "suspected_missing")
                        bound = compatible
                    elif segment["cue_kind"] == "logic_connector":
                        status, bound = "logic_connector", []
                    elif overlapping:
                        status, bound = "covered_by_dimension", overlapping
                    else:
                        status, bound = "context_only", []
                    ledger.append({"segment_id": segment["id"], "status": status,
                                   "dimension_ids": [item["id"] for item in bound]})
                return {"decision": "accept", "repair_quote": "",
                        "repair_issue": "", "probes": probes,
                        "coverage_ledger": ledger, "notes": ""}

        plan = runtime_contract.TargetPlanner(Client()).prepare(target)
        artifact = plan.to_payload()
        artifact["projections"] = {
            stage: plan.projection(stage).to_payload()
            for stage in ("atoms", "lineage", "critic", "evidence", "world")}
        return target, artifact

    @staticmethod
    def _manual_v4_plan(target, claim_specs, logic="single", ledger_override=None):
        """Build deterministic v4 fixtures without invoking planner validation."""
        signature = _canonical_digest(target)
        claims = []
        for spec in claim_specs:
            quote = spec["quote"]
            claim_start = spec.get("start", target["text"].index(quote))
            claim_anchor = {"start": claim_start, "end": claim_start + len(quote),
                            "quote": quote}
            raw_dimensions = []
            for index, dimension_spec in enumerate(spec["dimensions"]):
                kind, dimension_quote = dimension_spec[:2]
                relative = (dimension_spec[2] if len(dimension_spec) == 3 else
                            quote.index(dimension_quote))
                start = claim_start + relative
                raw_dimensions.append({"name": str(index), "kind": kind,
                    "anchor": {"start": start, "end": start + len(dimension_quote),
                               "quote": dimension_quote}})
            predicate = next(item for item in raw_dimensions
                             if item["kind"] == "predicate")["anchor"]
            claim_id = target["id"] + ":claim:" + _canonical_digest([
                signature, claim_anchor["start"], claim_anchor["end"], spec["role"],
                predicate["start"], predicate["end"],
            ])[:20]
            dimensions = []
            for raw in raw_dimensions:
                anchor = raw["anchor"]
                dimensions.append({"id": target["id"] + ":dimension:" +
                    _canonical_digest([signature, claim_id, raw["kind"],
                                       anchor["start"], anchor["end"]])[:20],
                    "kind": raw["kind"], "anchor": anchor})
            claims.append({"id": claim_id, "statement": quote, "anchor": claim_anchor,
                           "role": spec["role"], "dimensions": sorted(dimensions,
                               key=lambda item: (item["anchor"]["start"],
                                                 item["anchor"]["end"],
                                                 item["kind"], item["id"])),
                           "parent_claim_id": spec.get("parent")})
        for claim in claims:
            if type(claim["parent_claim_id"]) is int:
                claim["parent_claim_id"] = claims[claim["parent_claim_id"]]["id"]
        claims.sort(key=lambda item: (item["anchor"]["start"], item["anchor"]["end"],
                                      item["role"], item["id"]))
        claims_by_id = {claim["id"]: claim for claim in claims}
        probes = []
        for claim in claims:
            dimensions_by_id = {item["id"]: item for item in claim["dimensions"]}
            bindings = _required_probe_bindings(
                claim, target["assessment_mode"], logic, plan_schema=PLAN_SCHEMA_V4)
            for kind, dimension_ids in bindings:
                dimension_ids = tuple(sorted(dimension_ids))
                probe_id = target["id"] + ":probe:" + _canonical_digest([
                    signature, claim["id"], kind, *dimension_ids])[:20]
                routes, gate = V4_PROBE_ROUTES_AND_GATES[kind]
                probes.append({"id": probe_id, "claim_id": claim["id"], "kind": kind,
                    "dimension_ids": list(dimension_ids),
                    "question": _canonical_probe_question(
                        claim, kind, dimension_ids, dimensions_by_id, claims_by_id),
                    "decision_impact": _canonical_decision_impact(kind),
                    "match_policy": ({"entity_identity": "same_referent",
                                      "exact_designation": "exact_designation"}.get(
                                          kind, "semantic_constraint")),
                    "routes": list(routes), "gate": gate})
        probes.sort(key=lambda item: (item["claim_id"], item["kind"],
                                      tuple(item["dimension_ids"]), item["id"]))
        dimensions = {item["id"]: item for claim in claims
                      for item in claim["dimensions"]}
        ledger = []
        for segment in _target_segments_v4(target, signature, claims):
            overlapping = [item for item in dimensions.values()
                           if max(segment["anchor"]["start"], item["anchor"]["start"]) <
                           min(segment["anchor"]["end"], item["anchor"]["end"])]
            allowed = CUE_DIMENSION_KINDS.get(segment["cue_kind"])
            compatible = [item for item in overlapping
                          if allowed is None or item["kind"] in allowed]
            if compatible:
                status, bound = "covered_by_dimension", compatible
            elif segment["cue_kind"] == "logic_connector":
                status, bound = "logic_connector", []
            elif segment["cue_kind"] == "context" and not segment["high_signal"]:
                status, bound = "context_only", []
            elif (segment["cue_kind"] == "lexical_content" and
                  (logic in {"conditional", "comparison", "causal"} or
                   claims_by_id[segment["claim_id"]]["role"] == "attributed_content")):
                status, bound = "covered_by_relation", []
            else:
                status, bound = "suspected_missing", []
            entry = {**segment, "status": status,
                     "dimension_ids": sorted(item["id"] for item in bound)}
            if ledger_override is not None:
                entry = ledger_override(entry, dimensions) or entry
            ledger.append(entry)
        ledger.sort(key=lambda item: (item["claim_id"], item["anchor"]["start"],
                                      item["anchor"]["end"], item["cue_kind"],
                                      item["segment_id"]))
        plan = {"schema_version": PLAN_SCHEMA_V4, "target_signature": signature,
                "logic": logic, "claims": claims, "probes": probes,
                "coverage_ledger": ledger, "notes": PLAN_NOTES_V4,
                "sha256": "", "projections": {}}
        return TargetExtensionCompareTests._rehash(plan)

    @staticmethod
    def _rehash(plan):
        fields = ["target_signature", "logic", "claims", "probes", "notes"]
        if plan.get("schema_version") == PLAN_SCHEMA_V4:
            fields.insert(-1, "coverage_ledger")
        payload = {key: plan[key] for key in fields}
        plan_schema = plan.get("schema_version")
        if plan_schema in {PLAN_SCHEMA_V2, PLAN_SCHEMA_V3, PLAN_SCHEMA_V4}:
            payload = {"schema_version": plan_schema, **payload}
        plan["sha256"] = _canonical_digest(payload)
        for stage in ("atoms", "lineage", "critic", "evidence", "world"):
            selected = (plan["probes"] if stage == "critic" else
                        [probe for probe in plan["probes"] if stage in probe["routes"]])
            claim_ids = {probe["claim_id"] for probe in selected}
            claims = (plan["claims"] if stage == "critic" else
                      [claim for claim in plan["claims"] if claim["id"] in claim_ids])
            view = {"target_signature": plan["target_signature"],
                    "plan_sha256": plan["sha256"], "stage": stage,
                    "claims": claims, "probes": selected}
            if plan_schema in {PLAN_SCHEMA_V2, PLAN_SCHEMA_V3, PLAN_SCHEMA_V4}:
                view = {"schema_version": plan_schema, **view,
                        "logic": plan["logic"], "notes": plan["notes"]}
                if plan_schema == PLAN_SCHEMA_V4:
                    dimension_ids = {dimension["id"] for claim in claims
                                     for dimension in claim["dimensions"]}
                    view["coverage_ledger"] = [entry for entry in plan["coverage_ledger"]
                        if entry["claim_id"] in claim_ids or
                        set(entry["dimension_ids"]) & dimension_ids]
            plan["projections"][stage] = view
        return plan

    @staticmethod
    def _probe_result(probe, stage, basis, status="supported"):
        return {"probe_id": probe["id"], "claim_id": probe["claim_id"],
                "stage": stage, "status": status, "basis": basis,
                "rationale": "checked against the exact source span",
                "referent_relation": "not_applicable"}

    @staticmethod
    def _canonical_identity_question(anchor, match_policy):
        quote = json.dumps(anchor["quote"], ensure_ascii=False)
        if match_policy == "same_referent":
            return ("Does the evidence identify the same referent as " + quote +
                    " by exact mention, alias, unambiguous description or anaphora?")
        if match_policy == "exact_designation":
            return ("Does the evidence establish " + quote +
                    " as the exact asserted name, title, label or designation?")
        raise AssertionError("test fixture requested an identity question for another policy")

    @staticmethod
    def _identity_plan(kind="entity_identity", plan_schema=PLAN_SCHEMA_V2):
        target = {"id": "named", "text": "Alpha is called Item Three.",
                  "as_of": "2026-09-05T00:00:00Z", "source_version_id": "m1",
                  "assessment_mode": "evidence", "evidence_scope": ["m1"]}
        claim = {"id": "named:claim:one", "statement": target["text"],
                 "anchor": {"start": 0, "end": len(target["text"]),
                            "quote": target["text"]}, "role": "main",
                 "dimensions": [
                     {"id": "named:dimension:subject", "kind": "subject",
                      "anchor": {"start": 0, "end": 5, "quote": "Alpha"}},
                     {"id": "named:dimension:predicate", "kind": "predicate",
                      "anchor": {"start": 9, "end": 15, "quote": "called"}},
                     {"id": "named:dimension:identity", "kind": kind,
                      "anchor": {"start": 16, "end": 26, "quote": "Item Three"}},
                 ], "parent_claim_id": None}
        specs = [
            ("semantic", "semantic_core",
             ["named:dimension:subject", "named:dimension:predicate"]),
            ("identity", kind, ["named:dimension:identity"]),
            ("lineage", "source_lineage", []),
        ]
        if kind == "exact_designation":
            specs.insert(2, ("designation-relation", "designation_relation",
                [item["id"] for item in claim["dimensions"]]))
        probes = []
        for name, probe_kind, dimensions in specs:
            routes = (["lineage", "world"] if probe_kind == "source_lineage" else
                      ["atoms", "evidence", "world"])
            match_policy = ({"entity_identity": "same_referent",
                             "exact_designation": "exact_designation"}.get(
                                 probe_kind, "semantic_constraint"))
            question = name + "?"
            if (plan_schema == PLAN_SCHEMA_V3 and
                    probe_kind in {"entity_identity", "exact_designation"}):
                dimension = next(item for item in claim["dimensions"]
                                 if item["id"] == dimensions[0])
                question = TargetExtensionCompareTests._canonical_identity_question(
                    dimension["anchor"], match_policy)
            probes.append({"id": "named:probe:" + name, "claim_id": claim["id"],
                "kind": probe_kind, "dimension_ids": dimensions,
                "question": question, "decision_impact": name + " impact",
                "match_policy": match_policy,
                "routes": routes,
                "gate": "provenance" if probe_kind == "source_lineage" else "always"})
        TargetExtensionCompareTests._stabilize_v2_ids(target, [claim], probes)
        plan = {"schema_version": plan_schema,
                "target_signature": _canonical_digest(target), "logic": "single",
                "claims": [claim], "probes": probes, "notes": "",
                "sha256": "", "projections": {}}
        return target, TargetExtensionCompareTests._rehash(plan)

    @staticmethod
    def _renamed_plan(plan_schema=PLAN_SCHEMA_V2):
        target = {"id": "p07", "text":
                  "NOAA renamed GOES-U to GOES-19 on June 25, 2024.",
                  "as_of": "2026-09-05T00:00:00Z", "source_version_id": "m12",
                  "assessment_mode": "evidence", "evidence_scope": ["m13"]}
        text = target["text"]

        def dimension(name, kind, quote):
            start = text.index(quote)
            return {"id": "p07:dimension:" + name, "kind": kind,
                    "anchor": {"start": start, "end": start + len(quote),
                               "quote": quote}}

        dimensions = [
            dimension("subject", "subject", "NOAA"),
            dimension("predicate", "predicate", "renamed"),
            dimension("actor", "entity_identity", "NOAA"),
            dimension("old", "entity_identity", "GOES-U"),
            dimension("new", "exact_designation", "GOES-19"),
            dimension("time", "time", "June 25, 2024"),
        ]
        claim = {"id": "p07:claim:one", "statement": text,
                 "anchor": {"start": 0, "end": len(text), "quote": text},
                 "role": "main", "dimensions": dimensions,
                 "parent_claim_id": None}
        by_name = {item["id"].rsplit(":", 1)[-1]: item["id"]
                   for item in dimensions}
        specs = [
            ("semantic", "semantic_core",
             [by_name["subject"], by_name["predicate"]]),
            ("actor", "entity_identity", [by_name["actor"]]),
            ("old", "entity_identity", [by_name["old"]]),
            ("new", "exact_designation", [by_name["new"]]),
            ("time", "time_boundary", [by_name["time"]]),
            ("designation-relation", "designation_relation",
             [item["id"] for item in dimensions]),
            ("lineage", "source_lineage", []),
        ]
        probes = []
        for name, kind, dimension_ids in specs:
            match_policy = {"entity_identity": "same_referent",
                            "exact_designation": "exact_designation"}.get(
                                kind, "semantic_constraint")
            question = "Check " + name + "."
            if (plan_schema == PLAN_SCHEMA_V3 and
                    kind in {"entity_identity", "exact_designation"}):
                dimension = next(item for item in dimensions
                                 if item["id"] == dimension_ids[0])
                question = TargetExtensionCompareTests._canonical_identity_question(
                    dimension["anchor"], match_policy)
            probes.append({"id": "p07:probe:" + name, "claim_id": claim["id"],
                "kind": kind, "dimension_ids": dimension_ids,
                "question": question,
                "decision_impact": "A mismatch changes the decision.",
                "match_policy": match_policy,
                "routes": (["lineage", "world"] if kind == "source_lineage"
                           else ["atoms", "evidence", "world"]),
                "gate": "provenance" if kind == "source_lineage" else "always"})
        TargetExtensionCompareTests._stabilize_v2_ids(target, [claim], probes)
        plan = {"schema_version": plan_schema,
                "target_signature": _canonical_digest(target), "logic": "single",
                "claims": [claim], "probes": probes, "notes": "",
                "sha256": "", "projections": {}}
        return target, TargetExtensionCompareTests._rehash(plan)

    @staticmethod
    def _dual_time_plan():
        target = {"id": "p08", "text":
                  "The USGS Unified Geologic Map of the Moon released in 2020 combined "
                  "six Apollo-era regional maps with newer lunar-mission data.",
                  "as_of": "2026-09-05T00:00:00Z", "source_version_id": "m14",
                  "assessment_mode": "evidence", "evidence_scope": ["m15"]}
        text = target["text"]

        def dimension(name, kind, quote):
            start = text.index(quote)
            return {"id": "p08:dimension:" + name, "kind": kind,
                    "anchor": {"start": start, "end": start + len(quote),
                               "quote": quote}}

        dimensions = [
            dimension("subject", "subject", "The USGS Unified Geologic Map of the Moon"),
            dimension("predicate", "predicate", "combined"),
            dimension("apollo-era", "time", "Apollo-era"),
            dimension("newer", "time", "newer"),
        ]
        claim = {"id": "p08:claim:one", "statement":
                 "The USGS Unified Geologic Map of the Moon combined six Apollo-era "
                 "regional maps with newer lunar-mission data.",
                 "anchor": {"start": 0, "end": len(text), "quote": text},
                 "role": "main", "dimensions": dimensions,
                 "parent_claim_id": None}
        by_name = {item["id"].rsplit(":", 1)[-1]: item["id"]
                   for item in dimensions}
        specs = [
            ("semantic", "semantic_core",
             [by_name["subject"], by_name["predicate"]]),
            ("apollo-era", "time_boundary", [by_name["apollo-era"]]),
            ("newer", "time_boundary", [by_name["newer"]]),
            ("lineage", "source_lineage", []),
        ]
        probes = []
        for name, kind, dimension_ids in specs:
            probes.append({"id": "p08:probe:" + name, "claim_id": claim["id"],
                "kind": kind, "dimension_ids": dimension_ids,
                "question": "Check " + name + ".",
                "decision_impact": "A mismatch changes the decision.",
                "match_policy": "semantic_constraint",
                "routes": (["lineage", "world"] if kind == "source_lineage"
                           else ["atoms", "evidence", "world"]),
                "gate": "provenance" if kind == "source_lineage" else "always"})
        TargetExtensionCompareTests._stabilize_v2_ids(target, [claim], probes)
        plan = {"schema_version": PLAN_SCHEMA_V3,
                "target_signature": _canonical_digest(target), "logic": "single",
                "claims": [claim], "probes": probes, "notes": "",
                "sha256": "", "projections": {}}
        return target, TargetExtensionCompareTests._rehash(plan)

    def _report(self, plan, span, overrides=None):
        overrides = overrides or {}
        result = {"round": 1}
        for stage in ("evidence", "world"):
            values = []
            for probe in plan["probes"]:
                if stage not in probe["routes"]:
                    continue
                item = self._probe_result(probe, stage, [span])
                item.update(overrides.get((stage, probe["id"]), {}))
                values.append(item)
            result[stage + "_probe_results"] = values
        return [result]

    def _audit_v4_static(self, target, plan):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan))
            return _audit_plan(
                path, target, {"target_plan_sha256": plan["sha256"]}, [], [],
                expected_contract=PLAN_SCHEMA_V4)

    def _v4_dynamic_fixture(self):
        """Return a minimal completed task-routed v4 artifact join."""
        target, plan = self._v4_plan()
        content = target["text"]
        source_span = {"version_id": "m1", "start": 0, "end": len(content),
                       "quote": content}
        atoms_ids = [probe["id"] for probe in plan["probes"]
                     if "atoms" in probe["routes"]]
        lineage_ids = [probe["id"] for probe in plan["probes"]
                       if "lineage" in probe["routes"]]

        def checks(ids, origin=False):
            result = []
            for index, probe_id in enumerate(ids):
                uses_finding = index == 0 and not origin
                result.append({"probe_id": probe_id,
                    "status": "addressed" if uses_finding or origin else "absent",
                    "finding_indexes": [0] if uses_finding else [],
                    "origin_used": origin,
                    "rationale": "Mapped only to retained material findings."})
            return result

        psi = [
            {"sequence": 1, "material": "m1", "round": 1, "stage": "atoms",
             "repair": 0, "status": "accepted", "probe_checks": checks(atoms_ids)},
            {"sequence": 2, "material": "m1", "round": 1, "stage": "lineage",
             "repair": 0, "status": "accepted", "probe_checks": checks(lineage_ids, True)},
            {"sequence": 3, "material": "m1", "round": 1, "stage": "critic",
             "repair": 0, "status": "accepted"},
        ]
        verification = [{"sequence": 1, "material": None, "round": 1,
                         "stage": "judgement_critic", "repair": 0,
                         "status": "accepted"}]

        unresolved_probe = next(probe for probe in plan["probes"]
                                if probe["kind"] == "source_independence")
        record = {"round": 1, "gaps": [], "resolutions": [],
                  "strict_probe_followups": True,
                  "probe_stops": [{"probe_id": unresolved_probe["id"],
                      "stage": "world", "reason": "no_source_lead",
                      "rationale": "No independent-source locator is visible."}]}
        for stage in ("evidence", "world"):
            results = []
            for probe in plan["probes"]:
                if stage not in probe["routes"]:
                    continue
                unresolved = stage == "world" and probe["id"] == unresolved_probe["id"]
                relation = ("exact" if probe["kind"] == "entity_identity" and
                            not unresolved else
                            "unresolved" if probe["kind"] in
                            {"entity_identity", "exact_designation"} else "not_applicable")
                results.append({"probe_id": probe["id"],
                    "claim_id": probe["claim_id"], "stage": stage,
                    "status": "unresolved" if unresolved else "supported",
                    "basis": [] if unresolved else [source_span],
                    "rationale": "Checked against the exact retained source span.",
                    "referent_relation": relation})
            record[stage + "_probe_results"] = results

        origin = {"target_id": target["id"], "version_id": "m1"}
        origin_resolution = {"gap_id": "origin:" + target["id"],
                             "basis": [source_span],
                             "rationale": "The retained material is the terminal source."}
        analysis = {"fragments": [{"id": "fragment:one"}], "relations": [],
                    "gaps": [], "resolutions": [origin_resolution],
                    "origins": [origin], "notes": ""}
        task = {"id": "origin:" + target["id"],
                "question": "Find the producing record and evidenced lineage for: " +
                            target["text"],
                "stage": "provenance", "dimension": "auto", "blocking": True,
                "target_id": None, "basis": [], "decision_impact": "",
                "action": "search", "locator": None, "probe_id": None}
        retrieval = [{"round": 1, "kind": "task_routed_fixed_corpus",
                      "tasks": [task], "returned": ["m1"],
                      "attribution": [{"version_id": "m1",
                                       "task_ids": [task["id"]]}],
                      "feedback": []}]
        history_entry = {"revision": 1, "round": 1, "version_id": "m1",
                         "duplicate": False, "revisit": False, "accepted": True,
                         "analysis": analysis, "exclusion_reasons": [],
                         "trigger_task_ids": [task["id"]], "trigger_probe_ids": []}
        report = {"target": target, "retrieval_attribution_mode": "strict",
                  "materials": [{"version_id": "m1"}],
                  "eligible_version_ids": ["m1"], "analyses": {"m1": analysis},
                  "analysis_history": [history_entry], "relations": [],
                  "origins": [origin], "gaps": [], "gap_registry": [task],
                  "resolutions": [origin_resolution], "verification_history": [record],
                  "usage": {"rounds": 1, "verification_calls": 1},
                  "observations": [{"version_id": "m1", "duplicate": False,
                                    "eligible": True,
                                    "trigger_task_ids": [task["id"]],
                                    "trigger_probe_ids": []}],
                  "operations": [
                      {"sequence": 1, "round": 1, "action": "search",
                       "tasks": [task], "limit": 1},
                      {"sequence": 2, "round": 1,
                       "action": "retrieval_attribution_validated", "version_id": "m1",
                       "trigger_task_ids": [task["id"]], "trigger_probe_ids": []},
                      {"sequence": 3, "round": 1, "action": "snapshot_saved",
                       "version_id": "m1", "eligible": True},
                      {"sequence": 4, "round": 1, "action": "decompose_started",
                       "version_id": "m1", "trigger_task_ids": [task["id"]],
                       "trigger_probe_ids": []},
                      {"sequence": 5, "round": 1, "action": "decompose_completed",
                       "version_id": "m1", "trigger_task_ids": [task["id"]],
                       "trigger_probe_ids": []},
                      {"sequence": 6, "round": 1,
                       "action": "probe_followups_validated",
                       "active": [], "stops": record["probe_stops"],
                       "stopped_prior_gap_ids": [],
                       "superseded_prior_gaps": []},
                  ]}
        materials = [{"version_id": "m1", "content": content}]
        return target, plan, psi, verification, report, materials, retrieval

    def _audit_v4_dynamic(self, fixture):
        target, plan, psi, verification, report, materials, retrieval = fixture
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan))
            return _audit_plan(path, target,
                {"status": "completed", "target_plan_sha256": plan["sha256"]},
                psi, verification, expected_contract=PLAN_SCHEMA_V4,
                report_history=report["verification_history"], materials=materials,
                report=report, retrieval=retrieval, strict_retrieval=True)

    def test_v4_canonical_plan_covers_ledger_composition_actor_and_location(self):
        target, plan = self._v4_plan()
        result = self._audit_v4_static(target, plan)

        self.assertEqual(PLAN_SCHEMA_V4, result["plan_contract"])
        self.assertTrue(plan["coverage_ledger"])
        self.assertEqual(1, sum(probe["kind"] == "claim_composition"
                                for probe in plan["probes"]))
        for kind in ("actor_role", "location"):
            probe, = [item for item in plan["probes"] if item["kind"] == kind]
            self.assertEqual(1, len(probe["dimension_ids"]))

    def test_v4_rejects_rehashed_missing_or_forged_coverage_ledger(self):
        target, original = self._v4_plan()
        variants = {}
        missing = deepcopy(original)
        missing["coverage_ledger"].pop()
        variants["missing"] = missing
        forged = deepcopy(original)
        forged["coverage_ledger"][0]["segment_id"] = "v4:segment:forged"
        variants["forged"] = forged

        for name, candidate in variants.items():
            with self.subTest(name=name):
                self._rehash(candidate)
                with self.assertRaisesRegex(ValueError, "coverage|segment"):
                    self._audit_v4_static(target, candidate)

    def test_v4_rejects_rehashed_noncanonical_question_and_impact(self):
        target, original = self._v4_plan()
        variants = {}
        question = deepcopy(original)
        next(item for item in question["probes"]
             if item["kind"] == "entity_identity")["question"] = (
                 "Did the agency perform the delivery action?")
        variants["question"] = (question, "question.*canonical")
        impact = deepcopy(original)
        next(item for item in impact["probes"]
             if item["kind"] == "actor_role")["decision_impact"] = (
                 "A generic mismatch might matter.")
        variants["impact"] = (impact, "impact.*canonical")

        for name, (candidate, message) in variants.items():
            with self.subTest(name=name):
                self._rehash(candidate)
                with self.assertRaisesRegex(ValueError, message):
                    self._audit_v4_static(target, candidate)

    def test_v4_rejects_missing_composition_and_merged_actor_location_bindings(self):
        target, original = self._v4_plan()
        missing = deepcopy(original)
        missing["probes"] = [probe for probe in missing["probes"]
                             if probe["kind"] != "claim_composition"]
        self._rehash(missing)
        with self.assertRaisesRegex(ValueError, "composition|exactly cover|required"):
            self._audit_v4_static(target, missing)

        claim = original["claims"][0]
        extra_ids = {
            "actor_role": next(item["id"] for item in claim["dimensions"]
                               if item["kind"] == "predicate"),
            "location": next(item["id"] for item in claim["dimensions"]
                             if item["kind"] == "time"),
        }
        for kind, extra_id in extra_ids.items():
            with self.subTest(kind=kind):
                merged = deepcopy(original)
                probe = next(item for item in merged["probes"]
                             if item["kind"] == kind)
                probe["dimension_ids"] = sorted([*probe["dimension_ids"], extra_id])
                probe["id"] = target["id"] + ":probe:" + _canonical_digest([
                    merged["target_signature"], probe["claim_id"],
                    probe["kind"], *probe["dimension_ids"]])[:20]
                self._rehash(merged)
                with self.assertRaisesRegex(
                        ValueError, "singleton|binding|exactly cover|required"):
                    self._audit_v4_static(target, merged)

    def test_v4_rejects_rehashed_context_only_conjunct_omission(self):
        target = {"id": "conj", "text":
                  "Alpha launched Orion and Beta cancelled Nova.",
                  "as_of": "2026-09-05T00:00:00Z", "source_version_id": "m1",
                  "assessment_mode": "evidence", "evidence_scope": ["m1"]}

        def hide_missing(entry, _dimensions):
            if entry["status"] == "suspected_missing":
                entry.update(status="context_only", dimension_ids=[])
            return entry

        forged = self._manual_v4_plan(target, [{"quote": target["text"],
            "role": "main", "dimensions": [
                ("subject", "Alpha"), ("predicate", "launched")]}],
            ledger_override=hide_missing)
        with self.assertRaisesRegex(ValueError, "obvious conjunct|separately anchored predicate"):
            self._audit_v4_static(target, forged)

    def test_v4_month_has_time_precedence_and_rejects_rehashed_location(self):
        target = {"id": "month", "text": "Revenue rose in March.",
                  "as_of": "2026-09-05T00:00:00Z", "source_version_id": "m1",
                  "assessment_mode": "evidence", "evidence_scope": ["m1"]}
        correct = self._manual_v4_plan(target, [{"quote": target["text"],
            "role": "main", "dimensions": [
                ("subject", "Revenue"), ("predicate", "rose"), ("time", "March")]}])
        self.assertEqual(PLAN_SCHEMA_V4,
                         self._audit_v4_static(target, correct)["plan_contract"])
        date_entry, = [item for item in correct["coverage_ledger"]
                       if item["cue_kind"] == "date"]
        time_id, = date_entry["dimension_ids"]
        self.assertEqual("time", next(dimension["kind"] for dimension in
            correct["claims"][0]["dimensions"] if dimension["id"] == time_id))

        def forge_location(entry, dimensions):
            if entry["cue_kind"] == "date":
                location_id = next(identifier for identifier, dimension in dimensions.items()
                                   if dimension["kind"] == "location")
                entry.update(status="covered_by_dimension", dimension_ids=[location_id])
            return entry

        wrong = self._manual_v4_plan(target, [{"quote": target["text"],
            "role": "main", "dimensions": [
                ("subject", "Revenue"), ("predicate", "rose"),
                ("location", "in March")]}], ledger_override=forge_location)
        with self.assertRaisesRegex(ValueError, "compatible dimension"):
            self._audit_v4_static(target, wrong)

    def test_v4_recomputes_attribution_boundary_and_parent_orientation(self):
        proper_target = {"id": "said", "text": "NOAA said that temperatures rose.",
                         "as_of": "2026-09-05T00:00:00Z",
                         "source_version_id": "m1", "assessment_mode": "evidence",
                         "evidence_scope": ["m1"]}
        proper = self._manual_v4_plan(proper_target, [
            {"quote": "NOAA said that", "role": "attribution",
             "dimensions": [("subject", "NOAA"), ("predicate", "said")]},
            {"quote": "temperatures rose.", "role": "attributed_content", "parent": 0,
             "dimensions": [("subject", "temperatures"), ("predicate", "rose")]},
        ], logic="attribution")
        self.assertEqual(PLAN_SCHEMA_V4,
                         self._audit_v4_static(proper_target, proper)["plan_contract"])

        reversed_plan = self._manual_v4_plan(proper_target, [
            {"quote": "NOAA said that", "role": "attributed_content", "parent": 1,
             "dimensions": [("subject", "NOAA"), ("predicate", "said")]},
            {"quote": "temperatures rose.", "role": "attribution",
             "dimensions": [("subject", "temperatures"), ("predicate", "rose")]},
        ], logic="attribution")
        with self.assertRaisesRegex(ValueError, "reverse|reporting cue owner"):
            self._audit_v4_static(proper_target, reversed_plan)

        according = {"id": "according", "text":
                     "According to NOAA, temperatures rose.",
                     "as_of": "2026-09-05T00:00:00Z", "source_version_id": "m1",
                     "assessment_mode": "evidence", "evidence_scope": ["m1"]}

        def hide_missing(entry, _dimensions):
            if entry["status"] == "suspected_missing":
                entry.update(status="context_only", dimension_ids=[])
            return entry

        flattened = self._manual_v4_plan(according, [{"quote": according["text"],
            "role": "main", "dimensions": [
                ("subject", "temperatures"), ("predicate", "rose")]}],
            ledger_override=hide_missing)
        with self.assertRaisesRegex(ValueError, "explicit attribution|flattened"):
            self._audit_v4_static(according, flattened)

    def test_v4_dynamic_audits_join_findings_followups_and_task_returns(self):
        result = self._audit_v4_dynamic(self._v4_dynamic_fixture())
        audit = result["v4_audit"]
        self.assertEqual(audit["coverage_ledger"]["expected_segments"],
                         audit["coverage_ledger"]["ledger_entries"])
        self.assertEqual(2, audit["material_probe_ledger"]["stage_calls"])
        self.assertEqual(audit["material_probe_ledger"]["expected_probe_checks"],
                         audit["material_probe_ledger"]["probe_checks"])
        self.assertEqual(1, audit["material_probe_ledger"]["findings"])
        self.assertEqual(1, audit["material_probe_ledger"]["referenced_findings"])
        self.assertEqual(1, audit["strict_followups"]["unresolved_probe_slots"])
        self.assertEqual(1, audit["strict_followups"]["allowed_stops"])
        self.assertEqual(1, audit["retrieval_attribution"]["provider_returns"])
        self.assertEqual(1, audit["retrieval_attribution"]["valid_task_links"])

    def test_v4_rejects_material_probe_ledger_tampering(self):
        variants = {}
        missing = self._v4_dynamic_fixture()
        missing[2][0]["probe_checks"].pop()
        variants["missing projected check"] = (missing, "exactly cover")
        invalid_index = self._v4_dynamic_fixture()
        check = next(item for item in invalid_index[2][0]["probe_checks"]
                     if item["finding_indexes"])
        check["finding_indexes"] = [1]
        variants["invalid finding index"] = (invalid_index, "finding index")
        origin = self._v4_dynamic_fixture()
        origin[2][0]["probe_checks"][0]["origin_used"] = True
        variants["atoms origin"] = (origin, "atoms.*origin")
        no_lineage = self._v4_dynamic_fixture()
        no_lineage[2].pop(1)
        no_lineage[2][1]["sequence"] = 2
        variants["stage call is not coverage"] = (no_lineage, "lacks final atoms or lineage")
        for name, (fixture, message) in variants.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    self._audit_v4_dynamic(fixture)

    def test_v4_rejects_strict_followup_tampering(self):
        missing = self._v4_dynamic_fixture()
        missing[4]["verification_history"][0]["probe_stops"] = []
        with self.assertRaisesRegex(ValueError, "exactly one task or allowed stop"):
            self._audit_v4_dynamic(missing)

        duplicate = self._v4_dynamic_fixture()
        stop = duplicate[4]["verification_history"][0]["probe_stops"][0]
        duplicate[4]["verification_history"][0]["probe_stops"].append(deepcopy(stop))
        with self.assertRaisesRegex(ValueError, "repeats a probe slot"):
            self._audit_v4_dynamic(duplicate)

        invalid_stop = self._v4_dynamic_fixture()
        invalid_stop[4]["verification_history"][0]["probe_stops"][0]["reason"] = (
            "corpus_exhausted")
        with self.assertRaisesRegex(ValueError, "allowed unresolved-probe stop"):
            self._audit_v4_dynamic(invalid_stop)

    def test_v4_accepts_nonblocking_gap_and_rejects_gap_field_tampering(self):
        def fixture_with_gap():
            fixture = self._v4_dynamic_fixture()
            target, plan, _, _, report, materials, _ = fixture
            record = report["verification_history"][0]
            probe = next(item for item in plan["probes"]
                         if item["kind"] == "source_independence")
            source = {"version_id": "m1", "start": 0,
                      "end": len(materials[0]["content"]),
                      "quote": materials[0]["content"]}
            gap = {"id": "world-followup", "question": probe["question"],
                   "stage": "verification", "dimension": "world",
                   "blocking": False, "target_id": target["id"], "basis": [source],
                   "decision_impact": probe["decision_impact"], "action": "search",
                   "locator": "Agency independent parcels", "probe_id": probe["id"]}
            record.update(verdict="supported", evidence_verdict="supported",
                          world_verdict="supported", gaps=[gap], probe_stops=[])
            report["gaps"] = [gap]
            report["gap_registry"].append(gap)
            followup_event = next(item for item in report["operations"]
                                  if item["action"] == "probe_followups_validated")
            followup_event.update(
                active=[{"stage": "world", "probe_id": probe["id"],
                         "gap_id": gap["id"], "blocking": False}],
                stops=[], stopped_prior_gap_ids=[], superseded_prior_gaps=[])
            return fixture

        result = self._audit_v4_dynamic(fixture_with_gap())
        self.assertEqual(1, result["v4_audit"]["strict_followups"]["task_followups"])

        variants = {}
        blocking = fixture_with_gap()
        blocking[4]["verification_history"][0]["gaps"][0]["blocking"] = True
        variants["blocking"] = (blocking, "invalid probe ownership")
        question = fixture_with_gap()
        question[4]["verification_history"][0]["gaps"][0]["question"] = "Other question?"
        variants["question"] = (question, "canonical probe question")
        impact = fixture_with_gap()
        impact[4]["verification_history"][0]["gaps"][0]["decision_impact"] = (
            "Arbitrary impact.")
        variants["impact"] = (impact, "canonical probe question or impact")
        locator = fixture_with_gap()
        locator[4]["verification_history"][0]["gaps"][0]["locator"] = "banana aliens"
        variants["locator"] = (locator, "unrelated to its probe and basis")
        for name, (fixture, message) in variants.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    self._audit_v4_dynamic(fixture)

    def test_v4_rejects_round_one_basis_from_round_two_material(self):
        fixture = self._v4_dynamic_fixture()
        future_content = "Future-only corroborating record."
        fixture[5].append({"version_id": "m2", "content": future_content})
        fixture[4]["analysis_history"].append({
            "revision": 2, "round": 2, "version_id": "m2", "accepted": True})
        fixture[6].append({"round": 2, "returned": ["m2"]})
        result = next(item for item in
                      fixture[4]["verification_history"][0]["world_probe_results"]
                      if item["status"] == "supported" and
                      next(probe for probe in fixture[1]["probes"]
                           if probe["id"] == item["probe_id"])["kind"] == "actor_role")
        result["basis"] = [{"version_id": "m2", "start": 0,
                            "end": len(future_content), "quote": future_content}]
        with self.assertRaisesRegex(ValueError, "outside the stage evidence scope"):
            self._audit_v4_dynamic(fixture)

    def test_v4_world_source_probes_require_direct_independent_terminal_roots(self):
        target = {"id": "roots", "source_version_id": "seed"}
        report = {
            "materials": [{"version_id": item} for item in ("seed", "r1", "r2")],
            "eligible_version_ids": ["seed", "r1", "r2"],
            "origins": [{"target_id": "roots", "version_id": "r1"},
                        {"target_id": "roots", "version_id": "r2"}],
            "relations": [
                {"from_version": "seed", "to_version": "r1",
                 "kind": "cites", "status": "direct"},
                {"from_version": "seed", "to_version": "r2",
                 "kind": "derives", "status": "direct"},
            ],
        }

        def span(version_id):
            return {"version_id": version_id, "start": 0, "end": 1, "quote": "x"}

        independence = {"kind": "source_independence"}
        supported = {"status": "supported", "basis": [span("r1"), span("r2")]}
        self.assertIsNone(_audit_world_provenance_probe(
            independence, supported, report, target, "independence"))

        single = deepcopy(supported)
        single["basis"] = [span("r1")]
        with self.assertRaisesRegex(ValueError, "fewer than two sources"):
            _audit_world_provenance_probe(independence, single, report, target, "single")

        disconnected = deepcopy(report)
        disconnected["relations"].pop()
        with self.assertRaisesRegex(ValueError, "fewer than two sources"):
            _audit_world_provenance_probe(
                independence, supported, disconnected, target, "disconnected")

        copy_chain = deepcopy(report)
        copy_chain["materials"].extend([{"version_id": "a"}, {"version_id": "b"}])
        copy_chain["eligible_version_ids"].extend(["a", "b"])
        copy_chain["relations"].extend([
            {"from_version": "a", "to_version": "r1",
             "kind": "reprints", "status": "direct"},
            {"from_version": "b", "to_version": "a",
             "kind": "quotes", "status": "direct"},
        ])
        copied_basis = {"status": "supported", "basis": [span("a"), span("b")]}
        with self.assertRaisesRegex(ValueError, "shares one derivation component"):
            _audit_world_provenance_probe(
                independence, copied_basis, copy_chain, target, "copied")

        broken_support = deepcopy(copy_chain)
        broken_support["materials"].append({"version_id": "orphan"})
        broken_support["eligible_version_ids"].append("orphan")
        broken_basis = {"status": "supported",
                        "basis": [span("r1"), span("orphan")]}
        with self.assertRaisesRegex(ValueError, "derivation component"):
            _audit_world_provenance_probe(
                independence, broken_basis, broken_support, target, "broken")

        lineage = {"kind": "source_lineage"}
        no_path = deepcopy(report)
        no_path["relations"] = []
        with self.assertRaisesRegex(ValueError, "confirmed terminal path"):
            _audit_world_provenance_probe(
                lineage, {"status": "supported", "basis": [span("seed")]},
                no_path, target, "lineage")

    def test_v4_rejects_retrieval_attribution_and_outcome_tampering(self):
        variants = {}
        unknown = self._v4_dynamic_fixture()
        unknown[6][0]["attribution"][0]["task_ids"] = ["not-issued"]
        variants["unknown task link"] = (unknown, "issued-task attribution")
        mismatched = self._v4_dynamic_fixture()
        mismatched[4]["analysis_history"][0]["trigger_task_ids"] = ["not-issued"]
        variants["report cross-check"] = (mismatched, "disagrees across provider and report")
        for name, (fixture, message) in variants.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    self._audit_v4_dynamic(fixture)

        no_outcome = self._v4_dynamic_fixture()
        no_outcome[6][0]["returned"] = []
        no_outcome[6][0]["attribution"] = []
        with self.assertRaisesRegex(ValueError, "hit-or-exhausted"):
            _audit_retrieval_attribution(
                no_outcome[1], no_outcome[6], no_outcome[4], [(1, "m1")],
                strict=True, required=True)

    def test_v4_rejects_missing_reversed_or_mismatched_snapshot_chain(self):
        variants = {}
        missing = self._v4_dynamic_fixture()
        missing[4]["operations"] = [item for item in missing[4]["operations"]
                                    if item["action"] != "snapshot_saved"]
        for sequence, item in enumerate(missing[4]["operations"], 1):
            item["sequence"] = sequence
        variants["missing"] = missing

        reversed_chain = self._v4_dynamic_fixture()
        operations = reversed_chain[4]["operations"]
        saved = next(item for item in operations if item["action"] == "snapshot_saved")
        operations.remove(saved)
        completed_index = next(index for index, item in enumerate(operations)
                               if item["action"] == "decompose_completed")
        operations.insert(completed_index, saved)
        for sequence, item in enumerate(operations, 1):
            item["sequence"] = sequence
        variants["reversed"] = reversed_chain

        mismatched = self._v4_dynamic_fixture()
        next(item for item in mismatched[4]["operations"]
             if item["action"] == "snapshot_saved")["round"] = 2
        variants["wrong round"] = mismatched

        for name, fixture in variants.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "save/decompose|precede save"):
                    self._audit_v4_dynamic(fixture)

    def test_v4_two_round_active_replay_accepts_provenance_and_probe_tasks(self):
        target, plan = self._v4_plan()
        origin = {"id": "origin:" + target["id"],
                  "question": "Find the producing record and evidenced lineage for: " +
                              target["text"],
                  "stage": "provenance", "dimension": "auto", "blocking": True,
                  "target_id": None, "basis": [], "decision_impact": "",
                  "action": "search", "locator": None, "probe_id": None}
        source = {"version_id": "m1", "start": 0, "end": len(target["text"]),
                  "quote": target["text"]}
        locator = "Agency record"
        upstream_id = "m1:upstream:" + hashlib.sha256(json.dumps(
            [target["id"], "search", locator], sort_keys=True,
            ensure_ascii=False).encode()).hexdigest()[:20]
        upstream = {"id": upstream_id,
                    "question": "Locate the explicitly cited upstream: " + locator,
                    "stage": "provenance", "dimension": "provenance",
                    "blocking": True, "target_id": target["id"], "basis": [source],
                    "decision_impact": "The upstream record determines lineage.",
                    "action": "search", "locator": locator, "probe_id": None}
        probe = next(item for item in plan["probes"]
                     if item["kind"] == "source_independence")
        follow = {"id": "verify:test", "question": probe["question"],
                  "stage": "verification", "dimension": "world", "blocking": True,
                  "target_id": target["id"], "basis": [source],
                  "decision_impact": probe["decision_impact"], "action": "search",
                  "locator": "Agency independent record", "probe_id": probe["id"]}
        resolution = {"gap_id": origin["id"], "basis": [source],
                      "rationale": "The second record closes the origin task."}
        empty = {"fragments": [], "relations": [], "gaps": [], "resolutions": [],
                 "origins": [], "notes": ""}
        declared = {"id": "declared:one", "from_version": "m1",
                    "to_version": None, "kind": "cites", "status": "declared",
                    "basis": [source], "rationale": "The source names an upstream.",
                    "upstream_locator": locator}
        first_analysis = {**empty, "relations": [declared], "gaps": [upstream]}
        second_analysis = {**empty, "resolutions": [resolution]}
        first_record = {"round": 1, "verdict": "unresolved",
                        "evidence_verdict": "supported", "world_verdict": "unresolved",
                        "strict_probe_followups": True,
                        "evidence_probe_results": [],
                        "world_probe_results": [{"probe_id": probe["id"],
                                                  "status": "unresolved"}],
                        "gaps": [follow], "resolutions": [], "probe_stops": []}
        second_record = {"round": 2, "verdict": "supported",
                         "evidence_verdict": "supported", "world_verdict": "supported",
                         "strict_probe_followups": True,
                         "evidence_probe_results": [],
                         "world_probe_results": [{"probe_id": probe["id"],
                                                   "status": "supported"}],
                         "gaps": [], "resolutions": [], "probe_stops": []}
        round_one = {"round": 1, "kind": "task_routed_fixed_corpus",
                     "tasks": [origin], "returned": ["m1"],
                     "attribution": [{"version_id": "m1",
                                      "task_ids": [origin["id"]]}], "feedback": []}
        feedback = {"task_id": origin["id"], "probe_id": None,
                    "status": "corpus_exhausted", "detail": "origin already returned"}
        round_two_tasks = [origin, upstream, follow]
        round_two = {"round": 2, "kind": "task_routed_fixed_corpus",
                     "tasks": round_two_tasks, "returned": ["m2"],
                     "attribution": [{"version_id": "m2",
                                      "task_ids": [upstream["id"], follow["id"]]}],
                     "feedback": [feedback]}
        report = {"target": target, "retrieval_attribution_mode": "strict",
                  "analysis_history": [
                      {"revision": 1, "round": 1, "version_id": "m1",
                       "accepted": True, "duplicate": False, "revisit": False,
                       "analysis": first_analysis, "trigger_task_ids": [origin["id"]],
                       "trigger_probe_ids": []},
                      {"revision": 2, "round": 2, "version_id": "m2",
                       "accepted": True, "duplicate": False, "revisit": False,
                       "analysis": second_analysis,
                       "trigger_task_ids": [upstream["id"], follow["id"]],
                       "trigger_probe_ids": [probe["id"]]},
                      {"revision": 3, "round": 2, "version_id": "m1",
                       "accepted": True, "duplicate": True, "revisit": True,
                       "analysis": empty}],
                  "verification_history": [first_record, second_record],
                  "gap_registry": [origin, upstream, follow], "gaps": [],
                  "observations": [
                      {"version_id": "m1", "trigger_task_ids": [origin["id"]],
                       "trigger_probe_ids": []},
                      {"version_id": "m2",
                       "trigger_task_ids": [upstream["id"], follow["id"]],
                       "trigger_probe_ids": [probe["id"]]}],
                  "operations": []}
        operations = [
            {"round": 1, "action": "search", "tasks": [origin]},
            {"round": 1, "action": "retrieval_attribution_validated",
             "version_id": "m1", "trigger_task_ids": [origin["id"]],
             "trigger_probe_ids": []},
            {"round": 1, "action": "snapshot_saved", "version_id": "m1",
             "eligible": True},
            {"round": 1, "action": "decompose_started", "version_id": "m1",
             "trigger_task_ids": [origin["id"]], "trigger_probe_ids": []},
            {"round": 1, "action": "decompose_completed", "version_id": "m1",
             "trigger_task_ids": [origin["id"]], "trigger_probe_ids": []},
            {"round": 1, "action": "verification_started"},
            {"round": 1, "action": "probe_followups_validated",
             "active": [{"stage": "world", "probe_id": probe["id"],
                         "gap_id": follow["id"], "blocking": True}],
             "stops": [], "stopped_prior_gap_ids": [],
             "superseded_prior_gaps": []},
            {"round": 2, "action": "search", "tasks": round_two_tasks},
            {"round": 2, "action": "retrieval_attribution_validated",
             "version_id": "m2", "trigger_task_ids": [upstream["id"], follow["id"]],
             "trigger_probe_ids": [probe["id"]]},
            {"round": 2, "action": "snapshot_saved", "version_id": "m2",
             "eligible": True},
            {"round": 2, "action": "decompose_started", "version_id": "m2",
             "trigger_task_ids": [upstream["id"], follow["id"]],
             "trigger_probe_ids": [probe["id"]]},
            {"round": 2, "action": "decompose_completed", "version_id": "m2",
             "trigger_task_ids": [upstream["id"], follow["id"]],
             "trigger_probe_ids": [probe["id"]]},
            {"round": 2, "action": "decompose_started", "version_id": "m1"},
            {"round": 2, "action": "decompose_completed", "version_id": "m1"},
            {"round": 2, "action": "retrieval_feedback", "feedback": feedback},
            {"round": 2, "action": "verification_started"},
            {"round": 2, "action": "probe_followups_validated",
             "active": [], "stops": [], "stopped_prior_gap_ids": [],
             "superseded_prior_gaps": []},
        ]
        for sequence, item in enumerate(operations, 1):
            item["sequence"] = sequence
        report["operations"] = operations
        metrics = _audit_retrieval_attribution(
            plan, [round_one, round_two], report,
            [(1, "m1"), (2, "m2"), (2, "m1")], strict=True, required=True)
        self.assertEqual((1, 1), (metrics["later_probe_owned_tasks"],
                                  metrics["later_probe_owned_hit_tasks"]))

        stale = deepcopy(report)
        stale["analysis_history"].insert(1, {
            "revision": 2, "round": 1, "version_id": "m1", "accepted": True,
            "duplicate": True, "revisit": True, "analysis": empty})
        with self.assertRaisesRegex(ValueError, "replayed active gap set"):
            _audit_retrieval_attribution(
                plan, [round_one, round_two], stale,
                [(1, "m1"), (2, "m2"), (2, "m1")], strict=True, required=True)

    def test_v4_replays_exact_probe_slot_supersession_and_forbids_reissue(self):
        target = {"id": "supersede", "text": "Alpha changed.",
                  "source_version_id": "m1"}
        source = {"version_id": "m1", "start": 0, "end": 1, "quote": "A"}

        def gap(identifier, locator):
            return {"id": identifier, "question": "Did Alpha change?",
                    "stage": "verification", "dimension": "world",
                    "blocking": True, "target_id": target["id"], "basis": [source],
                    "decision_impact": "This decides the claim.", "action": "search",
                    "locator": locator, "probe_id": "probe:change"}

        old = gap("follow:old", "initial source lead")
        replacement = gap("follow:new", "better source lead")
        result = {"probe_id": "probe:change", "status": "unresolved"}
        empty_results = {"evidence_probe_results": [],
                         "world_probe_results": [result],
                         "world_verdict": "unresolved",
                         "strict_probe_followups": True,
                         "resolutions": [], "probe_stops": []}
        first = {"round": 1, **empty_results, "gaps": [old]}
        second = {"round": 2, **empty_results, "gaps": [replacement]}
        origin_resolution = {"gap_id": "origin:" + target["id"],
                             "basis": [source], "rationale": "Located."}
        analysis = {"gaps": [], "resolutions": [origin_resolution],
                    "relations": [],
                    "origins": [{"target_id": target["id"], "version_id": "m1"}]}
        supersession = [{"stage": "world", "probe_id": "probe:change",
                         "prior_gap_id": old["id"],
                         "replacement_gap_id": replacement["id"]}]
        report = {
            "analysis_history": [{"round": 1, "version_id": "m1",
                                  "accepted": True, "analysis": analysis}],
            "verification_history": [first, second],
            "operations": [
                {"sequence": 1, "round": 1,
                 "action": "probe_followups_validated",
                 "active": [{"stage": "world", "probe_id": "probe:change",
                             "gap_id": old["id"], "blocking": True}],
                 "stops": [], "stopped_prior_gap_ids": [],
                 "superseded_prior_gaps": []},
                {"sequence": 2, "round": 2,
                 "action": "probe_followups_validated",
                 "active": [{"stage": "world", "probe_id": "probe:change",
                             "gap_id": replacement["id"], "blocking": True}],
                 "stops": [], "stopped_prior_gap_ids": [],
                 "superseded_prior_gaps": supersession},
            ],
        }
        expected, final_active = _active_retrieval_tasks_by_round(report, target, 3)
        self.assertEqual([old["id"]], [item["id"] for item in expected[2]])
        self.assertEqual([replacement["id"]], [item["id"] for item in expected[3]])
        self.assertEqual([replacement["id"]], [item["id"] for item in final_active])

        forged = deepcopy(report)
        forged["operations"][1]["superseded_prior_gaps"][0]["prior_gap_id"] = (
            "follow:forged")
        with self.assertRaisesRegex(ValueError, "supersession ledger disagrees"):
            _active_retrieval_tasks_by_round(forged, target, 3)

        missing = deepcopy(report)
        missing["operations"].pop()
        with self.assertRaisesRegex(ValueError, "exact verifier supersession events"):
            _active_retrieval_tasks_by_round(missing, target, 3)

        reissued = deepcopy(report)
        reissued["verification_history"].append(
            {"round": 3, **empty_results, "gaps": [old]})
        reissued["operations"].append(
            {"sequence": 3, "round": 3, "action": "probe_followups_validated",
             "active": [{"stage": "world", "probe_id": "probe:change",
                         "gap_id": old["id"], "blocking": True}],
             "stops": [], "stopped_prior_gap_ids": [],
             "superseded_prior_gaps": [{
                 "stage": "world", "probe_id": "probe:change",
                 "prior_gap_id": replacement["id"],
                 "replacement_gap_id": old["id"],
             }]})
        with self.assertRaisesRegex(ValueError, "reissues a superseded"):
            _active_retrieval_tasks_by_round(reissued, target, 3)

    def test_v4_probe_delta_requires_probe_owned_novel_receipt_and_decisive_change(self):
        target, plan = self._v4_plan()
        probe = next(item for item in plan["probes"] if item["kind"] == "actor_role")
        m1 = {"version_id": "m1", "start": 0, "end": 1, "quote": "a"}
        m2 = {"version_id": "m2", "start": 0, "end": 1, "quote": "b"}

        def result(status, basis):
            return {"probe_id": probe["id"], "status": status, "basis": basis,
                    "referent_relation": "not_applicable"}

        history = [
            {"round": 1, "verdict": "supported", "world_verdict": "supported",
             "evidence_probe_results": [],
             "world_probe_results": [result("supported", [m1])]},
            {"round": 2, "verdict": "contradicted", "world_verdict": "contradicted",
             "evidence_probe_results": [],
             "world_probe_results": [result("contradicted", [m2])]},
        ]
        origin = {"id": "origin", "probe_id": None, "action": "search",
                  "dimension": "provenance"}
        task = {"id": "probe-task", "probe_id": probe["id"], "action": "search",
                "dimension": "world"}
        retrieval = [
            {"round": 1, "tasks": [origin], "returned": ["m1"],
             "attribution": [{"version_id": "m1", "task_ids": ["origin"]}]},
            {"round": 2, "tasks": [task], "returned": ["m2"],
             "attribution": [{"version_id": "m2", "task_ids": ["probe-task"]}]},
        ]
        empty = {"relations": [], "origins": []}
        report = {"analysis_history": [
            {"round": 1, "version_id": "m1", "accepted": True,
             "revisit": False, "analysis": empty,
             "trigger_task_ids": ["origin"], "trigger_probe_ids": []},
            {"round": 2, "version_id": "m2", "accepted": True,
             "revisit": False, "analysis": empty,
             "trigger_task_ids": ["probe-task"], "trigger_probe_ids": [probe["id"]]},
        ]}
        row = {"prediction": "false", "checkpoints": [
            {"round": 1, "decision": "true"}, {"round": 2, "decision": "false"}]}
        psi = [{"round": 2, "material": "m2", "stage": "atoms",
                "status": "accepted", "probe_checks": [
                    {"probe_id": probe["id"], "status": "addressed"}]}]
        metrics = _audit_probe_delta_attribution(
            plan, history, retrieval, report, target, row, strict=True, required=True,
            psi=psi)
        self.assertEqual(1, metrics["semantic_deltas"])
        self.assertEqual(1, metrics["traced_semantic_deltas"])
        self.assertEqual(1, metrics["probe_owned_novel_second_pass_cases"])
        self.assertEqual(1, metrics["label_changes_with_decisive_delta"])

        no_trigger = deepcopy(retrieval)
        no_trigger[1]["tasks"][0]["probe_id"] = None
        with self.assertRaisesRegex(ValueError, "not attributable"):
            _audit_probe_delta_attribution(
                plan, history, no_trigger, report, target, row, strict=True, required=True,
                psi=psi)
        wrong_layer = deepcopy(retrieval)
        wrong_layer[1]["tasks"][0]["dimension"] = "evidence"
        with self.assertRaisesRegex(ValueError, "not attributable"):
            _audit_probe_delta_attribution(
                plan, history, wrong_layer, report, target, row, strict=True,
                required=True, psi=psi)
        old_basis = deepcopy(history)
        old_basis[1]["world_probe_results"][0]["basis"] = [m1]
        with self.assertRaisesRegex(ValueError, "not attributable"):
            _audit_probe_delta_attribution(
                plan, old_basis, retrieval, report, target, row, strict=True, required=True,
                psi=psi)

        absent = deepcopy(psi)
        absent[0]["probe_checks"][0]["status"] = "absent"
        with self.assertRaisesRegex(ValueError, "not attributable"):
            _audit_probe_delta_attribution(
                plan, history, retrieval, report, target, row, strict=True, required=True,
                psi=absent)

        drift = deepcopy(history)
        drift[1]["world_probe_results"][0]["status"] = "supported"
        drift[1]["world_verdict"] = drift[1]["verdict"] = "supported"
        drift_metrics = _audit_probe_delta_attribution(
            plan, drift, retrieval, report, target,
            {"prediction": "true", "checkpoints": [{"round": 1, "decision": "true"}]},
            strict=True, required=True, psi=psi)
        self.assertEqual((0, 1), (drift_metrics["semantic_deltas"],
                                  drift_metrics["basis_drifts"]))

        label_only = deepcopy(drift)
        with self.assertRaisesRegex(ValueError, "public label change|counterfactual"):
            _audit_probe_delta_attribution(
                plan, label_only, retrieval, report, target, row,
                strict=True, required=True, psi=psi)

        ineligible_target = {**target, "assessment_mode": "evidence",
                             "evidence_scope": ["m2"]}
        ineligible_history = [
            {"round": 1, "evidence_verdict": "unresolved",
             "evidence_probe_results": [result("unresolved", [])],
             "world_probe_results": []},
            {"round": 2, "evidence_verdict": "supported",
             "evidence_probe_results": [result("supported", [m2])],
             "world_probe_results": []},
        ]
        ineligible_retrieval = deepcopy(retrieval)
        ineligible_retrieval[1]["tasks"][0]["dimension"] = "evidence"
        ineligible_report = deepcopy(report)
        ineligible_report["analysis_history"][1]["accepted"] = False
        with self.assertRaisesRegex(ValueError, "aggregate disagrees"):
            _audit_probe_delta_attribution(
                plan, ineligible_history, ineligible_retrieval, ineligible_report,
                ineligible_target, {"prediction": "unverifiable", "checkpoints": []},
                strict=True, required=True, psi=psi)

    def test_v4_graph_probe_delta_can_trace_a_novel_lineage_path(self):
        target, plan = self._v4_plan()
        probe = next(item for item in plan["probes"]
                     if item["kind"] == "source_lineage")
        base_probe = next(item for item in plan["probes"]
                          if item["kind"] == "claim_composition")
        base = {"probe_id": base_probe["id"], "status": "supported",
                "basis": [{"version_id": "m1", "start": 0,
                           "end": 1, "quote": "a"}],
                "referent_relation": "not_applicable"}
        prior = {"probe_id": probe["id"], "status": "unresolved", "basis": [],
                 "referent_relation": "not_applicable"}
        final = {**prior, "status": "supported",
                 "basis": [{"version_id": "m2", "start": 0,
                            "end": 1, "quote": "b"}]}
        history = [
            {"round": 1, "verdict": "unresolved", "world_verdict": "unresolved",
             "evidence_probe_results": [], "world_probe_results": [base, prior]},
            {"round": 2, "verdict": "supported", "world_verdict": "supported",
             "evidence_probe_results": [], "world_probe_results": [base, final]},
        ]
        retrieval = [
            {"round": 1, "tasks": [{"id": "origin", "probe_id": None,
                                      "action": "search", "dimension": "provenance"}],
             "returned": ["m1"],
             "attribution": [{"version_id": "m1", "task_ids": ["origin"]}]},
            {"round": 2, "tasks": [{"id": "upstream", "probe_id": probe["id"],
                                      "action": "fetch", "dimension": "world"}],
             "returned": ["m2"],
             "attribution": [{"version_id": "m2", "task_ids": ["upstream"]}]},
        ]
        report = {"analysis_history": [
            {"round": 1, "version_id": "m1", "accepted": True,
             "revisit": False, "analysis": {"relations": [], "origins": []},
             "trigger_task_ids": ["origin"], "trigger_probe_ids": []},
            {"round": 2, "version_id": "m2", "accepted": True,
             "revisit": False, "analysis": {"relations": [], "origins": [
                 {"target_id": target["id"], "version_id": "m2"}]},
             "trigger_task_ids": ["upstream"], "trigger_probe_ids": []},
            {"round": 2, "version_id": "m1", "accepted": True,
             "revisit": True, "analysis": {"relations": [
                 {"from_version": "m1", "to_version": "m2", "kind": "cites",
                  "status": "direct"}], "origins": []}},
        ]}
        psi = [{"round": 2, "material": "m2", "stage": "lineage",
                "status": "accepted", "probe_checks": [
                    {"probe_id": probe["id"], "status": "addressed"}]}]
        metrics = _audit_probe_delta_attribution(
            plan, history, retrieval, report, target,
            {"prediction": "true", "checkpoints": [{"round": 1,
                                                       "decision": "unverifiable"}]},
            strict=True, required=True, psi=psi)
        self.assertEqual(1, metrics["graph_traced_semantic_deltas"])
        self.assertEqual(1, metrics["probe_owned_novel_second_pass_cases"])

        broken = deepcopy(report)
        broken["analysis_history"][-1]["analysis"]["relations"] = []
        with self.assertRaisesRegex(ValueError, "not attributable"):
            _audit_probe_delta_attribution(
                plan, history, retrieval, broken, target,
                {"prediction": "unverifiable", "checkpoints": []},
                strict=True, required=True, psi=psi)

        absent = deepcopy(psi)
        absent[0]["probe_checks"][0]["status"] = "absent"
        with self.assertRaisesRegex(ValueError, "not attributable"):
            _audit_probe_delta_attribution(
                plan, history, retrieval, report, target,
                {"prediction": "unverifiable", "checkpoints": []},
                strict=True, required=True, psi=absent)

        # A provenance-only hit may build the new graph while an unrelated
        # probe-owned hit arrives in the same round.  That combination must not
        # be credited to this frozen source-lineage slot.
        mixed_retrieval = deepcopy(retrieval)
        mixed_retrieval[1] = {
            "round": 2,
            "tasks": [
                {"id": "provenance-upstream", "probe_id": None,
                 "action": "fetch", "dimension": "provenance"},
                {"id": "probe-unrelated", "probe_id": probe["id"],
                 "action": "search", "dimension": "world"},
            ],
            "returned": ["m2", "m3"],
            "attribution": [
                {"version_id": "m2", "task_ids": ["provenance-upstream"]},
                {"version_id": "m3", "task_ids": ["probe-unrelated"]},
            ],
        }
        mixed_report = deepcopy(report)
        mixed_report["analysis_history"][1]["trigger_task_ids"] = [
            "provenance-upstream"]
        mixed_report["analysis_history"][1]["trigger_probe_ids"] = []
        mixed_report["analysis_history"].insert(2, {
            "round": 2, "version_id": "m3", "accepted": True,
            "revisit": False, "analysis": {"relations": [
                # This is a real graph delta, but it is only a side edge:
                # m3 is not reachable from the target m1.  Counting every edge
                # endpoint would falsely let this probe-owned return borrow the
                # target path that the provenance-only m2 return established.
                {"from_version": "m3", "to_version": "m2", "kind": "cites",
                 "status": "direct"}], "origins": []},
            "trigger_task_ids": ["probe-unrelated"],
            "trigger_probe_ids": [probe["id"]],
        })
        mixed_psi = deepcopy(psi) + [{
            "round": 2, "material": "m3", "stage": "lineage",
            "status": "accepted", "probe_checks": [
                {"probe_id": probe["id"], "status": "addressed"}],
        }]
        with self.assertRaisesRegex(ValueError, "not attributable"):
            _audit_probe_delta_attribution(
                plan, history, mixed_retrieval, mixed_report, target,
                {"prediction": "unverifiable", "checkpoints": []},
                strict=True, required=True, psi=mixed_psi)

    def test_v4_label_change_uses_frozen_and_or_novel_counterfactual(self):
        target = {"id": "counterfactual", "assessment_mode": "world",
                  "source_version_id": "m1", "evidence_scope": []}
        claims = [{"id": "claim:a"}, {"id": "claim:b"}]
        probes = [{"id": "probe:a", "claim_id": "claim:a", "kind": "actor_role",
                   "routes": ["atoms", "world"], "gate": "always"},
                  {"id": "probe:b", "claim_id": "claim:b", "kind": "actor_role",
                   "routes": ["atoms", "world"], "gate": "always"}]
        m1 = {"version_id": "m1", "start": 0, "end": 1, "quote": "a"}
        m2 = {"version_id": "m2", "start": 0, "end": 1, "quote": "b"}
        retrieval = [
            {"round": 1,
             "tasks": [{"id": "origin", "probe_id": None,
                        "action": "search", "dimension": "provenance"}],
             "returned": ["m1"],
             "attribution": [{"version_id": "m1", "task_ids": ["origin"]}]},
            {"round": 2,
             "tasks": [
                 {"id": "novel-a", "probe_id": "probe:a",
                  "action": "search", "dimension": "world"},
                 {"id": "reanalyse-b", "probe_id": "probe:b",
                  "action": "reanalyse", "dimension": "world"},
             ],
             "returned": ["m2", "m1"],
             "attribution": [
                 {"version_id": "m2", "task_ids": ["novel-a"]},
                 {"version_id": "m1", "task_ids": ["reanalyse-b"]},
             ]},
        ]
        report = {"analysis_history": [
            {"round": 1, "version_id": "m1", "accepted": True,
             "revisit": False, "analysis": {"relations": [], "origins": [],
                                               "interpretation": "old"},
             "trigger_task_ids": ["origin"], "trigger_probe_ids": []},
            {"round": 2, "version_id": "m2", "accepted": True,
             "revisit": False, "analysis": {"relations": [], "origins": []},
             "trigger_task_ids": ["novel-a"], "trigger_probe_ids": ["probe:a"]},
            {"round": 2, "version_id": "m1", "accepted": True,
             "revisit": False, "analysis": {"relations": [], "origins": [],
                                               "interpretation": "new"},
             "trigger_task_ids": ["reanalyse-b"],
             "trigger_probe_ids": ["probe:b"]},
        ]}
        psi = [
            {"round": 2, "material": "m2", "stage": "atoms", "status": "accepted",
             "probe_checks": [{"probe_id": "probe:a", "status": "addressed"}]},
            {"round": 2, "material": "m1", "stage": "atoms", "status": "accepted",
             "probe_checks": [{"probe_id": "probe:b", "status": "addressed"}]},
        ]

        def result(probe_id, status, basis):
            return {"probe_id": probe_id, "status": status, "basis": basis,
                    "referent_relation": "not_applicable"}

        def run(logic, prior, final, prior_verdict, final_verdict):
            plan = {"logic": logic, "claims": claims, "probes": probes}
            history = [
                {"round": 1, "world_verdict": prior_verdict,
                 "evidence_probe_results": [], "world_probe_results": [
                     result("probe:a", prior[0], [] if prior[0] == "unresolved" else [m1]),
                     result("probe:b", prior[1], [] if prior[1] == "unresolved" else [m1])]},
                {"round": 2, "world_verdict": final_verdict,
                 "evidence_probe_results": [], "world_probe_results": [
                     result("probe:a", final[0], [] if final[0] == "unresolved" else [m2]),
                     result("probe:b", final[1], [] if final[1] == "unresolved" else [m1])]},
            ]
            labels = {"supported": "true", "contradicted": "false",
                      "conflicting": "mixed", "unresolved": "unverifiable"}
            row = {"prediction": labels[final_verdict], "checkpoints": [
                {"round": 1, "decision": labels[prior_verdict]}]}
            return _audit_probe_delta_attribution(
                plan, history, retrieval, report, target, row,
                strict=True, required=True, psi=psi)

        # In both negative cases, the novel A delta is insufficient without B's
        # duplicate reanalysis, so the headline label change is not novel-driven.
        with self.assertRaisesRegex(ValueError, "counterfactual"):
            run("and", ("unresolved", "unresolved"),
                ("supported", "supported"), "unresolved", "supported")
        with self.assertRaisesRegex(ValueError, "counterfactual"):
            run("or", ("contradicted", "contradicted"),
                ("unresolved", "supported"), "contradicted", "supported")

        # Novel A alone determines these final AND/OR aggregates; B's changed
        # interpretation is retained diagnostically but is not causal credit.
        and_positive = run("and", ("supported", "supported"),
                           ("contradicted", "unresolved"),
                           "supported", "contradicted")
        or_positive = run("or", ("contradicted", "contradicted"),
                          ("supported", "unresolved"),
                          "contradicted", "supported")
        self.assertEqual(1, and_positive["label_changes_with_decisive_delta"])
        self.assertEqual(1, or_positive["label_changes_with_decisive_delta"])

    def test_retained_failed_smoke_uses_extension_subset_as_denominator(self):
        result = score(self.GOLD, self.ORIGINAL, self.STAGED, self.EXTENSION_SMOKE)
        self.assertEqual(["p02", "p04"], result["selected_case_ids"])
        self.assertEqual(2, result["scheduled_cases_per_arm"])
        self.assertEqual((2, 2, 0), tuple(result["arms"][arm]["completed"]
                         for arm in ("original", "staged", "extension")))
        self.assertEqual((2, 2, 0), tuple(result["arms"][arm]["correct"]
                         for arm in ("original", "staged", "extension")))
        paired = result["label_comparisons"]["extension_vs_staged"]["all_scheduled"]
        self.assertEqual((0, 2), (paired["fixes"], paired["breaks"]))
        self.assertEqual(344, result["arms"]["extension"]["usage_total"]["reasoning_tokens"])
        self.assertEqual(2, result["loop_audit"]["extension"]["totals"]["target_planner_calls"])
        coverage = result["target_plan_probe_coverage"]["totals"]
        self.assertEqual((0, 0), (coverage["plans_present"], coverage["plans_valid"]))
        self.assertIsNone(coverage["required_probe_coverage"])

    def test_valid_plan_reports_required_projection_and_stage_delivery_coverage(self):
        target, plan = self._plan()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan))
            result = _audit_plan(path, target, {"target_plan_sha256": plan["sha256"]},
                [{"stage": "atoms"}, {"stage": "lineage"}, {"stage": "critic"}],
                [{"stage": "evidence"}, {"stage": "world"},
                 {"stage": "judgement_critic", "status": "accepted"}])
        self.assertEqual((3, 3), (result["required_probe_slots"],
                                  result["required_probe_slots_covered"]))
        self.assertEqual((9, 9), (result["projection_slots"],
                                  result["projection_slots_covered"]))
        self.assertEqual((6, 6), (result["routed_stage_slots"],
                                  result["projected_slots_at_executed_stages"]))
        self.assertFalse(result["probe_result_audit"]["available"])

    def test_valid_plan_coverage_aggregates_numeric_fields_and_stage_counts(self):
        target, plan = self._plan()
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "c1-target-plan.json").write_text(json.dumps(plan))
            histories = {
                "target-planner-history": [
                    {"sequence": 1, "stage": "claim_contract",
                     "repair": {"structure": 0, "extension": 0},
                     "status": "validated"},
                    {"sequence": 2, "stage": "extension",
                     "repair": {"structure": 0, "extension": 0},
                     "status": "accepted"},
                ],
                "psi-history": [
                    {"sequence": 1, "stage": "atoms", "repair": 0, "status": "validated"},
                    {"sequence": 2, "stage": "lineage", "repair": 0, "status": "validated"},
                    {"sequence": 3, "stage": "critic", "repair": 0, "status": "accepted"},
                ],
                "verification-history": [
                    {"sequence": 1, "stage": "evidence", "repair": 0, "status": "accepted"},
                    {"sequence": 2, "stage": "world", "repair": 0, "status": "accepted"},
                    {"sequence": 3, "stage": "judgement_critic", "repair": 0,
                     "status": "accepted"},
                ],
            }
            for suffix, value in histories.items():
                (run / f"c1-{suffix}.json").write_text(json.dumps(value))
            result = _plan_coverage(run, ["c1"], {"c1": {"target": target}},
                [{"id": "c1", "status": "completed", "target_plan_sha256": plan["sha256"]}])
        totals = result["totals"]
        self.assertEqual((1, 1), (totals["plans_present"], totals["plans_valid"]))
        self.assertEqual((1, 1, 1), (totals["required_probe_coverage"],
                                     totals["projection_coverage"],
                                     totals["executed_stage_projection_coverage"]))
        self.assertEqual(3, totals["projected_probes_critic"])
        self.assertIsNone(result["probe_results"]["totals"]["result_slot_coverage"])

    def test_loop_audit_accepts_and_counts_independent_output_repair_counter(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            history = [
                {"sequence": 1, "stage": "claim_contract",
                 "repair": {"structure": 0, "extension": 0,
                            "extension_output": 0},
                 "status": "validated"},
                {"sequence": 2, "stage": "extension",
                 "repair": {"structure": 0, "extension": 0,
                            "extension_output": 0},
                 "status": "output_repair_requested"},
                {"sequence": 3, "stage": "extension",
                 "repair": {"structure": 0, "extension": 0,
                            "extension_output": 1},
                 "status": "accepted"},
            ]
            path = run / "c1-target-planner-history.json"
            path.write_text(json.dumps(history))
            self.assertEqual(history, _history(run, "c1", "target-planner-history"))
            totals = _loop_counts(run, ["c1"])["totals"]
            self.assertEqual(1, totals["target_plan_repair_requests"])
            self.assertEqual(1, totals["target_plan_repair_followup_calls"])
            self.assertEqual(1, totals["target_extension_output_repair_requests"])
            self.assertEqual(1, totals["target_extension_output_repair_followup_calls"])

            history[0]["repair"]["unexpected"] = 0
            path.write_text(json.dumps(history))
            with self.assertRaisesRegex(ValueError, "invalid counters"):
                _history(run, "c1", "target-planner-history")

    def test_missing_required_probe_is_visible_but_not_mislabeled_as_semantic_error(self):
        target, plan = self._plan()
        missing = deepcopy(plan)
        removed = missing["probes"].pop(1)
        for stage, view in missing["projections"].items():
            view["probes"] = [item for item in view["probes"] if item["id"] != removed["id"]]
        base = {key: value for key, value in missing.items()
                if key not in {"sha256", "projections"}}
        missing["sha256"] = _canonical_digest(base)
        for view in missing["projections"].values():
            view["plan_sha256"] = missing["sha256"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(missing))
            result = _audit_plan(path, target,
                {"target_plan_sha256": missing["sha256"]}, [], [])
        self.assertEqual((3, 2), (result["required_probe_slots"],
                                  result["required_probe_slots_covered"]))

    def test_plan_checksum_and_projection_tampering_fail_closed(self):
        target, plan = self._plan()
        mutations = (
            lambda value: value.update(notes="changed"),
            lambda value: value["projections"]["atoms"]["probes"].pop(),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                changed = deepcopy(plan)
                mutate(changed)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "plan.json"
                    path.write_text(json.dumps(changed))
                    with self.assertRaisesRegex(ValueError, "checksum|projection"):
                        _audit_plan(path, target, {"target_plan_sha256": plan["sha256"]}, [], [])

    def test_v1_v2_and_v3_configs_are_accepted_but_not_interchangeable(self):
        config = json.loads((self.EXTENSION_SMOKE / "config.json").read_text())
        self.assertEqual(["p02", "p04"], _extension_schedule(config))
        v2 = deepcopy(config)
        v2["experiment"] = "target_extended_psi_development_v2"
        v2["target_plan_schema"] = PLAN_SCHEMA_V2
        self.assertEqual(["p02", "p04"], _extension_schedule(v2))
        v3 = deepcopy(config)
        v3["experiment"] = "target_extended_psi_development_v3"
        v3["target_plan_schema"] = PLAN_SCHEMA_V3
        self.assertEqual(["p02", "p04"], _extension_schedule(v3))

        for changed in (
                {**v2, "target_plan_schema": PLAN_SCHEMA_V3},
                {**v3, "target_plan_schema": PLAN_SCHEMA_V2},
                {**v2, "experiment": "target_extended_psi_development_v3"},
        ):
            with self.assertRaisesRegex(ValueError, "matching target plan schema"):
                _extension_schedule(changed)

        target, plan = self._plan(2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "schema.*experiment contract"):
                _audit_plan(path, target, {"target_plan_sha256": plan["sha256"]}, [], [],
                            expected_contract="legacy-decision-probe-v1")

    def test_v4_provider_modes_are_exact_and_task_routed_delta_is_withheld(self):
        base = json.loads((self.EXTENSION_SMOKE / "config.json").read_text())
        fixed = deepcopy(base)
        fixed.update(experiment=V4_FIXED_EXPERIMENT,
                     target_plan_schema=PLAN_SCHEMA_V4,
                     provider_mode="fixed_reanalysis",
                     strict_retrieval_attribution=False,
                     retrieval_attribution_mode="legacy")
        fixed["trace_config"]["max_rounds"] = 2
        fixed["trace_config"]["experimental_force_rounds"] = True
        self.assertEqual(["p02", "p04"], _extension_schedule(fixed))

        routed = deepcopy(fixed)
        routed.update(experiment=V4_TASK_ROUTED_EXPERIMENT,
                      provider=TASK_ROUTED_PROVIDER,
                      provider_mode="task_routed",
                      strict_retrieval_attribution=True,
                      retrieval_attribution_mode="strict")
        routed["trace_config"]["experimental_force_rounds"] = False
        self.assertEqual(["p02", "p04"], _extension_schedule(routed))

        for name, changed in {
                "undeclared provider": {**routed, "provider": "another corpus"},
                "forced routed rounds": {**routed, "trace_config": {
                    **routed["trace_config"], "experimental_force_rounds": True}},
                "unforced fixed rounds": {**fixed, "trace_config": {
                    **fixed["trace_config"], "experimental_force_rounds": False}},
        }.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "provider|contract|forced"):
                    _extension_schedule(changed)

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "routed"
            shutil.copytree(self.EXTENSION_SMOKE, run)
            (run / "config.json").write_text(json.dumps(routed))
            result = score(self.GOLD, self.ORIGINAL, self.STAGED, run)
        self.assertFalse(result["material_exposure_comparable"])
        self.assertFalse(result["cross_arm_extension_accuracy_comparable"])
        self.assertEqual("within_extension_round1_to_final_and_v4_ledgers",
                         result["primary_comparison"])
        cross_arm = result["label_comparisons"]["extension_vs_staged"]
        self.assertFalse(cross_arm["material_exposure_comparable"])
        self.assertIsNone(cross_arm["all_scheduled"]["candidate_minus_baseline"])
        self.assertIn("not_a_fair_or_causal", cross_arm["accuracy_interpretation"])
        self.assertTrue(any("no fair or causal accuracy delta" in item
                            for item in result["limits"]))

    def test_v2_archive_questions_pass_but_relabeling_the_plan_v3_fails(self):
        target, archive = self._renamed_plan(PLAN_SCHEMA_V2)

        def audit(plan, contract):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "plan.json"
                path.write_text(json.dumps(plan))
                return _audit_plan(path, target,
                    {"target_plan_sha256": plan["sha256"]}, [], [],
                    expected_contract=contract)

        self.assertIn("Check actor.", {probe["question"] for probe in archive["probes"]})
        self.assertEqual(PLAN_SCHEMA_V2, audit(archive, PLAN_SCHEMA_V2)["plan_contract"])

        relabeled = deepcopy(archive)
        relabeled["schema_version"] = PLAN_SCHEMA_V3
        for view in relabeled["projections"].values():
            view["schema_version"] = PLAN_SCHEMA_V3
        self._rehash(relabeled)
        with self.assertRaisesRegex(ValueError, "program-owned canonical"):
            audit(relabeled, PLAN_SCHEMA_V3)

        dual_target, dual_time = self._dual_time_plan()
        dual_time["schema_version"] = PLAN_SCHEMA_V2
        for view in dual_time["projections"].values():
            view["schema_version"] = PLAN_SCHEMA_V2
        self._rehash(dual_time)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(dual_time))
            with self.assertRaisesRegex(ValueError, "repeats a non-identity"):
                _audit_plan(path, dual_target,
                    {"target_plan_sha256": dual_time["sha256"]}, [], [],
                    expected_contract=PLAN_SCHEMA_V2)

    @staticmethod
    def _origin_report(candidates, origins, relations, gaps=None):
        version_ids = {"m1", *candidates, *origins}
        for relation in relations:
            version_ids.update((relation["from_version"], relation["to_version"]))
        analyses = {identifier: {"origins": []} for identifier in version_ids}
        for identifier in candidates:
            analyses[identifier]["origins"].append(
                {"target_id": "c1", "version_id": identifier})
        return {
            "materials": [{"version_id": identifier} for identifier in sorted(version_ids)],
            "eligible_version_ids": sorted(version_ids),
            "analyses": analyses,
            "origins": [{"target_id": "c1", "version_id": identifier}
                        for identifier in origins],
            "relations": relations,
            "gaps": [] if gaps is None else gaps,
        }

    @staticmethod
    def _lineage_gap():
        return {"id": "lineage:c1", "stage": "provenance", "blocking": True}

    def _audit_origin_report(self, version, final_report):
        target, plan = self._plan(version)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan))
            return _audit_plan(path, target,
                {"target_plan_sha256": plan["sha256"]}, [], [],
                expected_contract=(PLAN_SCHEMA_V3 if version == 3 else PLAN_SCHEMA_V2),
                report=final_report)

    def test_v3_rejects_omitted_terminal_with_intermediate_only_export(self):
        relations = [
            {"from_version": "m1", "to_version": "m14",
             "kind": "cites", "status": "direct"},
            {"from_version": "m14", "to_version": "m15",
             "kind": "quotes", "status": "direct"},
        ]
        tampered = self._origin_report(["m14", "m15"], ["m14"], relations)
        with self.assertRaisesRegex(ValueError, "terminal-root exact-set"):
            self._audit_origin_report(3, tampered)

    def test_v3_rejects_extra_intermediate_beside_terminal_export(self):
        relations = [
            {"from_version": "m1", "to_version": "m14",
             "kind": "cites", "status": "direct"},
            {"from_version": "m14", "to_version": "m15",
             "kind": "derives", "status": "direct"},
        ]
        tampered = self._origin_report(
            ["m14", "m15"], ["m14", "m15"], relations)
        with self.assertRaisesRegex(ValueError, "terminal-root exact-set"):
            self._audit_origin_report(3, tampered)

    def test_v3_accepts_exact_parallel_terminal_roots(self):
        relations = [
            {"from_version": "m1", "to_version": "left",
             "kind": "cites", "status": "direct"},
            {"from_version": "m1", "to_version": "right",
             "kind": "translates", "status": "direct"},
            {"from_version": "left", "to_version": "right",
             "kind": "supports", "status": "direct"},
            {"from_version": "right", "to_version": "left",
             "kind": "reprints", "status": "declared"},
        ]
        report = self._origin_report(["left", "right"], ["left", "right"], relations)
        terminal = self._audit_origin_report(3, report)["terminal_origin_audit"]
        self.assertEqual({"applicable": True, "origin_count": 2,
                          "direct_lineage_edges": 2, "reachable_origin_count": 2,
                          "terminal_roots": True}, terminal)

    def test_v3_complete_chain_rejects_spurious_active_lineage_gap(self):
        relations = [
            {"from_version": "m1", "to_version": "m14",
             "kind": "cites", "status": "direct"},
            {"from_version": "m14", "to_version": "m15",
             "kind": "quotes", "status": "direct"},
        ]
        report = self._origin_report(
            ["m14", "m15"], ["m15"], relations, [self._lineage_gap()])
        with self.assertRaisesRegex(ValueError, "spurious active lineage gap"):
            self._audit_origin_report(3, report)

    def test_v2_archives_are_exempt_from_v3_terminal_candidate_reconstruction(self):
        legacy_shape = {"origins": [{"version_id": "downstream"},
                                     {"version_id": "root"}],
                        "relations": [
            {"from_version": "m1", "to_version": "downstream",
             "kind": "cites", "status": "direct"},
            {"from_version": "downstream", "to_version": "root",
             "kind": "quotes", "status": "direct"},
        ]}
        self.assertIsNone(
            self._audit_origin_report(2, legacy_shape)["terminal_origin_audit"])

    def test_v3_rejects_disconnected_final_origin_even_when_declared_candidate(self):
        disconnected = self._origin_report(["disconnected"], ["disconnected"], [])
        with self.assertRaisesRegex(ValueError, "terminal-root exact-set"):
            self._audit_origin_report(3, disconnected)

    def test_v3_disconnected_omitted_candidate_requires_active_lineage_gap(self):
        without_gap = self._origin_report(["disconnected"], [], [])
        with self.assertRaisesRegex(ValueError, "blocking lineage gap"):
            self._audit_origin_report(3, without_gap)

        with_gap = self._origin_report(
            ["disconnected"], [], [], [self._lineage_gap()])
        self.assertTrue(self._audit_origin_report(
            3, with_gap)["terminal_origin_audit"]["terminal_roots"])

    def test_v3_reachable_candidate_cycle_requires_active_lineage_gap(self):
        relations = [
            {"from_version": "m1", "to_version": "left",
             "kind": "cites", "status": "direct"},
            {"from_version": "left", "to_version": "right",
             "kind": "quotes", "status": "direct"},
            {"from_version": "right", "to_version": "left",
             "kind": "reprints", "status": "direct"},
        ]
        without_gap = self._origin_report(["left", "right"], [], relations)
        with self.assertRaisesRegex(ValueError, "blocking lineage gap"):
            self._audit_origin_report(3, without_gap)

        with_gap = self._origin_report(
            ["left", "right"], [], relations, [self._lineage_gap()])
        result = self._audit_origin_report(3, with_gap)["terminal_origin_audit"]
        self.assertEqual((0, 3),
                         (result["origin_count"], result["direct_lineage_edges"]))

    def test_v2_requires_program_owned_match_policy_routes_and_gate(self):
        target, original = self._plan(2)
        mutations = {
            "match policy": lambda probe: probe.update(match_policy="same_referent"),
            "routes": lambda probe: probe.update(routes=["atoms"]),
            "gate": lambda probe: probe.update(gate="positive_world_only"),
        }
        for message, mutate in mutations.items():
            with self.subTest(field=message):
                plan = deepcopy(original)
                mutate(plan["probes"][0])
                self._rehash(plan)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "plan.json"
                    path.write_text(json.dumps(plan))
                    with self.assertRaisesRegex(ValueError, message):
                        _audit_plan(path, target,
                            {"target_plan_sha256": plan["sha256"]}, [], [],
                            expected_contract=PLAN_SCHEMA_V2)

    def test_v2_sha_comes_from_full_critic_projection_and_every_view_has_notes(self):
        target, original = self._plan(2)
        changed = deepcopy(original)
        changed["projections"]["critic"]["notes"] = "critic changed"
        canonical = {key: changed["projections"]["critic"][key] for key in
            ("schema_version", "target_signature", "logic", "claims", "probes", "notes")}
        changed["sha256"] = _canonical_digest(canonical)
        for view in changed["projections"].values():
            view["plan_sha256"] = changed["sha256"]
        variants = {
            "top-level": changed,
            "projection contract": deepcopy(original),
        }
        variants["projection contract"]["projections"]["atoms"].pop("notes")
        for message, plan in variants.items():
            with self.subTest(problem=message), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "plan.json"
                path.write_text(json.dumps(plan))
                with self.assertRaisesRegex(ValueError, message):
                    _audit_plan(path, target,
                        {"target_plan_sha256": plan["sha256"]}, [], [],
                        expected_contract=PLAN_SCHEMA_V2)

    def test_v2_rechecks_anchors_identity_spans_designation_and_attribution_boundary(self):
        cases = []
        target, plan = self._plan(2)
        plan["claims"][0]["anchor"]["quote"] = "Alpho reported 3 items."
        cases.append(("immutable target", target, self._rehash(plan)))

        target, plan = self._plan(2)
        plan["claims"][0]["anchor"] = {"start": 0, "end": 14,
                                        "quote": "Alpha reported"}
        cases.append(("escapes", target, self._rehash(plan)))

        target, plan = self._plan(2)
        plan["claims"][0]["dimensions"].extend([
            {"id": "identity:outer", "kind": "entity_identity",
             "anchor": {"start": 0, "end": 5, "quote": "Alpha"}},
            {"id": "identity:inner", "kind": "entity_identity",
             "anchor": {"start": 0, "end": 4, "quote": "Alph"}},
        ])
        cases.append(("overlapping identity", target, self._rehash(plan)))

        target, plan = self._plan(2)
        plan["claims"][0]["dimensions"][2]["kind"] = "exact_designation"
        cases.append(("naming predicate", target, self._rehash(plan)))

        target, plan = self._plan(2)
        target["text"] = "Alpha reported that 3 items."
        plan["target_signature"] = _canonical_digest(target)
        claim = plan["claims"][0]
        claim.update(statement=target["text"], role="attribution",
                     anchor={"start": 0, "end": len(target["text"]),
                             "quote": target["text"]})
        claim["dimensions"][1]["anchor"] = {"start": 6, "end": 19,
                                             "quote": "reported that"}
        claim["dimensions"][2]["anchor"] = {"start": 20, "end": 27,
                                             "quote": "3 items"}
        plan["logic"] = "attribution"
        cases.append(("attribution dimensions", target, self._rehash(plan)))

        for message, target, plan in cases:
            with self.subTest(problem=message), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "plan.json"
                path.write_text(json.dumps(plan))
                with self.assertRaisesRegex(ValueError, message):
                    _audit_plan(path, target,
                        {"target_plan_sha256": plan["sha256"]}, [], [],
                        expected_contract=PLAN_SCHEMA_V2)

    def test_v2_requires_relation_probe_bound_to_every_relational_claim_dimension(self):
        claim = {"dimensions": [
            {"id": "subject", "kind": "subject"},
            {"id": "predicate", "kind": "predicate"},
            {"id": "condition", "kind": "condition"},
        ]}
        for logic, kind in (("conditional", "conditional_relation"),
                            ("comparison", "comparison_relation"),
                            ("causal", "causal_relation")):
            with self.subTest(logic=logic):
                bindings = _required_probe_bindings(claim, "evidence", logic)
                self.assertIn((kind, ("condition", "predicate", "subject")), bindings)

        target, plan = self._plan(2)
        plan["logic"] = "conditional"
        self._rehash(plan)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "exactly cover"):
                _audit_plan(path, target,
                    {"target_plan_sha256": plan["sha256"]}, [], [],
                    expected_contract=PLAN_SCHEMA_V2)

    def test_exact_designation_has_an_individual_required_probe_binding(self):
        claim = {"dimensions": [
            {"id": "subject", "kind": "subject"},
            {"id": "predicate", "kind": "predicate"},
            {"id": "name-a", "kind": "exact_designation"},
            {"id": "name-b", "kind": "exact_designation"},
        ]}
        bindings = _required_probe_bindings(claim, "evidence")
        self.assertIn(("exact_designation", ("name-a",)), bindings)
        self.assertIn(("exact_designation", ("name-b",)), bindings)
        self.assertNotIn(("exact_designation", ("name-a", "name-b")), bindings)
        self.assertIn(("designation_relation",
                       ("name-a", "name-b", "predicate", "subject")), bindings)
        legacy = _required_probe_bindings(claim, "evidence",
                                          plan_schema="legacy-decision-probe-v1")
        self.assertNotIn(("designation_relation",
                          ("name-a", "name-b", "predicate", "subject")), legacy)

    def test_v3_p07_identity_questions_are_canonical_and_cannot_encode_roles(self):
        target, original = self._renamed_plan(PLAN_SCHEMA_V3)

        def audit(plan):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "plan.json"
                path.write_text(json.dumps(plan))
                return _audit_plan(path, target,
                    {"target_plan_sha256": plan["sha256"]}, [], [],
                    expected_contract=PLAN_SCHEMA_V3)

        audit(original)
        claim = original["claims"][0]
        dimensions = {item["id"]: item for item in claim["dimensions"]}
        identity_probes = [probe for probe in original["probes"]
                           if probe["kind"] in {"entity_identity", "exact_designation"}]
        expected = {
            "NOAA": ('Does the evidence identify the same referent as "NOAA" by exact '
                     'mention, alias, unambiguous description or anaphora?'),
            "GOES-U": ('Does the evidence identify the same referent as "GOES-U" by exact '
                       'mention, alias, unambiguous description or anaphora?'),
            "GOES-19": ('Does the evidence establish "GOES-19" as the exact asserted '
                        'name, title, label or designation?'),
        }
        self.assertEqual(expected, {
            dimensions[probe["dimension_ids"][0]]["anchor"]["quote"]: probe["question"]
            for probe in identity_probes
        })
        self.assertTrue(all("actor" not in probe["question"].casefold() and
                            "renaming action" not in probe["question"].casefold()
                            for probe in identity_probes))

        variants = {}
        role_question = deepcopy(original)
        next(probe for probe in role_question["probes"]
             if probe["kind"] == "entity_identity" and
             dimensions[probe["dimension_ids"][0]]["anchor"]["quote"] == "NOAA")["question"] = (
                 "Did NOAA act as the actor performing the renaming action?")
        variants["role wording"] = self._rehash(role_question)

        wrong_anchor = deepcopy(original)
        noaa_probe = next(probe for probe in wrong_anchor["probes"]
                          if probe["kind"] == "entity_identity" and
                          next(item for item in wrong_anchor["claims"][0]["dimensions"]
                               if item["id"] == probe["dimension_ids"][0])
                          ["anchor"]["quote"] == "NOAA")
        goes_u = next(item for item in wrong_anchor["claims"][0]["dimensions"]
                      if item["anchor"]["quote"] == "GOES-U")
        noaa_probe["question"] = self._canonical_identity_question(
            goes_u["anchor"], "same_referent")
        variants["wrong anchor"] = self._rehash(wrong_anchor)

        wrong_policy_wording = deepcopy(original)
        exact = next(probe for probe in wrong_policy_wording["probes"]
                     if probe["kind"] == "exact_designation")
        exact_dimension = next(item for item in
                               wrong_policy_wording["claims"][0]["dimensions"]
                               if item["id"] == exact["dimension_ids"][0])
        exact["question"] = self._canonical_identity_question(
            exact_dimension["anchor"], "same_referent")
        variants["wrong policy wording"] = self._rehash(wrong_policy_wording)

        for problem, plan in variants.items():
            with self.subTest(problem=problem), self.assertRaisesRegex(
                    ValueError, "program-owned canonical"):
                audit(plan)

    def test_v3_p08_requires_one_singleton_time_probe_per_nonoverlapping_time_dimension(self):
        target, original = self._dual_time_plan()

        def audit(plan):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "plan.json"
                path.write_text(json.dumps(plan))
                return _audit_plan(path, target,
                    {"target_plan_sha256": plan["sha256"]}, [], [],
                    expected_contract=PLAN_SCHEMA_V3)

        time_ids = tuple(sorted(item["id"] for item in
                                original["claims"][0]["dimensions"]
                                if item["kind"] == "time"))
        bindings = _required_probe_bindings(original["claims"][0], "evidence",
                                            plan_schema=PLAN_SCHEMA_V3)
        self.assertEqual({("time_boundary", (identifier,)) for identifier in time_ids},
                         {binding for binding in bindings
                          if binding[0] == "time_boundary"})
        self.assertNotIn(("time_boundary", time_ids), bindings)
        for old_schema in (PLAN_SCHEMA_V2, "legacy-decision-probe-v1"):
            legacy = _required_probe_bindings(original["claims"][0], "evidence",
                                              plan_schema=old_schema)
            self.assertIn(("time_boundary", time_ids), legacy)
        self.assertEqual(4, audit(original)["required_probe_slots"])

        merged = deepcopy(original)
        time_probes = [probe for probe in merged["probes"]
                       if probe["kind"] == "time_boundary"]
        time_probes[0]["dimension_ids"] = list(time_ids)
        merged["probes"].remove(time_probes[1])
        self._rehash(merged)
        with self.assertRaisesRegex(ValueError, "binding does not match"):
            audit(merged)

        missing = deepcopy(original)
        missing["probes"].remove(next(probe for probe in missing["probes"]
                                      if probe["kind"] == "time_boundary"))
        self._rehash(missing)
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            audit(missing)

        overlapping = deepcopy(original)
        times = [item for item in overlapping["claims"][0]["dimensions"]
                 if item["kind"] == "time"]
        start = target["text"].index("Apollo-era")
        times[1]["anchor"] = {"start": start, "end": start + len("Apollo-era regional"),
                              "quote": "Apollo-era regional"}
        self._rehash(overlapping)
        with self.assertRaisesRegex(ValueError, "overlapping time"):
            audit(overlapping)

    def test_v3_p08_dual_time_probe_results_retain_partial_support(self):
        target, plan = self._dual_time_plan()
        dimensions = {item["id"]: item for item in plan["claims"][0]["dimensions"]}
        newer = next(probe for probe in plan["probes"]
                     if probe["kind"] == "time_boundary" and
                     dimensions[probe["dimension_ids"][0]]["anchor"]["quote"] == "newer")
        span = {"version_id": "m15", "start": 0, "end": len(target["text"]),
                "quote": target["text"]}
        overrides = {(stage, newer["id"]): {"status": "unresolved", "basis": []}
                     for stage in ("evidence", "world")}
        report = self._report(plan, span, overrides)
        verification = [
            {"stage": "evidence", "status": "accepted", "round": 1},
            {"stage": "world", "status": "accepted", "round": 1},
            {"stage": "judgement_critic", "status": "accepted", "round": 1},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan))
            result = _audit_plan(path, target,
                {"status": "completed", "target_plan_sha256": plan["sha256"]},
                [], verification, expected_contract=PLAN_SCHEMA_V3,
                report_history=report,
                materials=[{"version_id": "m15", "content": target["text"]}]
                )["probe_result_audit"]
        self.assertEqual((7, 7, 5, 2),
                         (result["expected_result_slots"], result["result_slots"],
                          result["status_counts"]["supported"],
                          result["status_counts"]["unresolved"]))
        self.assertEqual(5, result["grounded_conclusive_results"])

    def test_v2_renamed_plan_requires_one_all_dimension_designation_relation(self):
        target, original = self._renamed_plan()

        def audit(plan):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "plan.json"
                path.write_text(json.dumps(plan))
                return _audit_plan(path, target,
                    {"target_plan_sha256": plan["sha256"]}, [], [],
                    expected_contract=PLAN_SCHEMA_V2)

        result = audit(original)
        self.assertEqual(result["required_probe_slots"],
                         result["required_probe_slots_covered"])
        relation, = [probe for probe in original["probes"]
                     if probe["kind"] == "designation_relation"]
        self.assertEqual({item["id"] for item in original["claims"][0]["dimensions"]},
                         set(relation["dimension_ids"]))
        self.assertEqual("semantic_constraint", relation["match_policy"])
        self.assertEqual(["atoms", "evidence", "world"], relation["routes"])

        missing = deepcopy(original)
        missing["probes"] = [probe for probe in missing["probes"]
                             if probe["kind"] != "designation_relation"]
        self._rehash(missing)
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            audit(missing)

        partial = deepcopy(original)
        next(probe for probe in partial["probes"]
             if probe["kind"] == "designation_relation")["dimension_ids"].pop()
        self._rehash(partial)
        with self.assertRaisesRegex(ValueError, "binding does not match"):
            audit(partial)

        repeated = deepcopy(original)
        old_identity = next(item for item in repeated["claims"][0]["dimensions"]
                            if item["anchor"]["quote"] == "GOES-U")
        old_identity["kind"] = "exact_designation"
        self._rehash(repeated)
        with self.assertRaisesRegex(ValueError, "repeats a non-identity"):
            audit(repeated)

    def test_v2_designation_result_needs_one_connected_naming_sentence(self):
        target, plan = self._renamed_plan()
        relation = next(probe for probe in plan["probes"]
                        if probe["kind"] == "designation_relation")
        verification = [
            {"stage": "evidence", "status": "accepted", "round": 1},
            {"stage": "world", "status": "accepted", "round": 1},
            {"stage": "judgement_critic", "status": "accepted", "round": 1},
        ]

        def audit_content(content, relation_status):
            span = {"version_id": "m13", "start": 0, "end": len(content),
                    "quote": content}
            overrides = {}
            for stage in ("evidence", "world"):
                for probe in plan["probes"]:
                    if stage not in probe["routes"]:
                        continue
                    if probe["match_policy"] in {"same_referent", "exact_designation"}:
                        overrides[(stage, probe["id"])] = {"referent_relation": "exact"}
                overrides[(stage, relation["id"])] = {"status": relation_status}
            report = self._report(plan, span, overrides)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "plan.json"
                path.write_text(json.dumps(plan))
                return _audit_plan(path, target, {"status": "completed",
                    "target_plan_sha256": plan["sha256"]}, [], verification,
                    expected_contract=PLAN_SCHEMA_V2, report_history=report,
                    materials=[{"version_id": "m13", "content": content}])

        disconnected = (
            "The archive renamed another satellite to TEST-1. NOAA operates GOES-U. "
            "GOES-19 appears in a catalogue dated June 25, 2024."
        )
        for status in ("supported", "contradicted"):
            with self.subTest(status=status), self.assertRaisesRegex(
                    ValueError, "naming-predicate basis span"):
                audit_content(disconnected, status)

        connected = "NOAA RENAMED GOES-U to goes-19 on July 7, 2024."
        self.assertEqual(PLAN_SCHEMA_V2,
                         audit_content(connected, "contradicted")["plan_contract"])

    def test_v2_scorer_rechecks_stable_ids_bounds_and_boolean_roles(self):
        def audit(target, plan, message):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "plan.json"
                path.write_text(json.dumps(plan))
                with self.assertRaisesRegex(ValueError, message):
                    _audit_plan(path, target,
                        {"target_plan_sha256": plan["sha256"]}, [], [],
                        expected_contract=PLAN_SCHEMA_V2)

        target, original = self._plan(2)
        unstable = deepcopy(original)
        old_claim = unstable["claims"][0]["id"]
        unstable["claims"][0]["id"] = "c1:claim:not-stable"
        for probe in unstable["probes"]:
            if probe["claim_id"] == old_claim:
                probe["claim_id"] = unstable["claims"][0]["id"]
        self._rehash(unstable)
        audit(target, unstable, "claim ID is not planner-stable")

        too_many_claims = deepcopy(original)
        too_many_claims["claims"] = too_many_claims["claims"] * 5
        self._rehash(too_many_claims)
        audit(target, too_many_claims, "schema bounds")

        too_many_dimensions = deepcopy(original)
        too_many_dimensions["claims"][0]["dimensions"] += (
            too_many_dimensions["claims"][0]["dimensions"][:1] * 7)
        self._rehash(too_many_dimensions)
        audit(target, too_many_dimensions, "dimensions")

        too_many_probes = deepcopy(original)
        while len(too_many_probes["probes"]) < 45:
            too_many_probes["probes"].append(deepcopy(too_many_probes["probes"][0]))
        self._rehash(too_many_probes)
        audit(target, too_many_probes, "schema bounds")

        boolean_target = {"id": "bool", "text": "Alpha rose and Beta fell.",
            "as_of": "2026-09-05T00:00:00Z", "source_version_id": "m1",
            "assessment_mode": "evidence", "evidence_scope": ["m1"]}

        def claim(name, statement, start, role, subject, predicate):
            subject_start = boolean_target["text"].index(subject, start)
            predicate_start = boolean_target["text"].index(predicate, start)
            return {"id": name, "statement": statement,
                "anchor": {"start": start, "end": start + len(statement),
                           "quote": statement}, "role": role,
                "dimensions": [
                    {"id": name + ":subject", "kind": "subject",
                     "anchor": {"start": subject_start,
                                "end": subject_start + len(subject), "quote": subject}},
                    {"id": name + ":predicate", "kind": "predicate",
                     "anchor": {"start": predicate_start,
                                "end": predicate_start + len(predicate), "quote": predicate}},
                ], "parent_claim_id": None}

        claims = [claim("left", "Alpha rose", 0, "main", "Alpha", "rose"),
                  claim("right", "Beta fell", 15, "comparison", "Beta", "fell")]
        probes = []
        for item in claims:
            probes.extend([
                {"id": item["id"] + ":semantic", "claim_id": item["id"],
                 "kind": "semantic_core",
                 "dimension_ids": [dimension["id"] for dimension in item["dimensions"]],
                 "question": "Check the clause.", "decision_impact": "Changes result.",
                 "match_policy": "semantic_constraint",
                 "routes": ["atoms", "evidence", "world"], "gate": "always"},
                {"id": item["id"] + ":lineage", "claim_id": item["id"],
                 "kind": "source_lineage", "dimension_ids": [],
                 "question": "Check lineage.", "decision_impact": "Changes result.",
                 "match_policy": "semantic_constraint",
                 "routes": ["lineage", "world"], "gate": "provenance"},
            ])
        self._stabilize_v2_ids(boolean_target, claims, probes)
        boolean_plan = {"schema_version": PLAN_SCHEMA_V2,
            "target_signature": _canonical_digest(boolean_target), "logic": "and",
            "claims": claims, "probes": probes, "notes": "", "sha256": "",
            "projections": {}}
        self._rehash(boolean_plan)
        audit(boolean_target, boolean_plan, "nested relation role")

    def test_v2_audits_one_grounded_result_per_projected_probe_per_cycle(self):
        target, plan = self._plan(2)
        span = {"version_id": "m1", "start": 0, "end": 5, "quote": "Alpha"}
        evidence = [probe for probe in plan["probes"] if "evidence" in probe["routes"]]
        world = [probe for probe in plan["probes"] if "world" in probe["routes"]]
        report_history = [{"round": 1,
            "evidence_probe_results": [
                self._probe_result(evidence[0], "evidence", [span]),
                self._probe_result(evidence[1], "evidence", [], "unresolved"),
            ],
            "world_probe_results": [self._probe_result(probe, "world", [span])
                                    for probe in world]}]
        verification = [
            {"stage": "evidence", "status": "accepted", "round": 1},
            {"stage": "world", "status": "accepted", "round": 1},
            {"stage": "judgement_critic", "status": "accepted", "round": 1},
        ]
        materials = [{"version_id": "m1", "content": target["text"]}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan))
            result = _audit_plan(path, target,
                {"status": "completed", "target_plan_sha256": plan["sha256"]},
                [{"stage": "atoms"}, {"stage": "lineage"}], verification,
                expected_contract=PLAN_SCHEMA_V2, report_history=report_history,
                materials=materials)["probe_result_audit"]
        self.assertTrue(result["available"])
        self.assertEqual((5, 5, 1), (result["expected_result_slots"],
                                     result["result_slots"],
                                     result["accepted_judgement_cycles"]))
        self.assertEqual((4, 4, 4), (result["results_with_basis"],
                                     result["conclusive_results"],
                                     result["grounded_conclusive_results"]))
        self.assertEqual(1, result["result_slot_coverage"])

    def test_v2_rechecks_status_policy_relation_and_basis_semantics(self):
        verification = [{"stage": "judgement_critic", "status": "accepted", "round": 1}]

        def audit(target, plan, history):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "plan.json"
                path.write_text(json.dumps(plan))
                return _audit_plan(path, target,
                    {"status": "completed", "target_plan_sha256": plan["sha256"]},
                    [], verification, expected_contract=PLAN_SCHEMA_V2,
                    report_history=history,
                    materials=[{"version_id": "m1", "content": target["text"]}])

        target, plan = self._identity_plan("entity_identity")
        span = {"version_id": "m1", "start": 0, "end": 5, "quote": "Alpha"}
        identity = next(probe for probe in plan["probes"]
                        if probe["kind"] == "entity_identity")
        valid_overrides = {(stage, identity["id"]): {"referent_relation": "description"}
                           for stage in ("evidence", "world")}
        self.assertTrue(audit(target, plan, self._report(
            plan, span, valid_overrides))["probe_result_audit"]["available"])

        history = self._report(plan, span, valid_overrides)
        next(item for item in history[0]["evidence_probe_results"]
             if item["probe_id"] == identity["id"])["referent_relation"] = "unresolved"
        with self.assertRaisesRegex(ValueError, "status conflicts"):
            audit(target, plan, history)

        target, plan = self._identity_plan("exact_designation")
        exact = next(probe for probe in plan["probes"]
                     if probe["kind"] == "exact_designation")
        overrides = {(stage, exact["id"]): {"referent_relation": "description"}
                     for stage in ("evidence", "world")}
        with self.assertRaisesRegex(ValueError, "status conflicts"):
            audit(target, plan, self._report(plan, span, overrides))

        target, plan = self._plan(2)
        semantic = next(probe for probe in plan["probes"]
                        if probe["kind"] == "semantic_core")
        history = self._report(plan, span,
            {("evidence", semantic["id"]): {"referent_relation": "exact"}})
        with self.assertRaisesRegex(ValueError, "semantic constraint"):
            audit(target, plan, history)

        history = self._report(plan, span,
            {("evidence", semantic["id"]): {"basis": []}})
        with self.assertRaisesRegex(ValueError, "conclusive status"):
            audit(target, plan, history)

        history = self._report(plan, span,
            {("evidence", semantic["id"]): {"status": "conflicting"}})
        with self.assertRaisesRegex(ValueError, "two distinct"):
            audit(target, plan, history)

    def test_v2_rejects_missing_duplicate_and_out_of_scope_probe_results(self):
        target, plan = self._plan(2)
        target["evidence_scope"] = ["m1"]
        # Updating the immutable target requires a correspondingly signed plan.
        plan["target_signature"] = _canonical_digest(target)
        for view in plan["projections"].values():
            view["target_signature"] = plan["target_signature"]
        self._rehash(plan)
        evidence = [probe for probe in plan["probes"] if "evidence" in probe["routes"]]
        world = [probe for probe in plan["probes"] if "world" in probe["routes"]]
        span1 = {"version_id": "m1", "start": 0, "end": 5, "quote": "Alpha"}
        span2 = {"version_id": "m2", "start": 0, "end": 5, "quote": "Alpha"}
        base = {"round": 1,
            "evidence_probe_results": [self._probe_result(probe, "evidence", [span1])
                                       for probe in evidence],
            "world_probe_results": [self._probe_result(probe, "world", [span1])
                                    for probe in world]}
        verification = [{"stage": "judgement_critic", "status": "accepted", "round": 1}]
        materials = [{"version_id": "m1", "content": target["text"]},
                     {"version_id": "m2", "content": target["text"]}]
        variants = {
            "exactly one": lambda item: item["evidence_probe_results"].pop(),
            "duplicate": lambda item: item["evidence_probe_results"].append(
                deepcopy(item["evidence_probe_results"][0])),
            "scope": lambda item: item["evidence_probe_results"][0].update(basis=[span2]),
        }
        for message, mutate in variants.items():
            with self.subTest(problem=message):
                report = deepcopy(base)
                mutate(report)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "plan.json"
                    path.write_text(json.dumps(plan))
                    with self.assertRaisesRegex(ValueError, message):
                        _audit_plan(path, target,
                            {"status": "completed", "target_plan_sha256": plan["sha256"]},
                            [], verification, expected_contract=PLAN_SCHEMA_V2,
                            report_history=[report], materials=materials)

    def test_executed_stage_projection_is_not_reported_as_probe_result_coverage(self):
        target, plan = self._plan(2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan))
            result = _audit_plan(path, target,
                {"status": "error", "target_plan_sha256": plan["sha256"]},
                [{"stage": "atoms"}, {"stage": "lineage"}],
                [{"stage": "evidence"}, {"stage": "world"}],
                expected_contract=PLAN_SCHEMA_V2)
        self.assertEqual(result["routed_stage_slots"],
                         result["projected_slots_at_executed_stages"])
        self.assertFalse(result["probe_result_audit"]["available"])
        self.assertIsNone(result["probe_result_audit"]["result_slot_coverage"])

    def test_cli_writes_json_without_api_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.json"
            env = {key: value for key, value in os.environ.items() if "OPENAI" not in key}
            command = [sys.executable, str(ROOT / "experiments" /
                "target_extension_compare_score.py"), "--gold", str(self.GOLD),
                "--original-run", str(self.ORIGINAL), "--staged-run", str(self.STAGED),
                "--extension-run", str(self.EXTENSION_SMOKE), "--output", str(output)]
            run = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(0, run.returncode, run.stderr)
            result = json.loads(output.read_text())
            self.assertEqual("target-extension-compare-score-v2", result["schema_version"])
            self.assertEqual(0, result["target_probe_result_audit"]["totals"]
                             ["cases_with_results"])


if __name__ == "__main__":
    unittest.main()
