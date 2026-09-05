"""Offline target-extension comparison tests; no model or credentials are used."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from target_extension_compare_score import (PLAN_SCHEMA_V2, PLAN_SCHEMA_V3, _audit_plan,
                                             _canonical_digest, _extension_schedule,
                                             _history, _loop_counts, _plan_coverage,
                                             _required_probe_bindings, score)


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
    def _rehash(plan):
        fields = ("target_signature", "logic", "claims", "probes", "notes")
        payload = {key: plan[key] for key in fields}
        plan_schema = plan.get("schema_version")
        if plan_schema in {PLAN_SCHEMA_V2, PLAN_SCHEMA_V3}:
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
            if plan_schema in {PLAN_SCHEMA_V2, PLAN_SCHEMA_V3}:
                view = {"schema_version": plan_schema, **view,
                        "logic": plan["logic"], "notes": plan["notes"]}
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
