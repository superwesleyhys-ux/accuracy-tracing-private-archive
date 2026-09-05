"""Offline contract tests for target decomposition and decision-probe extension."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import importlib
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
e = importlib.import_module("extended_semantic")


TARGET = {
    "id": "target",
    "text": "Output did not rise above the 2024 baseline before 5 September.",
    "as_of": "2026-09-04T20:00:00Z",
    "source_version_id": "notice",
    "assessment_mode": "evidence",
    "evidence_scope": ["notice"],
}


def contract(*, include_baseline=True):
    dimensions = [
        {"kind": "subject", "quote": "Output"},
        {"kind": "predicate", "quote": "rise"},
        {"kind": "negation", "quote": "did not"},
        {"kind": "time", "quote": "before 5 September"},
    ]
    if include_baseline:
        dimensions.append({"kind": "baseline_scope", "quote": "above the 2024 baseline"})
    return {"claims": [{
        "statement": "Output did not rise above the 2024 baseline before 5 September.",
        "quote": TARGET["text"], "role": "main", "dimensions": dimensions,
    }], "logic": "single", "notes": ""}


def accepted_extension(payload):
    claim = payload["claim_contract"]["claims"][0]
    dimension_kinds = {item["kind"] for item in claim["dimensions"]}
    kinds = ["semantic_core", "source_lineage"]
    for dimension, probe in (
        ("negation", "polarity"), ("time", "time_boundary"),
        ("quantity_unit", "quantity_unit"), ("baseline_scope", "baseline_scope"),
        ("entity_identity", "entity_identity"),
    ):
        if dimension in dimension_kinds:
            kinds.append(probe)
    if dimension_kinds & {"condition", "modality"}:
        kinds.append("condition_modality")
    if payload["target"]["assessment_mode"] == "world":
        kinds.append("source_independence")
    return {"decision": "accept", "repair_quote": "", "repair_issue": "",
            "probes": [{"claim_id": claim["id"], "kind": kind,
                        "question": "Check " + kind + ".",
                        "decision_impact": "A mismatch could change the decision."}
                       for kind in reversed(kinds)],
            "notes": ""}


class ScriptClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def call(self, system, user, schema):
        stage = ("claim_contract" if system == e.CLAIM_CONTRACT_PROMPT
                 else "extension" if system == e.EXTENSION_PROMPT else None)
        if stage is None:
            raise AssertionError("unknown planner prompt")
        expected_schema = (e.CLAIM_CONTRACT_SCHEMA if stage == "claim_contract"
                           else e.EXTENSION_SCHEMA)
        if schema != expected_schema:
            raise AssertionError("planner schema mismatch")
        payload = json.loads(user)
        self.calls.append({"stage": stage, "payload": payload})
        response = self.responses.pop(0)
        return deepcopy(response(payload) if callable(response) else response)


class TargetPlannerTests(unittest.TestCase):
    def test_plan_has_exact_anchors_required_coverage_and_bounded_projections(self):
        client = ScriptClient(contract(), accepted_extension)
        plan = e.TargetPlanner(client).prepare(TARGET)

        self.assertEqual(["claim_contract", "extension"],
                         [item["stage"] for item in client.calls])
        claim, = plan.claims
        self.assertEqual(TARGET["text"], claim.anchor.quote)
        self.assertEqual(TARGET["text"].index("rise"),
                         next(item.anchor.start for item in claim.dimensions
                              if item.kind == "predicate"))
        self.assertEqual({"semantic_core", "source_lineage", "polarity",
                          "time_boundary", "baseline_scope"},
                         {item.kind for item in plan.probes})
        self.assertEqual(64, len(plan.sha256))
        self.assertEqual(plan.sha256, plan.to_payload()["sha256"])

        atoms = plan.projection("atoms")
        lineage = plan.projection("lineage")
        evidence = plan.projection("evidence")
        world = plan.projection("world")
        self.assertEqual({"semantic_core", "polarity", "time_boundary", "baseline_scope"},
                         {item.kind for item in atoms.probes})
        self.assertEqual({"source_lineage"}, {item.kind for item in lineage.probes})
        self.assertEqual({item.kind for item in atoms.probes},
                         {item.kind for item in evidence.probes})
        self.assertEqual({"source_lineage"}, {item.kind for item in world.probes})
        with self.assertRaises(FrozenInstanceError):
            atoms.stage = "world"
        serialized = json.dumps(plan.to_payload(), sort_keys=True)
        self.assertNotIn('"basis"', serialized)
        self.assertNotIn('"verdict"', serialized)
        self.assertNotIn('"resolutions"', serialized)

    def test_extension_repairs_claim_contract_once_and_discards_old_dependents(self):
        repair = {"decision": "repair", "repair_quote": "above the 2024 baseline",
                  "repair_issue": "The comparison baseline is missing.",
                  "probes": [], "notes": ""}
        client = ScriptClient(contract(include_baseline=False), repair,
                              contract(include_baseline=True), accepted_extension)
        planner = e.TargetPlanner(client, max_repairs=1)
        plan = planner.prepare(TARGET)

        self.assertEqual(["claim_contract", "extension", "claim_contract", "extension"],
                         [item["stage"] for item in client.calls])
        self.assertEqual({"quote": repair["repair_quote"], "issue": repair["repair_issue"]},
                         client.calls[2]["payload"]["repair"])
        self.assertIn("baseline_scope", {item.kind for item in plan.claims[0].dimensions})
        self.assertEqual("repair_requested", planner.history[1]["status"])
        self.assertEqual("accepted", planner.history[-1]["status"])

    def test_second_repair_request_exhausts_without_returning_a_plan(self):
        repair = {"decision": "repair", "repair_quote": "above the 2024 baseline",
                  "repair_issue": "The comparison baseline is still missing.",
                  "probes": [], "notes": ""}
        client = ScriptClient(contract(include_baseline=False), repair,
                              contract(include_baseline=False), repair)
        planner = e.TargetPlanner(client, max_repairs=1)
        with self.assertRaisesRegex(e.TargetPlanningError, "repair budget exhausted"):
            planner.prepare(TARGET)
        self.assertEqual("repair_exhausted", planner.history[-1]["status"])
        self.assertEqual([], client.responses)

    def test_repair_cannot_carry_probes_from_the_invalid_contract(self):
        bad = {"decision": "repair", "repair_quote": "above the 2024 baseline",
               "repair_issue": "The comparison baseline is missing.",
               "probes": [{"claim_id": "stale", "kind": "semantic_core",
                           "question": "Stale question.", "decision_impact": "Stale."}],
               "notes": ""}
        planner = e.TargetPlanner(ScriptClient(contract(include_baseline=False), bad))
        with self.assertRaisesRegex(e.TargetPlanningError, "cannot retain dependent probes"):
            planner.prepare(TARGET)

    def test_unknown_claim_and_missing_required_probe_fail_closed(self):
        def unknown(payload):
            answer = accepted_extension(payload)
            answer["probes"][0]["claim_id"] = "model-invented-claim"
            return answer

        def missing(payload):
            answer = accepted_extension(payload)
            answer["probes"] = [item for item in answer["probes"]
                                if item["kind"] != "time_boundary"]
            return answer

        for label, response, message in (
            ("unknown", unknown, "unknown claim"),
            ("missing", missing, "missing required claim probes"),
        ):
            with self.subTest(label=label):
                planner = e.TargetPlanner(ScriptClient(contract(), response))
                with self.assertRaisesRegex(e.TargetPlanningError, message):
                    planner.prepare(TARGET)

    def test_target_and_parent_relative_anchor_validation_fails_before_extension(self):
        absent = contract()
        absent["claims"][0]["quote"] = "This target passage does not exist."
        outside = contract()
        outside["claims"][0]["quote"] = "Output did not rise"
        for label, response in (("absent clause", absent), ("outside dimension", outside)):
            with self.subTest(label=label):
                client = ScriptClient(response)
                planner = e.TargetPlanner(client)
                with self.assertRaises(e.TargetPlanningError):
                    planner.prepare(TARGET)
                self.assertEqual(["claim_contract"], [item["stage"] for item in client.calls])

    def test_ids_and_plan_hash_are_order_invariant_and_cached_per_target(self):
        target = {**TARGET, "text": "Output fell and delays rose."}
        first = {"claims": [
            {"statement": "Output fell.", "quote": "Output fell", "role": "conjunct",
             "dimensions": [{"kind": "subject", "quote": "Output"},
                            {"kind": "predicate", "quote": "fell"}]},
            {"statement": "Delays rose.", "quote": "delays rose", "role": "conjunct",
             "dimensions": [{"kind": "subject", "quote": "delays"},
                            {"kind": "predicate", "quote": "rose"}]},
        ], "logic": "and", "notes": ""}
        second = deepcopy(first)
        second["claims"].reverse()

        def extension_for_all(payload):
            probes = []
            for claim in reversed(payload["claim_contract"]["claims"]):
                for kind in ("source_lineage", "semantic_core"):
                    probes.append({"claim_id": claim["id"], "kind": kind,
                                   "question": "Check " + kind + " for " + claim["statement"],
                                   "decision_impact": "This check can change the decision."})
            return {"decision": "accept", "repair_quote": "", "repair_issue": "",
                    "probes": probes, "notes": ""}

        client_one = ScriptClient(first, extension_for_all)
        planner_one = e.TargetPlanner(client_one)
        plan_one = planner_one.prepare(target)
        self.assertIs(plan_one, planner_one.prepare(target))
        self.assertEqual(2, len(client_one.calls))

        plan_two = e.TargetPlanner(ScriptClient(second, extension_for_all)).prepare(target)
        self.assertEqual([item.id for item in plan_one.claims], [item.id for item in plan_two.claims])
        self.assertEqual([item.id for item in plan_one.probes], [item.id for item in plan_two.probes])
        self.assertEqual(plan_one.sha256, plan_two.sha256)

    def test_changed_target_signature_never_reuses_cached_plan(self):
        def claim_for(payload):
            text = payload["target"]["text"]
            predicate = "fell" if "fell" in text else "rose"
            return {"claims": [{"statement": text, "quote": text, "role": "main",
                "dimensions": [{"kind": "subject", "quote": "Output"},
                               {"kind": "predicate", "quote": predicate}]}],
                "logic": "single", "notes": ""}

        client = ScriptClient(claim_for, accepted_extension, claim_for, accepted_extension)
        planner = e.TargetPlanner(client)
        first = planner.prepare({**TARGET, "text": "Output fell."})
        second = planner.prepare({**TARGET, "text": "Output rose."})
        self.assertNotEqual(first.target_signature, second.target_signature)
        self.assertNotEqual(first.claims[0].id, second.claims[0].id)
        self.assertEqual(4, len(client.calls))

    def test_projection_detects_tampered_frozen_plan_copy(self):
        plan = e.TargetPlanner(ScriptClient(contract(), accepted_extension)).prepare(TARGET)
        altered = replace(plan, notes="silently changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            e.project_plan(altered, "atoms")


if __name__ == "__main__":
    unittest.main()

