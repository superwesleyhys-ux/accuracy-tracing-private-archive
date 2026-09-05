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
    probes = []
    for claim in payload["claim_contract"]["claims"]:
        dimensions = {}
        for item in claim["dimensions"]:
            dimensions.setdefault(item["kind"], []).append(item["id"])
        bindings = [
            ("semantic_core", dimensions["subject"] + dimensions["predicate"]),
            ("source_lineage", []),
        ]
        for dimension, probe in (
            ("negation", "polarity"),
            ("quantity_unit", "quantity_unit"), ("baseline_scope", "baseline_scope"),
        ):
            if dimension in dimensions:
                bindings.append((probe, dimensions[dimension]))
        bindings.extend(("time_boundary", [identifier])
                        for identifier in dimensions.get("time", []))
        if set(dimensions) & {"condition", "modality"}:
            bindings.append(("condition_modality",
                             dimensions.get("condition", []) + dimensions.get("modality", [])))
        bindings.extend(("entity_identity", [identifier])
                        for identifier in dimensions.get("entity_identity", []))
        bindings.extend(("exact_designation", [identifier])
                        for identifier in dimensions.get("exact_designation", []))
        if dimensions.get("exact_designation"):
            bindings.append(("designation_relation", [item["id"]
                for item in claim["dimensions"]]))
        logic = payload["claim_contract"]["logic"]
        if logic in {"conditional", "comparison", "causal"}:
            bindings.append((logic + "_relation", [item["id"]
                for item in claim["dimensions"]]))
        if payload["target"]["assessment_mode"] == "world":
            bindings.append(("source_independence", []))
        probes.extend({"claim_id": claim["id"], "kind": kind,
                       "dimension_ids": list(reversed(dimension_ids)),
                       "question": "Check " + kind + ".",
                       "decision_impact": "A mismatch could change the decision."}
                      for kind, dimension_ids in reversed(bindings))
    return {"decision": "accept", "repair_quote": "", "repair_issue": "",
            "probes": probes,
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
        self.assertTrue(all(item.id.startswith("target:dimension:")
                            for item in claim.dimensions))
        semantic = next(item for item in plan.probes if item.kind == "semantic_core")
        self.assertEqual({item.id for item in claim.dimensions
                          if item.kind in {"subject", "predicate"}},
                         set(semantic.dimension_ids))
        self.assertEqual((), next(item for item in plan.probes
                                  if item.kind == "source_lineage").dimension_ids)
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
        self.assertEqual({"semantic_core", "polarity", "time_boundary",
                          "baseline_scope", "source_lineage"},
                         {item.kind for item in world.probes})
        self.assertEqual(e.PLAN_SCHEMA_VERSION, plan.to_payload()["schema_version"])
        self.assertEqual(plan.logic, world.logic)
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

    def test_nested_attribution_and_extension_use_independent_repair_budgets(self):
        target = {**TARGET, "text": ("On 10 January 2025, NOAA reported that global surface "
            "temperature in 2024 was 1.46°C above the 1850–1900 baseline.")}
        collapsed = {"claims": [{
            "statement": target["text"], "quote": target["text"], "role": "main",
            "dimensions": [
                {"kind": "subject", "quote": "NOAA"},
                {"kind": "predicate", "quote": "reported"},
                {"kind": "time", "quote": "On 10 January 2025"},
                {"kind": "entity_identity", "quote": "NOAA"},
                {"kind": "subject", "quote": "global surface temperature"},
                {"kind": "predicate", "quote": "was"},
                {"kind": "time", "quote": "2024"},
                {"kind": "quantity_unit", "quote": "1.46°C"},
                {"kind": "baseline_scope", "quote": "above the 1850–1900 baseline"},
            ],
        }], "logic": "single", "notes": ""}
        split = {"claims": [
            {"statement": "NOAA reported the statement on 10 January 2025.",
             "quote": "On 10 January 2025, NOAA reported", "role": "attribution",
             "dimensions": [
                 {"kind": "subject", "quote": "NOAA"},
                 {"kind": "predicate", "quote": "reported"},
                 {"kind": "time", "quote": "On 10 January 2025"},
                 {"kind": "entity_identity", "quote": "NOAA"},
             ]},
            {"statement": ("Global surface temperature in 2024 was 1.46°C above the "
                           "1850–1900 baseline."),
             "quote": ("global surface temperature in 2024 was 1.46°C above the "
                       "1850–1900 baseline"), "role": "attributed_content",
             "dimensions": [
                 {"kind": "subject", "quote": "global surface temperature"},
                 {"kind": "predicate", "quote": "was"},
                 {"kind": "time", "quote": "2024"},
                 {"kind": "quantity_unit", "quote": "1.46°C"},
                 {"kind": "baseline_scope", "quote": "above the 1850–1900 baseline"},
             ]},
        ], "logic": "attribution", "notes": ""}
        review_repair = {"decision": "repair",
            "repair_quote": "above the 1850–1900 baseline",
            "repair_issue": "Confirm that the baseline remains attached to the content clause.",
            "probes": [], "notes": ""}
        client = ScriptClient(collapsed, split, review_repair, split, accepted_extension)
        planner = e.TargetPlanner(client, max_repairs=1, max_structure_repairs=1)
        plan = planner.prepare(target)

        self.assertEqual(["claim_contract", "claim_contract", "extension",
                          "claim_contract", "extension"],
                         [item["stage"] for item in client.calls])
        self.assertEqual("structure_repair_requested", planner.history[0]["status"])
        self.assertIn("Split nested attribution", client.calls[1]["payload"]["repair"]["issue"])
        self.assertEqual({"quote": review_repair["repair_quote"],
                          "issue": review_repair["repair_issue"]},
                         client.calls[3]["payload"]["repair"])
        self.assertEqual("repair_requested", planner.history[2]["status"])
        self.assertEqual({"attribution", "attributed_content"},
                         {claim.role for claim in plan.claims})

    def test_repeated_entities_get_stable_ids_and_one_probe_each(self):
        target = {**TARGET, "assessment_mode": "world", "evidence_scope": [],
                  "text": "NASA's OSIRIS-REx mission sampled asteroid Bennu."}
        response = {"claims": [{"statement": target["text"], "quote": target["text"],
            "role": "main", "dimensions": [
                {"kind": "subject", "quote": "NASA's OSIRIS-REx mission"},
                {"kind": "predicate", "quote": "sampled"},
                {"kind": "entity_identity", "quote": "NASA"},
                {"kind": "entity_identity", "quote": "OSIRIS-REx"},
                {"kind": "entity_identity", "quote": "asteroid Bennu"},
            ]}], "logic": "single", "notes": ""}
        plan = e.TargetPlanner(ScriptClient(response, accepted_extension)).prepare(target)
        entities = [item for item in plan.claims[0].dimensions
                    if item.kind == "entity_identity"]
        entity_probes = [item for item in plan.probes if item.kind == "entity_identity"]
        self.assertEqual(3, len({item.id for item in entities}))
        self.assertEqual(3, len({item.id for item in entity_probes}))
        self.assertEqual({(item.id,) for item in entities},
                         {item.dimension_ids for item in entity_probes})
        self.assertEqual({"same_referent"}, {item.match_policy for item in entity_probes})

        reordered = deepcopy(response)
        reordered["claims"][0]["dimensions"].reverse()
        again = e.TargetPlanner(ScriptClient(reordered, accepted_extension)).prepare(target)
        self.assertEqual(plan.sha256, again.sha256)
        self.assertEqual([item.id for item in plan.probes], [item.id for item in again.probes])

    def test_nested_entity_is_repaired_to_maximal_non_overlapping_referents(self):
        target = {**TARGET,
                  "text": "The Unified Geologic Map of the Moon was released by USGS."}
        nested = {"claims": [{"statement": target["text"], "quote": target["text"],
            "role": "main", "dimensions": [
                {"kind": "subject", "quote": "The Unified Geologic Map of the Moon"},
                {"kind": "predicate", "quote": "was released"},
                {"kind": "entity_identity", "quote": "Unified Geologic Map of the Moon"},
                {"kind": "entity_identity", "quote": "Moon"},
                {"kind": "entity_identity", "quote": "USGS"},
            ]}], "logic": "single", "notes": ""}
        maximal = deepcopy(nested)
        maximal["claims"][0]["dimensions"] = [item for item in
            maximal["claims"][0]["dimensions"] if item["quote"] != "Moon"]
        client = ScriptClient(nested, maximal, accepted_extension)
        planner = e.TargetPlanner(client, max_structure_repairs=1)
        plan = planner.prepare(target)

        self.assertEqual(["claim_contract", "claim_contract", "extension"],
                         [item["stage"] for item in client.calls])
        self.assertEqual("structure_repair_requested", planner.history[0]["status"])
        self.assertIn("maximal non-overlapping", client.calls[1]["payload"]["repair"]["issue"])
        self.assertEqual({"Unified Geologic Map of the Moon", "USGS"},
                         {item.anchor.quote for item in plan.claims[0].dimensions
                          if item.kind == "entity_identity"})

    def test_exact_designation_requires_explicit_naming_assertion(self):
        ordinary_target = {**TARGET, "text": "The Artemis mission launched."}
        invalid = {"claims": [{"statement": ordinary_target["text"],
            "quote": ordinary_target["text"], "role": "main", "dimensions": [
                {"kind": "subject", "quote": "The Artemis mission"},
                {"kind": "predicate", "quote": "launched"},
                {"kind": "exact_designation", "quote": "Artemis"},
            ]}], "logic": "single", "notes": ""}
        planner = e.TargetPlanner(ScriptClient(invalid), max_structure_repairs=0)
        with self.assertRaisesRegex(e.TargetPlanningError, "exact_designation"):
            planner.prepare(ordinary_target)

        naming_target = {**TARGET, "text": "The mission was officially named Artemis II."}
        valid = {"claims": [{"statement": naming_target["text"],
            "quote": naming_target["text"], "role": "main", "dimensions": [
                {"kind": "subject", "quote": "The mission"},
                {"kind": "predicate", "quote": "was officially named"},
                {"kind": "exact_designation", "quote": "Artemis II"},
            ]}], "logic": "single", "notes": ""}
        plan = e.TargetPlanner(ScriptClient(valid, accepted_extension)).prepare(naming_target)
        probe = next(item for item in plan.probes if item.kind == "exact_designation")
        self.assertEqual("exact_designation", probe.match_policy)
        self.assertEqual(("atoms", "evidence", "world"), probe.routes)

    def test_rename_morphology_and_p07_shape_require_full_designation_relation(self):
        for predicate in (
                "name", "named", "names", "naming",
                "rename", "renamed", "renames", "renaming",
                "call", "called", "calls", "calling",
                "title", "titled", "titles", "titling",
                "designate", "designated", "designates", "designating",
                "label", "labeled", "labels", "labeling", "labelled", "labelling",
                "term", "termed", "terms", "terming"):
            with self.subTest(predicate=predicate):
                self.assertIsNotNone(e._DESIGNATION_ASSERTION_CUE.fullmatch(predicate))
        for non_cue in ("unnamed", "surname", "callback", "predesignated"):
            with self.subTest(non_cue=non_cue):
                self.assertIsNone(e._DESIGNATION_ASSERTION_CUE.search(non_cue))

        target = {**TARGET, "text":
            "NOAA renamed GOES-U to GOES-19 on June 25, 2024."}
        renamed = {"claims": [{"statement": target["text"],
            "quote": target["text"], "role": "main", "dimensions": [
                {"kind": "subject", "quote": "NOAA"},
                {"kind": "predicate", "quote": "renamed"},
                {"kind": "entity_identity", "quote": "NOAA"},
                {"kind": "entity_identity", "quote": "GOES-U"},
                {"kind": "exact_designation", "quote": "GOES-19"},
                {"kind": "time", "quote": "June 25, 2024"},
            ]}], "logic": "single", "notes": ""}
        client = ScriptClient(renamed, accepted_extension)
        plan = e.TargetPlanner(client).prepare(target)

        self.assertEqual(["claim_contract", "extension"],
                         [item["stage"] for item in client.calls])
        claim, = plan.claims
        designation, = [item for item in claim.dimensions
                        if item.kind == "exact_designation"]
        self.assertEqual("GOES-19", designation.anchor.quote)
        relation, = [item for item in plan.probes
                     if item.kind == "designation_relation"]
        self.assertEqual({item.id for item in claim.dimensions},
                         set(relation.dimension_ids))
        self.assertEqual("semantic_constraint", relation.match_policy)
        self.assertEqual(("atoms", "evidence", "world"), relation.routes)

        def disconnected(payload):
            answer = accepted_extension(payload)
            answer["probes"] = [item for item in answer["probes"]
                                if item["kind"] != "designation_relation"]
            return answer

        def partial_relation(payload):
            answer = accepted_extension(payload)
            relation_probe = next(item for item in answer["probes"]
                                  if item["kind"] == "designation_relation")
            relation_probe["dimension_ids"].pop()
            return answer

        for label, response in (("disconnected", disconnected),
                                ("partial", partial_relation)):
            with self.subTest(invalid_relation=label), self.assertRaisesRegex(
                    e.TargetPlanningError,
                    "missing required dimension-bound probes|binding does not match"):
                e.TargetPlanner(ScriptClient(renamed, response),
                                max_repairs=0).prepare(target)

    def test_p07_identity_questions_are_program_canonical_across_voice_variants(self):
        active_target = {**TARGET, "text":
            "NOAA renamed GOES-U to GOES-19 on June 25, 2024."}
        passive_target = {**TARGET, "text":
            "NOAA's GOES-U satellite was renamed GOES-19 on June 25, 2024."}

        def claim_contract(target, subject, predicate):
            return {"claims": [{"statement": target["text"],
                "quote": target["text"], "role": "main", "dimensions": [
                    {"kind": "subject", "quote": subject},
                    {"kind": "predicate", "quote": predicate},
                    {"kind": "entity_identity", "quote": "NOAA"},
                    {"kind": "entity_identity", "quote": "GOES-U"},
                    {"kind": "exact_designation", "quote": "GOES-19"},
                    {"kind": "time", "quote": "June 25, 2024"},
                ]}], "logic": "single", "notes": ""}

        def contaminated_extension(wording):
            def response(payload):
                answer = accepted_extension(payload)
                for probe in answer["probes"]:
                    if probe["kind"] in {"entity_identity", "exact_designation"}:
                        probe["question"] = wording
                return answer
            return response

        active_contract = claim_contract(active_target, "NOAA", "renamed")
        passive_contract = claim_contract(
            passive_target, "NOAA's GOES-U satellite", "was renamed")
        active = e.TargetPlanner(ScriptClient(active_contract,
            contaminated_extension("Did NOAA perform the renaming action?"))).prepare(
                active_target)
        active_again = e.TargetPlanner(ScriptClient(active_contract,
            contaminated_extension("Is the target subject also the actor?"))).prepare(
                active_target)
        passive = e.TargetPlanner(ScriptClient(passive_contract,
            contaminated_extension("Does the possessive make NOAA the actor?"))).prepare(
                passive_target)

        def question(plan, kind, quote):
            claim, = plan.claims
            dimension = next(item for item in claim.dimensions
                             if item.kind == kind and item.anchor.quote == quote)
            return next(item.question for item in plan.probes
                        if item.kind == kind and item.dimension_ids == (dimension.id,))

        expected_identity = e._canonical_identity_question(
            e.TextAnchor(0, 4, "NOAA"), "same_referent")
        expected_designation = e._canonical_identity_question(
            e.TextAnchor(0, 7, "GOES-19"), "exact_designation")
        for plan in (active, active_again, passive):
            self.assertEqual(expected_identity,
                             question(plan, "entity_identity", "NOAA"))
            self.assertEqual(expected_designation,
                             question(plan, "exact_designation", "GOES-19"))
            self.assertNotIn("actor", question(
                plan, "entity_identity", "NOAA").casefold())
            self.assertNotIn("renam", question(
                plan, "entity_identity", "NOAA").casefold())
        self.assertEqual([item.id for item in active.probes],
                         [item.id for item in active_again.probes])
        self.assertEqual(active.sha256, active_again.sha256)

    def test_p08_multiple_times_require_one_single_dimension_probe_each(self):
        target = {**TARGET, "text": ("The USGS Unified Geologic Map of the Moon released in "
            "2020 combined six Apollo-era regional maps with newer lunar-mission data.")}
        response = {"claims": [
            {"statement": "The map was released in 2020.",
             "quote": "The USGS Unified Geologic Map of the Moon released in 2020",
             "role": "conjunct", "dimensions": [
                 {"kind": "subject", "quote": "The USGS Unified Geologic Map of the Moon"},
                 {"kind": "predicate", "quote": "released"},
                 {"kind": "time", "quote": "in 2020"},
             ]},
            {"statement": ("The map combined six Apollo-era regional maps with newer "
                           "lunar-mission data."),
             "quote": target["text"], "role": "conjunct", "dimensions": [
                 {"kind": "subject", "quote": "The USGS Unified Geologic Map of the Moon"},
                 {"kind": "predicate", "quote": "combined"},
                 {"kind": "quantity_unit", "quote": "six Apollo-era regional maps"},
                 {"kind": "time", "quote": "Apollo-era regional maps"},
                 {"kind": "time", "quote": "newer lunar-mission data"},
             ]},
        ], "logic": "and", "notes": ""}
        plan = e.TargetPlanner(ScriptClient(response, accepted_extension)).prepare(target)

        times = [dimension for claim in plan.claims for dimension in claim.dimensions
                 if dimension.kind == "time"]
        time_probes = [probe for probe in plan.probes
                       if probe.kind == "time_boundary"]
        self.assertEqual(3, len(times))
        self.assertEqual({"in 2020", "Apollo-era regional maps",
                          "newer lunar-mission data"},
                         {item.anchor.quote for item in times})
        self.assertEqual({(item.id,) for item in times},
                         {item.dimension_ids for item in time_probes})
        self.assertTrue(all(len(item.dimension_ids) == 1 for item in time_probes))

        def merged_time_probe(payload):
            answer = accepted_extension(payload)
            combination = next(claim for claim in payload["claim_contract"]["claims"]
                               if "combined" in claim["statement"])
            time_ids = [item["id"] for item in combination["dimensions"]
                        if item["kind"] == "time"]
            answer["probes"] = [probe for probe in answer["probes"]
                                if not (probe["claim_id"] == combination["id"] and
                                        probe["kind"] == "time_boundary")]
            answer["probes"].append({"claim_id": combination["id"],
                "kind": "time_boundary", "dimension_ids": time_ids,
                "question": "Check both time qualifiers together.",
                "decision_impact": "A mismatch could change the decision."})
            return answer

        with self.assertRaisesRegex(e.TargetPlanningError,
                                    "binding does not match claim dimensions"):
            e.TargetPlanner(ScriptClient(response, merged_time_probe),
                            max_repairs=0, max_output_repairs=0).prepare(target)

        overlapping = deepcopy(response)
        combination = overlapping["claims"][1]
        combination["dimensions"][-2]["quote"] = (
            "Apollo-era regional maps with newer lunar-mission data")
        with self.assertRaisesRegex(e.TargetPlanningError, "overlap"):
            e.TargetPlanner(ScriptClient(overlapping),
                            max_structure_repairs=0).prepare(target)

    def test_designation_cue_must_be_the_designation_claim_predicate(self):
        target = {**TARGET, "text": "Mission A was named Artemis while Mission B launched."}
        wrong = {"claims": [{"statement": "Mission B launched.",
            "quote": target["text"], "role": "main", "dimensions": [
                {"kind": "subject", "quote": "Mission B"},
                {"kind": "predicate", "quote": "launched"},
                {"kind": "exact_designation", "quote": "Mission B"},
            ]}], "logic": "single", "notes": ""}
        planner = e.TargetPlanner(ScriptClient(wrong), max_structure_repairs=0)
        with self.assertRaisesRegex(e.TargetPlanningError, "predicate"):
            planner.prepare(target)

    def test_repeated_exact_designations_require_a_structure_repair(self):
        target = {**TARGET, "text": "The mission was named Artemis, not Apollo."}
        repeated = {"claims": [{"statement": target["text"], "quote": target["text"],
            "role": "main", "dimensions": [
                {"kind": "subject", "quote": "The mission"},
                {"kind": "predicate", "quote": "was named"},
                {"kind": "negation", "quote": "not"},
                {"kind": "exact_designation", "quote": "Artemis"},
                {"kind": "exact_designation", "quote": "Apollo"},
            ]}], "logic": "single", "notes": ""}
        planner = e.TargetPlanner(ScriptClient(repeated), max_structure_repairs=0)
        with self.assertRaisesRegex(e.TargetPlanningError, "multiple exact_designation"):
            planner.prepare(target)

    def test_exact_designation_without_predicate_is_a_controlled_planning_error(self):
        target = {**TARGET, "text": "The mission was named Artemis."}
        malformed = {"claims": [{"statement": target["text"], "quote": target["text"],
            "role": "main", "dimensions": [
                {"kind": "subject", "quote": "The mission"},
                {"kind": "exact_designation", "quote": "Artemis"},
            ]}], "logic": "single", "notes": ""}
        client = ScriptClient(malformed)
        with self.assertRaisesRegex(e.TargetPlanningError, "subject and predicate"):
            e.TargetPlanner(client, max_structure_repairs=0).prepare(target)
        self.assertEqual(1, len(client.calls))

    def test_attribution_predicate_crossing_that_boundary_is_repaired(self):
        target = {**TARGET, "text": "NASA reported that samples weighed 70 grams."}
        valid = {"claims": [
            {"statement": "NASA reported the content.", "quote": "NASA reported",
             "role": "attribution", "dimensions": [
                 {"kind": "subject", "quote": "NASA"},
                 {"kind": "predicate", "quote": "reported"}]},
            {"statement": "samples weighed 70 grams.", "quote": "samples weighed 70 grams.",
             "role": "attributed_content", "dimensions": [
                 {"kind": "subject", "quote": "samples"},
                 {"kind": "predicate", "quote": "weighed"},
                 {"kind": "quantity_unit", "quote": "70 grams"}]},
        ], "logic": "attribution", "notes": ""}
        crossed = deepcopy(valid)
        crossed["claims"][0]["quote"] = target["text"]
        crossed["claims"][0]["dimensions"][1]["quote"] = "reported that samples weighed"
        with self.assertRaises(e.TargetPlanningError):
            e.TargetPlanner(ScriptClient(crossed), max_structure_repairs=0).prepare(target)
        client = ScriptClient(crossed, valid, accepted_extension)
        planner = e.TargetPlanner(client, max_structure_repairs=1)
        plan = planner.prepare(target)
        self.assertEqual("structure_repair_requested", planner.history[0]["status"])
        parent = next(claim for claim in plan.claims if claim.role == "attribution")
        self.assertEqual("reported", next(dim.anchor.quote for dim in parent.dimensions
                                         if dim.kind == "predicate"))

    def test_explicit_reported_that_cannot_bypass_attribution_split(self):
        target = {**TARGET, "text": ("NASA's February 15, 2024 bulk-sample announcement "
            "reported that the final total mass of the Bennu sample returned by "
            "OSIRIS-REx was 70.3 grams.")}
        collapsed = {"claims": [{"statement": target["text"], "quote": target["text"],
            "role": "main", "dimensions": [
                {"kind": "subject", "quote": "NASA's February 15, 2024 bulk-sample announcement"},
                {"kind": "predicate", "quote": ("reported that the final total mass of the "
                    "Bennu sample returned by OSIRIS-REx was 70.3 grams")},
                {"kind": "time", "quote": "February 15, 2024"},
                {"kind": "quantity_unit", "quote": "70.3 grams"},
                {"kind": "entity_identity", "quote": "NASA"},
                {"kind": "entity_identity", "quote": "Bennu"},
                {"kind": "entity_identity", "quote": "OSIRIS-REx"},
            ]}], "logic": "single", "notes": ""}
        split = {"claims": [
            {"statement": "NASA's announcement reported the embedded content.",
             "quote": "NASA's February 15, 2024 bulk-sample announcement reported",
             "role": "attribution", "dimensions": [
                 {"kind": "subject", "quote": "NASA's February 15, 2024 bulk-sample announcement"},
                 {"kind": "predicate", "quote": "reported"},
                 {"kind": "time", "quote": "February 15, 2024"},
                 {"kind": "entity_identity", "quote": "NASA"},
             ]},
            {"statement": "The returned Bennu sample's final total mass was 70.3 grams.",
             "quote": ("the final total mass of the Bennu sample returned by OSIRIS-REx "
                       "was 70.3 grams."),
             "role": "attributed_content", "dimensions": [
                 {"kind": "subject", "quote": ("the final total mass of the Bennu sample "
                     "returned by OSIRIS-REx")},
                 {"kind": "predicate", "quote": "was"},
                 {"kind": "quantity_unit", "quote": "70.3 grams"},
                 {"kind": "baseline_scope", "quote": "final total mass"},
                 {"kind": "entity_identity", "quote": "Bennu"},
                 {"kind": "entity_identity", "quote": "OSIRIS-REx"},
             ]},
        ], "logic": "attribution", "notes": ""}
        client = ScriptClient(collapsed, split, accepted_extension)
        planner = e.TargetPlanner(client, max_structure_repairs=1)
        plan = planner.prepare(target)

        self.assertEqual(["claim_contract", "claim_contract", "extension"],
                         [item["stage"] for item in client.calls])
        self.assertEqual("structure_repair_requested", planner.history[0]["status"])
        attribution = next(claim for claim in plan.claims if claim.role == "attribution")
        content = next(claim for claim in plan.claims if claim.role == "attributed_content")
        self.assertEqual(attribution.id, content.parent_claim_id)
        self.assertIsNone(attribution.parent_claim_id)
        self.assertEqual(attribution.id,
                         next(item for item in plan.to_payload()["claims"]
                              if item["role"] == "attributed_content")["parent_claim_id"])

    def test_attribution_parent_cannot_absorb_child_entity_dimensions(self):
        target = {**TARGET, "text": "NASA reported that Bennu samples weighed 70 grams."}
        leaked = {"claims": [
            {"statement": "NASA reported the content.", "quote": target["text"],
             "role": "attribution", "dimensions": [
                 {"kind": "subject", "quote": "NASA"},
                 {"kind": "predicate", "quote": "reported"},
                 {"kind": "entity_identity", "quote": "NASA"},
                 {"kind": "entity_identity", "quote": "Bennu"},
             ]},
            {"statement": "Bennu samples weighed 70 grams.",
             "quote": "Bennu samples weighed 70 grams.",
             "role": "attributed_content", "dimensions": [
                 {"kind": "subject", "quote": "Bennu samples"},
                 {"kind": "predicate", "quote": "weighed"},
                 {"kind": "quantity_unit", "quote": "70 grams"},
                 {"kind": "entity_identity", "quote": "Bennu"},
             ]},
        ], "logic": "attribution", "notes": ""}
        repaired = deepcopy(leaked)
        repaired["claims"][0]["quote"] = "NASA reported"
        repaired["claims"][0]["dimensions"] = [item for item in
            repaired["claims"][0]["dimensions"] if item["quote"] != "Bennu"]
        planner = e.TargetPlanner(
            ScriptClient(leaked, repaired, accepted_extension),
            max_structure_repairs=1)
        plan = planner.prepare(target)
        parent = next(item for item in plan.claims if item.role == "attribution")
        self.assertEqual({"NASA"}, {item.anchor.quote for item in parent.dimensions
                                    if item.kind == "entity_identity"})
        self.assertEqual("structure_repair_requested", planner.history[0]["status"])

    def test_structure_repair_budget_exhaustion_never_reaches_extension(self):
        collapsed = contract()
        collapsed["claims"][0]["dimensions"].append(
            {"kind": "predicate", "quote": "did not rise"})
        client = ScriptClient(collapsed, collapsed)
        planner = e.TargetPlanner(client, max_structure_repairs=1)
        with self.assertRaisesRegex(e.TargetPlanningError,
                                    "structure repair budget exhausted"):
            planner.prepare(TARGET)
        self.assertEqual(["claim_contract", "claim_contract"],
                         [item["stage"] for item in client.calls])
        self.assertEqual("structure_repair_exhausted", planner.history[-1]["status"])

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
                           "dimension_ids": [],
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
            ("missing", missing, "missing required dimension-bound probes"),
        ):
            with self.subTest(label=label):
                planner = e.TargetPlanner(
                    ScriptClient(contract(), response), max_repairs=0)
                with self.assertRaisesRegex(e.TargetPlanningError, message):
                    planner.prepare(TARGET)

    def test_probe_must_bind_the_exact_dimensions_it_checks(self):
        def wrong_binding(payload):
            answer = accepted_extension(payload)
            claim = payload["claim_contract"]["claims"][0]
            subject_id = next(item["id"] for item in claim["dimensions"]
                              if item["kind"] == "subject")
            time_probe = next(item for item in answer["probes"]
                              if item["kind"] == "time_boundary")
            time_probe["dimension_ids"] = [subject_id]
            return answer

        planner = e.TargetPlanner(
            ScriptClient(contract(), wrong_binding), max_repairs=0)
        with self.assertRaisesRegex(e.TargetPlanningError,
                                    "binding does not match claim dimensions"):
            planner.prepare(TARGET)

    def test_invalid_probe_set_gets_one_local_complete_replacement(self):
        def extra_relation(payload):
            answer = accepted_extension(payload)
            claim = payload["claim_contract"]["claims"][0]
            answer["probes"].append({
                "claim_id": claim["id"], "kind": "comparison_relation",
                "dimension_ids": [item["id"] for item in claim["dimensions"]],
                "question": "Incorrect optional comparison probe.",
                "decision_impact": "This extra probe must not enter the plan.",
            })
            return answer

        client = ScriptClient(contract(), extra_relation, accepted_extension)
        planner = e.TargetPlanner(client, max_repairs=1)
        plan = planner.prepare(TARGET)

        self.assertEqual(["claim_contract", "extension", "extension"],
                         [call["stage"] for call in client.calls])
        self.assertIsNone(client.calls[1]["payload"]["repair"])
        self.assertIn("binding does not match", client.calls[2]["payload"]["repair"]["issue"])
        self.assertEqual(client.calls[1]["payload"]["claim_contract"],
                         client.calls[2]["payload"]["claim_contract"])
        self.assertEqual("output_repair_requested", planner.history[1]["status"])
        self.assertEqual("accepted", planner.history[2]["status"])
        self.assertNotIn("comparison_relation", {probe.kind for probe in plan.probes})

    def test_contract_and_output_repairs_have_separate_budgets(self):
        contract_repair = {
            "decision": "repair",
            "repair_quote": "above the 2024 baseline",
            "repair_issue": "The comparison baseline is missing.",
            "probes": [],
            "notes": "",
        }

        def invalid_probe_set(payload):
            answer = accepted_extension(payload)
            claim = payload["claim_contract"]["claims"][0]
            answer["probes"].append({
                "claim_id": claim["id"],
                "kind": "comparison_relation",
                "dimension_ids": [item["id"] for item in claim["dimensions"]],
                "question": "An illegal optional relation probe.",
                "decision_impact": "The set must be replaced without changing claims.",
            })
            return answer

        client = ScriptClient(
            contract(include_baseline=False),
            contract_repair,
            contract(include_baseline=True),
            invalid_probe_set,
            accepted_extension,
        )
        planner = e.TargetPlanner(
            client,
            max_repairs=1,
            max_structure_repairs=1,
            max_output_repairs=1,
        )
        plan = planner.prepare(TARGET)

        self.assertEqual(
            ["claim_contract", "extension", "claim_contract", "extension", "extension"],
            [call["stage"] for call in client.calls],
        )
        self.assertEqual(1, sum(item["status"] == "repair_requested"
                                for item in planner.history))
        self.assertEqual(1, sum(item["status"] == "output_repair_requested"
                                for item in planner.history))
        self.assertEqual("accepted", planner.history[-1]["status"])
        self.assertIn("baseline_scope", {dimension.kind
                                          for dimension in plan.claims[0].dimensions})

    def test_output_repair_cannot_consume_later_contract_repair(self):
        def invalid_probe_set(payload):
            answer = accepted_extension(payload)
            claim = payload["claim_contract"]["claims"][0]
            answer["probes"].append({
                "claim_id": claim["id"],
                "kind": "comparison_relation",
                "dimension_ids": [item["id"] for item in claim["dimensions"]],
                "question": "An illegal optional relation probe.",
                "decision_impact": "The set must be replaced without changing claims.",
            })
            return answer

        contract_repair = {
            "decision": "repair",
            "repair_quote": "above the 2024 baseline",
            "repair_issue": "The comparison baseline is missing.",
            "probes": [],
            "notes": "",
        }
        client = ScriptClient(
            contract(include_baseline=False),
            invalid_probe_set,
            contract_repair,
            contract(include_baseline=True),
            accepted_extension,
        )
        planner = e.TargetPlanner(
            client,
            max_repairs=1,
            max_structure_repairs=1,
            max_output_repairs=1,
        )
        plan = planner.prepare(TARGET)

        self.assertEqual(
            ["claim_contract", "extension", "extension", "claim_contract", "extension"],
            [call["stage"] for call in client.calls],
        )
        self.assertEqual(1, sum(item["status"] == "output_repair_requested"
                                for item in planner.history))
        self.assertEqual(1, sum(item["status"] == "repair_requested"
                                for item in planner.history))
        self.assertEqual("accepted", planner.history[-1]["status"])
        self.assertIn("baseline_scope", {dimension.kind
                                          for dimension in plan.claims[0].dimensions})

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
                                   "dimension_ids": ([] if kind == "source_lineage" else
                                       [item["id"] for item in claim["dimensions"]
                                        if item["kind"] in {"subject", "predicate"}]),
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

    def test_relational_contract_requires_complete_matching_relation_probe(self):
        examples = (
            ("conditional", "Output rises if demand grows.", "Output", "rises",
             "condition", "if demand grows"),
            ("comparison", "Output exceeded the previous year's total.", "Output", "exceeded",
             "baseline_scope", "the previous year's total"),
            ("causal", "Rain caused the increase in river flow.", "Rain", "caused",
             "baseline_scope", "the increase in river flow"),
        )
        for logic, text, subject, predicate, extra_kind, extra_quote in examples:
            with self.subTest(logic=logic):
                target = {**TARGET, "text": text}
                raw = {"claims": [{"statement": text, "quote": text, "role": "main",
                    "dimensions": [{"kind": "subject", "quote": subject},
                        {"kind": "predicate", "quote": predicate},
                        {"kind": extra_kind, "quote": extra_quote}]}],
                    "logic": logic, "notes": ""}
                plan = e.TargetPlanner(ScriptClient(raw, accepted_extension)).prepare(target)
                relation = next(probe for probe in plan.probes
                                if probe.kind == logic + "_relation")
                self.assertEqual({dim.id for dim in plan.claims[0].dimensions},
                                 set(relation.dimension_ids))
                self.assertEqual(("atoms", "evidence", "world"), relation.routes)
                for problem in ("missing", "partial_binding", "wrong_relation"):
                    def malformed(payload, problem=problem):
                        result = accepted_extension(payload)
                        probe = next(item for item in result["probes"]
                                     if item["kind"] == logic + "_relation")
                        if problem == "missing":
                            result["probes"].remove(probe)
                        elif problem == "partial_binding":
                            probe["dimension_ids"].pop()
                        else:
                            probe["kind"] = ("causal_relation" if logic != "causal"
                                             else "comparison_relation")
                        return result
                    with self.subTest(problem=problem), self.assertRaises(e.TargetPlanningError):
                        e.TargetPlanner(ScriptClient(raw, malformed),
                            max_repairs=0).prepare(target)

    def test_projection_detects_tampered_frozen_plan_copy(self):
        plan = e.TargetPlanner(ScriptClient(contract(), accepted_extension)).prepare(TARGET)
        altered = replace(plan, notes="silently changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            e.project_plan(altered, "atoms")


if __name__ == "__main__":
    unittest.main()
