"""Artifact-only tests for generic frozen target-extension smoke gates."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from target_extension_smoke_gate import (PLAN_SCHEMA_V3, PLAN_SCHEMA_V4, _Checks,
                                         _canonical_identity_question,
                                         _checkpoint_audit, _exact_config,
                                         _evaluate_gate,
                                         _execution_control_declarations,
                                         _frozen_baseline_audit,
                                         _frozen_prior_artifact_audit,
                                         _one_sentence_has_p07_naming_evidence,
                                         _outer_loop_audit,
                                         _p02_checks, _p04_checks, _p07_checks, _p08_checks,
                                         _preserved_run_declarations, _smoke_label_audit,
                                         _secret_shape_audit,
                                         _v4_audit_state,
                                         check_smoke_gate)


class TargetExtensionSmokeGateTests(unittest.TestCase):
    FREEZE = ROOT / "experiments" / "target-extension-freeze-07.json"
    GOLD = ROOT / "experiments" / "proof_pilot" / "gold.json"
    ORIGINAL = ROOT / "reports" / "staged-baseline-01"
    STAGED = ROOT / "reports" / "staged-dev-04"
    SMOKE = ROOT / "reports" / "target-extension-v2-smoke-03"

    @staticmethod
    def read(path):
        return json.loads(Path(path).read_text())

    @staticmethod
    def item(checks, identifier):
        return next(item for item in checks.items if item["id"] == identifier)

    def test_completed_smoke_passes_every_mechanical_conjunct(self):
        result = check_smoke_gate(self.FREEZE, self.GOLD, self.ORIGINAL,
                                  self.STAGED, self.SMOKE, hash_mode="executed")
        self.assertTrue(result["passed"])
        self.assertEqual((35, 35, []), (result["checks_run"],
                         result["checks_passed"], result["failed_checks"]))
        self.assertEqual(2, len(result["uncheckable_clauses"]))
        contract = next(item for item in result["checks"]
                        if item["id"] == "probe_audit.frozen_plans")
        self.assertTrue(contract["passed"])
        self.assertIn("decision-probe-v2", contract["expected"])

    def test_incomplete_preserved_smoke_fails_closed(self):
        result = check_smoke_gate(self.FREEZE, self.GOLD, self.ORIGINAL,
                                  self.STAGED,
                                  ROOT / "reports" / "target-extension-v2-smoke-02",
                                  hash_mode="executed")
        self.assertFalse(result["passed"])
        self.assertIn("execution.required_artifacts", result["failed_checks"])
        self.assertIn("comparison.independent_audit", result["failed_checks"])

    def test_duplicate_snapshot_reanalysis_is_not_a_repeated_url_fetch(self):
        plan = self.read(self.SMOKE / "p08-target-plan.json")
        report = self.read(self.SMOKE / "p08-report.json")
        inputs = self.read(self.SMOKE / "inputs.json")
        case = next(item for item in inputs["cases"] if item["target"]["id"] == "p08")
        self.assertTrue(any(item.get("action") == "duplicate_observed"
                            for item in report["operations"]))
        checks = _Checks()
        _p08_checks(checks, plan, report, case, [1, 2])
        self.assertTrue(self.item(checks, "p08.no_repeated_visible_fetch")["passed"])

    def test_actual_fetch_of_visible_snapshot_url_is_rejected(self):
        plan = self.read(self.SMOKE / "p08-target-plan.json")
        report = self.read(self.SMOKE / "p08-report.json")
        inputs = self.read(self.SMOKE / "inputs.json")
        case = next(item for item in inputs["cases"] if item["target"]["id"] == "p08")
        m15_url = next(item["url"] for item in case["materials"]
                       if item["version_id"] == "m15")
        changed = deepcopy(report)
        changed["operations"].append({"sequence": 999, "round": 2,
            "action": "search", "tasks": [{"action": "fetch", "locator": m15_url}]})
        checks = _Checks()
        _p08_checks(checks, plan, changed, case, [1, 2])
        item = self.item(checks, "p08.no_repeated_visible_fetch")
        self.assertFalse(item["passed"])
        self.assertEqual(m15_url, item["actual"][0]["locator"])

    def p08_v3_fixture(self):
        plan = self.read(self.SMOKE / "p08-target-plan.json")
        report = self.read(self.SMOKE / "p08-report.json")
        inputs = self.read(self.SMOKE / "inputs.json")
        case = next(item for item in inputs["cases"] if item["target"]["id"] == "p08")
        plan["schema_version"] = PLAN_SCHEMA_V3
        combination = next(claim for claim in plan["claims"]
                           if "combined" in claim["statement"])
        old_time = next(item for item in combination["dimensions"]
                        if item["kind"] == "time")
        target_text = case["target"]["text"]
        apollo_quote = "Apollo-era regional maps"
        apollo_start = target_text.index(apollo_quote)
        old_time["anchor"] = {"start": apollo_start,
                              "end": apollo_start + len(apollo_quote),
                              "quote": apollo_quote}
        newer_quote = "newer lunar-mission data"
        newer_start = target_text.index(newer_quote)
        newer_time = {"id": "p08:dimension:newer-v3", "kind": "time",
                      "anchor": {"start": newer_start,
                                 "end": newer_start + len(newer_quote),
                                 "quote": newer_quote}}
        combination["dimensions"].append(newer_time)
        old_probe = next(item for item in plan["probes"]
                         if item["kind"] == "time_boundary" and
                         item["dimension_ids"] == [old_time["id"]])
        newer_probe = deepcopy(old_probe)
        newer_probe.update(id="p08:probe:newer-v3",
                           dimension_ids=[newer_time["id"]],
                           question="Does the evidence establish newer lunar-mission data?")
        plan["probes"].append(newer_probe)
        for record in report["verification_history"]:
            for stage in ("evidence", "world"):
                old_result = next(item for item in record[stage + "_probe_results"]
                                  if item["probe_id"] == old_probe["id"])
                new_result = deepcopy(old_result)
                new_result["probe_id"] = newer_probe["id"]
                record[stage + "_probe_results"].append(new_result)
        report["origins"] = [item for item in report["origins"]
                             if item["version_id"] == "m15"]
        return plan, report, case, old_probe["id"], newer_probe["id"]

    def test_p08_v3_terminal_root_independent_times_and_results_pass(self):
        plan, report, case, _, _ = self.p08_v3_fixture()
        checks = _Checks()
        _p08_checks(checks, plan, report, case, [1, 2])
        v3 = [item for item in checks.items if item["id"].startswith("p08.v3_")]
        self.assertEqual(6, len(v3))
        self.assertTrue(all(item["passed"] for item in v3), v3)

        changed = deepcopy(report)
        changed["gaps"] = [{"id": "lineage:p08", "stage": "provenance",
                            "blocking": True}]
        failed = _Checks()
        _p08_checks(failed, plan, changed, case, [1, 2])
        self.assertFalse(self.item(
            failed, "p08.v3_no_active_lineage_or_time_gap")["passed"])

    def test_p08_v4_retains_the_v3_root_time_and_gap_regressions(self):
        plan, report, case, _, _ = self.p08_v3_fixture()
        plan["schema_version"] = PLAN_SCHEMA_V4
        checks = _Checks()
        _p08_checks(checks, plan, report, case, [1, 2])
        inherited = [item for item in checks.items if item["id"].startswith("p08.v3_")]
        self.assertEqual(6, len(inherited))
        self.assertTrue(all(item["passed"] for item in inherited), inherited)

    def test_p08_v3_rejects_intermediate_root_merged_time_and_bad_result(self):
        plan, report, case, old_probe_id, newer_probe_id = self.p08_v3_fixture()
        report["origins"].append({"version_id": "m14"})
        report["analyses"]["m14"]["origins"] = []
        merged = next(item for item in plan["probes"] if item["id"] == old_probe_id)
        newer = next(item for item in plan["probes"] if item["id"] == newer_probe_id)
        merged["dimension_ids"].extend(newer["dimension_ids"])
        plan["probes"].remove(newer)
        result = next(item for item in report["verification_history"][1]
                      ["world_probe_results"] if item["probe_id"] == newer_probe_id)
        result["status"] = "unresolved"
        checks = _Checks()
        _p08_checks(checks, plan, report, case, [1, 2])
        for identifier in ("p08.v3_terminal_origin", "p08.v3_raw_origin_candidates",
                           "p08.v3_singleton_time_probes",
                           "p08.v3_time_results_each_round"):
            self.assertFalse(self.item(checks, identifier)["passed"], identifier)

    def test_p02_final_total_must_remain_contradicted_in_every_layer_and_round(self):
        plan = self.read(self.SMOKE / "p02-target-plan.json")
        report = self.read(self.SMOKE / "p02-report.json")
        inputs = self.read(self.SMOKE / "inputs.json")
        target = next(item["target"] for item in inputs["cases"]
                      if item["target"]["id"] == "p02")
        changed = deepcopy(report)
        probe_id = next(item["id"] for item in plan["probes"]
                        if item["kind"] == "baseline_scope")
        result = next(item for item in changed["verification_history"][1]
                      ["world_probe_results"] if item["probe_id"] == probe_id)
        result["status"] = "supported"
        checks = _Checks()
        _p02_checks(checks, plan, changed, target, [1, 2])
        self.assertFalse(self.item(checks, "p02.final_total_each_round")["passed"])

    def test_checkpoint_audit_uses_canonical_payload_hash(self):
        rows = {item["id"]: item for item in self.read(self.SMOKE / "results.json")}
        self.assertEqual([], _checkpoint_audit(self.SMOKE, rows,
                                                ["p02", "p08"], [1, 2]))

    @staticmethod
    def p07_fixture():
        text = "NOAA renamed GOES-U to GOES-19 on June 25, 2024."
        dimensions = [
            {"id": "subject", "kind": "subject",
             "anchor": {"start": 0, "end": 4, "quote": "NOAA"}},
            {"id": "predicate", "kind": "predicate",
             "anchor": {"start": 5, "end": 12, "quote": "renamed"}},
            {"id": "old", "kind": "entity_identity",
             "anchor": {"start": 13, "end": 19, "quote": "GOES-U"}},
            {"id": "new", "kind": "exact_designation",
             "anchor": {"start": 23, "end": 30, "quote": "GOES-19"}},
            {"id": "time", "kind": "time",
             "anchor": {"start": 34, "end": 47, "quote": "June 25, 2024"}},
        ]
        claim = {"id": "claim", "statement": text,
                 "anchor": {"start": 0, "end": len(text), "quote": text},
                 "role": "main", "dimensions": dimensions, "parent_claim_id": None}
        probes = [
            {"id": "old-probe", "claim_id": "claim", "kind": "entity_identity",
             "dimension_ids": ["old"], "match_policy": "same_referent",
             "routes": ["atoms", "evidence", "world"], "gate": "always"},
            {"id": "new-probe", "claim_id": "claim", "kind": "exact_designation",
             "dimension_ids": ["new"], "match_policy": "exact_designation",
             "routes": ["atoms", "evidence", "world"], "gate": "always"},
            {"id": "time-probe", "claim_id": "claim", "kind": "time_boundary",
             "dimension_ids": ["time"], "match_policy": "semantic_constraint",
             "routes": ["atoms", "evidence", "world"], "gate": "always"},
            {"id": "relation-probe", "claim_id": "claim", "kind": "designation_relation",
             "dimension_ids": [item["id"] for item in dimensions],
             "match_policy": "semantic_constraint",
             "routes": ["atoms", "evidence", "world"], "gate": "always"},
        ]
        plan = {"logic": "single", "claims": [claim], "probes": probes}
        basis = [{"version_id": "m13", "start": 0, "end": 12,
                  "quote": "On July 7, 2024, GOES-U was renamed GOES-19."}]

        def result(probe, stage):
            naming = probe["id"] in {"old-probe", "new-probe"}
            relation = ("exact" if probe["id"] == "new-probe" else
                        "description" if probe["id"] == "old-probe" else
                        "not_applicable")
            return {"probe_id": probe["id"], "claim_id": "claim", "stage": stage,
                    "status": "supported" if naming else "contradicted",
                    "basis": basis, "rationale": "Synthetic exact check.",
                    "referent_relation": relation}

        history = []
        for round_number in (1, 2):
            history.append({"round": round_number, "gaps": [],
                **{stage + "_probe_results": [result(probe, stage) for probe in probes]
                   for stage in ("evidence", "world")}})
        report = {"verification_history": history, "gaps": [], "gap_registry": []}
        return plan, report

    def test_synthetic_p07_contract_passes_offline_without_smoke04(self):
        plan, report = self.p07_fixture()
        checks = _Checks()
        _p07_checks(checks, plan, report, [1, 2])
        self.assertTrue(all(item["passed"] for item in checks.items), checks.items)

    def p07_v3_fixture(self):
        plan, report = self.p07_fixture()
        plan["schema_version"] = PLAN_SCHEMA_V3
        claim = plan["claims"][0]
        noaa = {"id": "noaa", "kind": "entity_identity",
                "anchor": {"start": 0, "end": 4, "quote": "NOAA"}}
        claim["dimensions"].append(noaa)
        relation = next(item for item in plan["probes"]
                        if item["kind"] == "designation_relation")
        relation["dimension_ids"].append(noaa["id"])
        noaa_probe = {"id": "noaa-probe", "claim_id": claim["id"],
                      "kind": "entity_identity", "dimension_ids": [noaa["id"]],
                      "question": _canonical_identity_question(
                          noaa["anchor"], "same_referent"),
                      "decision_impact": "Tests only NOAA's referent.",
                      "match_policy": "same_referent",
                      "routes": ["atoms", "evidence", "world"], "gate": "always"}
        semantic_probe = {"id": "semantic-probe", "claim_id": claim["id"],
                          "kind": "semantic_core",
                          "dimension_ids": ["subject", "predicate"],
                          "question": "Did NOAA perform the renaming?",
                          "decision_impact": "Tests the actor-action relation.",
                          "match_policy": "semantic_constraint",
                          "routes": ["atoms", "evidence", "world"], "gate": "always"}
        plan["probes"].extend((noaa_probe, semantic_probe))
        basis = [{"version_id": "m13", "start": 0, "end": 12,
                  "quote": "On July 7, 2024, NOAA's GOES-U was renamed GOES-19."}]
        for record in report["verification_history"]:
            for stage in ("evidence", "world"):
                record[stage + "_probe_results"].extend((
                    {"probe_id": noaa_probe["id"], "claim_id": claim["id"],
                     "stage": stage, "status": "supported", "basis": basis,
                     "rationale": "NOAA is an exact referent mention.",
                     "referent_relation": "exact"},
                    {"probe_id": semantic_probe["id"], "claim_id": claim["id"],
                     "stage": stage, "status": "unresolved", "basis": basis,
                     "rationale": "Passive wording does not identify the actor.",
                     "referent_relation": "not_applicable"},
                ))
        return plan, report

    def test_p07_v3_identity_and_actor_checks_are_separate(self):
        plan, report = self.p07_v3_fixture()
        checks = _Checks()
        _p07_checks(checks, plan, report, [1, 2])
        for identifier in ("p07.v3_noaa_identity_contract",
                           "p07.v3_noaa_identity_results",
                           "p07.v3_actor_role_separate"):
            self.assertTrue(self.item(checks, identifier)["passed"], checks.items)

        contaminated_plan = deepcopy(plan)
        next(item for item in contaminated_plan["probes"]
             if item["id"] == "noaa-probe")["question"] = "Did NOAA rename GOES-U?"
        contaminated_report = deepcopy(report)
        next(item for item in contaminated_report["verification_history"][0]
             ["evidence_probe_results"]
             if item["probe_id"] == "noaa-probe")["referent_relation"] = "alias"
        next(item for item in contaminated_report["verification_history"][1]
             ["world_probe_results"]
             if item["probe_id"] == "semantic-probe")["status"] = "supported"
        changed = _Checks()
        _p07_checks(changed, contaminated_plan, contaminated_report, [1, 2])
        self.assertFalse(self.item(changed, "p07.v3_noaa_identity_contract")["passed"])
        self.assertFalse(self.item(changed, "p07.v3_noaa_identity_results")["passed"])
        self.assertFalse(self.item(changed, "p07.v3_actor_role_separate")["passed"])

        weak_basis = deepcopy(report)
        identity = next(item for item in weak_basis["verification_history"][0]
                        ["evidence_probe_results"]
                        if item["probe_id"] == "noaa-probe")
        identity["basis"][0]["quote"] = "The publisher is identified elsewhere."
        weak = _Checks()
        _p07_checks(weak, plan, weak_basis, [1, 2])
        self.assertFalse(self.item(weak, "p07.v3_noaa_identity_results")["passed"])

    def p07_v4_fixture(self):
        plan, report = self.p07_v3_fixture()
        plan["schema_version"] = PLAN_SCHEMA_V4
        claim = plan["claims"][0]
        actor = {"id": "actor", "kind": "actor_role",
                 "anchor": {"start": 0, "end": 4, "quote": "NOAA"}}
        claim["dimensions"].append(actor)
        claim["dimensions"].sort(key=lambda item: (
            item["anchor"]["start"], item["anchor"]["end"], item["kind"], item["id"]))
        relation = next(item for item in plan["probes"]
                        if item["kind"] == "designation_relation")
        relation["dimension_ids"].append(actor["id"])
        semantic = next(item for item in plan["probes"]
                        if item["id"] == "semantic-probe")
        semantic.update(
            id="actor-probe", kind="actor_role", dimension_ids=[actor["id"]],
            question=("Does the evidence establish \"NOAA\" in the actor or agent role "
                      "asserted by target[0:48]?"),
            decision_impact="A mismatched actor prevents support.",
        )
        for record in report["verification_history"]:
            for stage in ("evidence", "world"):
                result = next(item for item in record[stage + "_probe_results"]
                              if item["probe_id"] == "semantic-probe")
                result["probe_id"] = "actor-probe"
        return plan, report

    def test_p07_v4_requires_distinct_actor_role_and_identity_contracts(self):
        plan, report = self.p07_v4_fixture()
        checks = _Checks()
        _p07_checks(checks, plan, report, [1, 2])
        for identifier in ("p07.v4_noaa_identity_contract",
                           "p07.v4_noaa_identity_results",
                           "p07.v4_actor_role_contract",
                           "p07.v4_actor_role_results"):
            self.assertTrue(self.item(checks, identifier)["passed"], checks.items)

        missing_actor = deepcopy(plan)
        missing_actor["claims"][0]["dimensions"] = [
            item for item in missing_actor["claims"][0]["dimensions"]
            if item["kind"] != "actor_role"]
        next(item for item in missing_actor["probes"]
             if item["kind"] == "actor_role")["kind"] = "semantic_core"
        missing_checks = _Checks()
        _p07_checks(missing_checks, missing_actor, report, [1, 2])
        self.assertFalse(self.item(
            missing_checks, "p07.v4_actor_role_contract")["passed"])

        changed_report = deepcopy(report)
        next(item for item in changed_report["verification_history"][1]
             ["world_probe_results"]
             if item["probe_id"] == "actor-probe")["status"] = "supported"
        result_checks = _Checks()
        _p07_checks(result_checks, plan, changed_report, [1, 2])
        self.assertFalse(self.item(
            result_checks, "p07.v4_actor_role_results")["passed"])

        stale_identity_gap = deepcopy(report)
        stale_identity_gap["gap_registry"] = [
            {"id": "gap:noaa", "probe_id": "noaa-probe"}]
        gap_checks = _Checks()
        _p07_checks(gap_checks, plan, stale_identity_gap, [1, 2])
        self.assertFalse(self.item(gap_checks, "p07.no_relevant_gaps")["passed"])

    def test_p04_rejects_unrelated_m08_origin(self):
        checks = _Checks()
        _p04_checks(checks, {"origins": [{"version_id": "m07"}]})
        self.assertTrue(self.item(checks, "p04.no_m08_origin")["passed"])
        changed = _Checks()
        _p04_checks(changed, {"origins": [{"version_id": "m07"},
                                           {"version_id": "m08"}]})
        self.assertFalse(self.item(changed, "p04.no_m08_origin")["passed"])

    def test_p04_crossed_magnitude_and_baseline_are_independently_audited(self):
        run = ROOT / "reports" / "target-extension-v2-dev-02"
        plan = self.read(run / "p04-target-plan.json")
        report = self.read(run / "p04-report.json")
        checks = _Checks()
        _p04_checks(checks, report, plan, [1, 2])
        for identifier in ("p04.crossed_variables_separate",
                           "p04.crossed_variable_probes",
                           "p04.crossed_variable_results"):
            self.assertTrue(self.item(checks, identifier)["passed"])

        changed = deepcopy(report)
        quantity_id = next(item["id"] for item in plan["probes"]
                           if item["kind"] == "quantity_unit")
        result = next(item for item in changed["verification_history"][0]
                      ["evidence_probe_results"] if item["probe_id"] == quantity_id)
        result["status"] = "supported"
        failed = _Checks()
        _p04_checks(failed, changed, plan, [1, 2])
        self.assertFalse(self.item(
            failed, "p04.crossed_variable_results")["passed"])

        weak_pair = deepcopy(report)
        result = next(item for item in weak_pair["verification_history"][0]
                      ["evidence_probe_results"] if item["probe_id"] == quantity_id)
        for span in result["basis"]:
            span["quote"] = span["quote"].replace("1900", "1901")
        failed_pair = _Checks()
        _p04_checks(failed_pair, weak_pair, plan, [1, 2])
        self.assertFalse(self.item(
            failed_pair, "p04.crossed_variable_results")["passed"])

    def test_synthetic_p07_partial_relation_wrong_date_and_identity_gap_fail(self):
        plan, report = self.p07_fixture()
        changed_plan = deepcopy(plan)
        next(item for item in changed_plan["probes"]
             if item["kind"] == "designation_relation")["dimension_ids"].pop()
        changed_report = deepcopy(report)
        relation = next(item for item in changed_report["verification_history"][0]
                        ["evidence_probe_results"]
                        if item["probe_id"] == "relation-probe")
        relation["basis"][0]["quote"] = "GOES-U was renamed GOES-19."
        changed_report["gap_registry"] = [{"id": "time-gap",
            "probe_id": "time-probe", "dimension": "time_boundary"}]
        checks = _Checks()
        _p07_checks(checks, changed_plan, changed_report, [1, 2])
        self.assertFalse(self.item(checks, "p07.designation_relation")["passed"])
        self.assertFalse(self.item(checks, "p07.time_and_designation_contradictions")["passed"])
        self.assertFalse(self.item(checks, "p07.no_relevant_gaps")["passed"])

    def test_freeze08_transport_and_every_repair_budget_are_declared(self):
        freeze = self.read(ROOT / "experiments" / "target-extension-freeze-08.json")
        freeze["max_probe_result_structure_repairs"] = 1
        config = deepcopy(self.read(self.SMOKE / "config.json"))
        config.update(case_ids=freeze["smoke_cases"],
                      scheduled_cases=len(freeze["smoke_cases"]),
                      workers=freeze["workers"],
                      automatic_transport_retries=freeze["automatic_transport_retries"],
                      target_extension_output_repairs=freeze["max_extension_output_repairs"])
        self.assertTrue(_exact_config(freeze, config)[0])
        ok, expected, actual = _execution_control_declarations(freeze, config, self.SMOKE)
        self.assertTrue(ok, (expected, actual))
        mutations = (
            ("automatic_transport_retries", 1),
            ("target_structure_repairs", 0),
            ("target_extension_repairs", 0),
            ("target_extension_output_repairs", 0),
            ("material_stage_repairs", 0),
            ("max_inner_repairs", 0),
            ("judgement_repairs", 0),
            ("probe_result_structure_repairs", 0),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                changed = deepcopy(config)
                changed[field] = value
                self.assertFalse(_execution_control_declarations(
                    freeze, changed, self.SMOKE)[0])

    def test_current_and_legacy_preserved_run_fields_are_combined(self):
        value = _preserved_run_declarations({
            "preserved_failed_runs": {"old": "legacy"},
            "preserved_runs": {"new": "current"},
        })
        self.assertEqual({"old": "legacy", "new": "current"}, value)
        freeze = self.read(ROOT / "experiments" / "target-extension-freeze-08.json")
        self.assertIn("reports/target-extension-v2-dev-01",
                      _preserved_run_declarations(freeze))

    def test_frozen_baseline_hashes_bind_paths_and_bytes(self):
        relative = "config.json"
        freeze = {"frozen_baselines": {
            "original": {"path": "reports/staged-baseline-01", "files": {
                relative: hashlib.sha256((self.ORIGINAL / relative).read_bytes()).hexdigest()}},
            "staged": {"path": "reports/staged-dev-04", "files": {
                relative: hashlib.sha256((self.STAGED / relative).read_bytes()).hexdigest()}},
        }}
        self.assertEqual([], _frozen_baseline_audit(
            freeze, ROOT, self.ORIGINAL, self.STAGED))
        changed = deepcopy(freeze)
        changed["frozen_baselines"]["staged"]["files"][relative] = "0" * 64
        mismatch = _frozen_baseline_audit(
            changed, ROOT, self.ORIGINAL, self.STAGED)
        self.assertEqual(("staged", relative),
                         (mismatch[0]["arm"], mismatch[0]["file"]))

    def test_frozen_prior_artifact_hashes_are_actually_enforced(self):
        relative = "reports/target-extension-v2-smoke-gate-06.json"
        correct = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        self.assertEqual([], _frozen_prior_artifact_audit(
            {"prior_artifacts": {relative: correct}}, ROOT))
        mismatch = _frozen_prior_artifact_audit(
            {"prior_artifacts": {relative: "0" * 64}}, ROOT)
        self.assertEqual(relative, mismatch[0]["file"])

    def test_v3_gate_cannot_skip_frozen_hashes(self):
        result = check_smoke_gate(
            ROOT / "experiments" / "target-extension-freeze-11.json",
            self.GOLD, self.ORIGINAL, self.STAGED,
            ROOT / "reports" / "target-extension-v3-smoke-01",
            hash_mode="skip")
        self.assertIn("freeze.v3_complete_current_hashes",
                      result["failed_checks"])

    def test_v4_gate_cannot_skip_frozen_hashes(self):
        freeze = self.read(ROOT / "experiments" / "target-extension-freeze-11.json")
        freeze.update(
            experiment="target_extended_psi_development_v4_task_routed",
            target_plan_schema=PLAN_SCHEMA_V4,
            provider_mode="task_routed",
            strict_retrieval_attribution=True,
            retrieval_attribution_mode="strict",
            provider="task-routed fixed eligible snapshots, no open-web collection",
        )
        result = _evaluate_gate(
            freeze, None, "not run",
            ROOT / "experiments" / "target-extension-freeze-11.json",
            self.GOLD, self.ORIGINAL, self.STAGED,
            ROOT / "reports" / "missing-v4-smoke", hash_mode="skip")
        self.assertIn("freeze.v4_complete_current_hashes",
                      result["failed_checks"])

    def test_secret_shape_audit_never_echoes_matched_value(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            secret = "-".join(part for part in ("sk", "proj", "A" * 32))
            artifact = root / "artifact.json"
            artifact.write_text(json.dumps({"credential": secret}))
            findings = _secret_shape_audit([artifact], root)
        self.assertEqual([{"file": "artifact.json", "detector": "openai_api_key"}],
                         findings)
        self.assertNotIn(secret, json.dumps(findings))

    @staticmethod
    def v4_audit(case_ids=("p04", "p07", "p08")):
        def blocks(multiplier=1):
            return {
                "coverage_ledger": {"expected_segments": 4 * multiplier,
                    "ledger_entries": 4 * multiplier,
                    "required_dimensions": 3 * multiplier,
                    "covered_dimensions": 3 * multiplier, "breaks": 0},
                "material_probe_ledger": {"stage_calls": 4 * multiplier,
                    "audited_stage_calls": 4 * multiplier,
                    "expected_probe_checks": 8 * multiplier,
                    "probe_checks": 8 * multiplier, "findings": 2 * multiplier,
                    "referenced_findings": 2 * multiplier, "breaks": 0},
                "strict_followups": {"unresolved_probe_slots": multiplier,
                    "task_followups": multiplier, "allowed_stops": 0,
                    "covered_slots": multiplier, "breaks": 0},
                "retrieval_attribution": {"search_rounds": 2 * multiplier,
                    "issued_tasks": 2 * multiplier, "provider_returns": multiplier,
                    "attributed_returns": multiplier, "task_links": multiplier,
                    "valid_task_links": multiplier,
                    "later_probe_owned_tasks": multiplier,
                    "later_probe_owned_hit_tasks": multiplier, "breaks": 0},
                "probe_delta_attribution": {
                    "round_transitions": multiplier,
                    "probe_slots_compared": 2 * multiplier,
                    "semantic_deltas": 0, "basis_drifts": 0,
                    "traced_semantic_deltas": 0,
                    "graph_traced_semantic_deltas": 0,
                    "probe_owned_novel_second_pass_cases": multiplier,
                    "label_changes": 0,
                    "label_changes_with_decisive_delta": 0, "breaks": 0},
                "live_artifact_receipts": {
                    "available": multiplier,
                    "retained_calls": 19 * multiplier,
                    "request_digests_verified": 19 * multiplier,
                    "psi_history_calls": 11 * multiplier,
                    "psi_calls_joined": 11 * multiplier,
                    "verification_history_calls": 6 * multiplier,
                    "verification_calls_joined": 6 * multiplier,
                    "accepted_material_transactions": 2 * multiplier,
                    "direct_attributed_transactions": multiplier,
                    "direct_stage_calls": 3 * multiplier,
                    "direct_stage_calls_with_exact_receipt": 3 * multiplier,
                    "provenance_only_direct_transactions": 0,
                    "revisit_transactions": 0,
                    "revisit_stage_calls_with_empty_receipt": 0,
                    "layer_calls": 4 * multiplier,
                    "layer_calls_with_exact_projection": 4 * multiplier,
                    "layer_receipt_deliveries": multiplier,
                    "second_pass_layer_calls": 2 * multiplier,
                    "second_pass_receipt_deliveries": multiplier,
                    "cross_layer_leaks": 0, "breaks": 0},
            }
        return {"target_plan_probe_coverage": {"v4_audit": {
            "applicable_cases": len(case_ids),
            "totals": blocks(len(case_ids)),
            "cases": [{"id": case_id, **blocks()} for case_id in case_ids],
        }}}

    def test_v4_scorer_ledgers_require_exact_nonempty_coverage(self):
        comparison = self.v4_audit()
        passed, _ = _v4_audit_state(comparison, ["p04", "p07", "p08"], [1, 2])
        self.assertTrue(passed)
        mutations = (
            ("coverage_ledger", "breaks", 1),
            ("material_probe_ledger", "audited_stage_calls", 3),
            ("strict_followups", "covered_slots", 0),
            ("retrieval_attribution", "attributed_returns", 0),
            ("probe_delta_attribution", "breaks", 1),
            ("probe_delta_attribution", "traced_semantic_deltas", 1),
            ("live_artifact_receipts", "request_digests_verified", 18),
            ("live_artifact_receipts", "second_pass_receipt_deliveries", 0),
            ("live_artifact_receipts", "cross_layer_leaks", 1),
        )
        for block, field, value in mutations:
            with self.subTest(block=block, field=field):
                changed = deepcopy(comparison)
                changed["target_plan_probe_coverage"]["v4_audit"]["cases"][0][block][field] = value
                self.assertFalse(_v4_audit_state(
                    changed, ["p04", "p07", "p08"], [1, 2])[0])

    @staticmethod
    def task_routed_label_comparison():
        case_ids = ["p04", "p07", "p08"]
        expected = {"p04": "false", "p07": "false", "p08": "true"}
        cases = [
            {"id": "p04", "reference": "false", "first_round": "true",
             "final": "false", "paired": True, "first_correct": False,
             "final_correct": True},
            {"id": "p07", "reference": "false", "first_round": "false",
             "final": "false", "paired": True, "first_correct": True,
             "final_correct": True},
            {"id": "p08", "reference": "true", "first_round": "true",
             "final": "true", "paired": True, "first_correct": True,
             "final_correct": True},
        ]
        comparison = {
            "arms": {
                "extension": {"completed": 3, "correct": 3},
                "original": {"completed": 3, "correct": 3},
                "staged": {"completed": 3, "correct": 3},
            },
            "label_comparisons": {
                f"extension_vs_{baseline}": {"all_scheduled": {
                    "case_pairs": 3, "case_ids": case_ids, "fixes": 0,
                    "breaks": 0, "candidate_correct": 3}}
                for baseline in ("original", "staged")
            },
            "round1_to_final": {"extension": {
                "label_evaluable_scheduled_cases": 3,
                "paired_completed_cases_with_round1": 3,
                "first_round_correct": 2, "final_correct": 3,
                "fixes": 1, "breaks": 0, "cases": cases,
            }},
        }
        return comparison, case_ids, expected

    def test_task_routed_smoke_allows_a_real_fix_and_reports_each_case(self):
        comparison, case_ids, expected = self.task_routed_label_comparison()
        audit = _smoke_label_audit(comparison, case_ids, expected, "task_routed")
        self.assertTrue(audit["accuracy_ok"])
        self.assertTrue(audit["baselines_ok"])
        self.assertTrue(audit["pairs_ok"])
        self.assertTrue(audit["round_ok"])
        self.assertEqual(case_ids,
                         [item["id"] for item in audit["round_state"]["cases"]])

        fixed = _smoke_label_audit(comparison, case_ids, expected, "fixed_reanalysis")
        self.assertFalse(fixed["round_ok"])
        self.assertTrue(_smoke_label_audit(
            comparison, case_ids, expected, "task_routed", {"p04"})["round_ok"])
        self.assertFalse(_smoke_label_audit(
            comparison, case_ids, expected, "task_routed", set())["round_ok"])

    def test_task_routed_smoke_rejects_a_round1_to_final_break(self):
        comparison, case_ids, expected = self.task_routed_label_comparison()
        round_change = comparison["round1_to_final"]["extension"]
        p07 = round_change["cases"][1]
        p07["final"] = "true"
        p07["final_correct"] = False
        round_change["final_correct"] = 2
        round_change["breaks"] = 1
        audit = _smoke_label_audit(comparison, case_ids, expected, "task_routed")
        self.assertFalse(audit["round_ok"])

    @staticmethod
    def adaptive_loop_fixture(kind):
        origin = {"id": "origin", "action": "search", "probe_id": None}
        first = {"round": 1, "tasks": [origin], "returned": ["b"],
                 "attribution": [{"version_id": "b", "task_ids": ["origin"]}],
                 "feedback": []}
        operations = [
            {"sequence": 1, "round": 1, "action": "search", "tasks": [origin]},
            {"sequence": 2, "round": 1,
             "action": "retrieval_attribution_validated", "version_id": "b",
             "trigger_task_ids": ["origin"]},
            {"sequence": 3, "round": 1, "action": "snapshot_saved",
             "version_id": "b", "eligible": True},
            {"sequence": 4, "round": 1, "action": "decompose_completed",
             "version_id": "b"},
            {"sequence": 5, "round": 1, "action": "verification_started"},
        ]
        retrieval = [first]
        verification_history = [{"round": 1}]
        if kind == "new_hit":
            task = {"id": "probe-task", "stage": "verification",
                    "dimension": "evidence", "action": "fetch",
                    "probe_id": "probe-1"}
            retrieval.append({"round": 2, "tasks": [task], "returned": ["a"],
                "attribution": [{"version_id": "a", "task_ids": ["probe-task"]}],
                "feedback": []})
            operations.extend([
                {"sequence": 6, "round": 2, "action": "search", "tasks": [task]},
                {"sequence": 7, "round": 2,
                 "action": "retrieval_attribution_validated", "version_id": "a",
                 "trigger_task_ids": ["probe-task"]},
                {"sequence": 8, "round": 2, "action": "snapshot_saved",
                 "version_id": "a", "eligible": True},
                {"sequence": 9, "round": 2, "action": "decompose_completed",
                 "version_id": "a"},
                {"sequence": 10, "round": 2,
                 "action": "probe_owned_novel_return_accepted",
                 "version_id": "a", "task_ids": ["probe-task"],
                 "probe_ids": ["probe-1"]},
                {"sequence": 11, "round": 2, "action": "verification_started",
                 "probe_owned_novel_returns": [{
                     "version_id": "a", "task_ids": ["probe-task"],
                     "probe_ids": ["probe-1"]}]},
            ])
            verification_history.append({"round": 2})
            rounds, verification_calls, stop = 2, 2, "round_budget"
        elif kind == "no_hit":
            task = {"id": "probe-task", "stage": "verification",
                    "dimension": "evidence", "action": "fetch",
                    "probe_id": "probe-1"}
            feedback = {"task_id": "probe-task", "probe_id": "probe-1",
                        "status": "corpus_exhausted"}
            retrieval.append({"round": 2, "tasks": [task], "returned": [],
                              "attribution": [], "feedback": [feedback]})
            operations.extend([
                {"sequence": 6, "round": 2, "action": "search", "tasks": [task]},
                {"sequence": 7, "round": 2, "action": "retrieval_feedback",
                 "feedback": feedback},
            ])
            rounds, verification_calls, stop = 2, 1, "provider_exhausted"
        else:
            rounds, verification_calls, stop = 1, 1, "complete"
        usage = {"rounds": rounds, "verification_calls": verification_calls}
        rows = {kind: {"engine_usage": usage}}
        reports = {kind: {"usage": usage, "verification_history": verification_history,
                          "operations": operations, "stop_reason": stop,
                          "retrieval_attribution_mode": "strict"}}
        return rows, reports, {kind: retrieval}

    def test_task_routed_outer_loop_accepts_no_gap_no_hit_and_real_second_pass(self):
        rows, reports, retrievals = {}, {}, {}
        for kind in ("no_gap", "no_hit", "new_hit"):
            case_rows, case_reports, case_retrievals = self.adaptive_loop_fixture(kind)
            rows.update(case_rows)
            reports.update(case_reports)
            retrievals.update(case_retrievals)
        passed, state, rounds, searches = _outer_loop_audit(
            rows, reports, retrievals, ["no_gap", "no_hit", "new_hit"], 2,
            "task_routed")
        self.assertTrue(passed, state)
        self.assertEqual({"no_gap": [1], "no_hit": [1], "new_hit": [1, 2]}, rounds)
        self.assertEqual({"no_gap": 1, "no_hit": 2, "new_hit": 2}, searches)
        self.assertEqual(["no_hit", "new_hit"], state["loop_opportunity_cases"])
        self.assertEqual(["new_hit"], state["second_pass_cases"])
        self.assertEqual(["new_hit"],
                         state["probe_owned_novel_second_pass_cases"])

    def test_task_routed_outer_loop_rejects_fake_second_pass_and_missing_terminal(self):
        rows, reports, retrievals = self.adaptive_loop_fixture("new_hit")
        reports["new_hit"]["operations"] = [item for item in
            reports["new_hit"]["operations"]
            if not (item["round"] == 2 and item["action"] == "decompose_completed")]
        for sequence, item in enumerate(reports["new_hit"]["operations"], 1):
            item["sequence"] = sequence
        self.assertFalse(_outer_loop_audit(
            rows, reports, retrievals, ["new_hit"], 2, "task_routed")[0])

        rows, reports, retrievals = self.adaptive_loop_fixture("no_hit")
        retrievals["no_hit"][1]["feedback"] = []
        self.assertFalse(_outer_loop_audit(
            rows, reports, retrievals, ["no_hit"], 2, "task_routed")[0])

        rows, reports, retrievals = self.adaptive_loop_fixture("no_gap")
        reports["no_gap"]["retrieval_attribution_mode"] = "legacy-compatible"
        self.assertFalse(_outer_loop_audit(
            rows, reports, retrievals, ["no_gap"], 2, "task_routed")[0])

    def test_task_routed_outer_loop_rejects_provenance_only_and_reanalysis_second_pass(self):
        rows, reports, retrievals = self.adaptive_loop_fixture("new_hit")
        retrievals["new_hit"][1]["tasks"][0]["probe_id"] = None
        self.assertFalse(_outer_loop_audit(
            rows, reports, retrievals, ["new_hit"], 2, "task_routed")[0])

        rows, reports, retrievals = self.adaptive_loop_fixture("new_hit")
        start = next(item for item in reports["new_hit"]["operations"]
                     if item["action"] == "verification_started" and
                     item["round"] == 2)
        start["probe_owned_novel_returns"][0]["task_ids"] = ["forged-task"]
        self.assertFalse(_outer_loop_audit(
            rows, reports, retrievals, ["new_hit"], 2, "task_routed")[0])

        rows, reports, retrievals = self.adaptive_loop_fixture("new_hit")
        task = retrievals["new_hit"][1]["tasks"][0]
        task["action"] = "reanalyse"
        retrievals["new_hit"][1]["returned"] = ["b"]
        retrievals["new_hit"][1]["attribution"][0]["version_id"] = "b"
        for operation in reports["new_hit"]["operations"]:
            if operation.get("round") != 2:
                continue
            if operation.get("version_id") == "a":
                operation["version_id"] = "b"
            if operation.get("action") == "snapshot_saved":
                operation["action"] = "duplicate_observed"
            for driver in operation.get("probe_owned_novel_returns", []):
                driver["version_id"] = "b"
        self.assertFalse(_outer_loop_audit(
            rows, reports, retrievals, ["new_hit"], 2, "task_routed")[0])

    def test_v4_config_requires_task_routed_strict_attribution(self):
        freeze = {"experiment": "target_extended_psi_development_v4_task_routed",
                  "target_plan_schema": PLAN_SCHEMA_V4,
                  "model": "gpt-6-astra", "reasoning_effort": "medium",
                  "smoke_cases": ["p04"], "outer_rounds": 2,
                  "max_documents": 24, "max_decomposition_calls": 24,
                  "automatic_transport_retries": 0,
                  "budget_per_case": {"calls": 64, "output_tokens": 64000,
                                      "per_call_output_tokens": 4000, "seconds": 900},
                  "workers": 1, "max_target_structure_repairs": 1,
                  "max_target_extension_repairs": 1, "max_extension_output_repairs": 1,
                  "max_material_structure_repairs_per_return": 1,
                  "max_material_semantic_repairs": 1, "max_judgement_repairs": 1,
                  "max_probe_result_structure_repairs": 1,
                  "inputs_sha256": "a" * 64, "ordering_seed": 20260906,
                  "provider": "task-routed fixed eligible snapshots, no open-web collection",
                  "provider_mode": "task_routed", "strict_retrieval_attribution": True,
                  "retrieval_attribution_mode": "strict",
                  "dataset_status": "previously_seen_development_cases_not_hidden_benchmark"}
        config = {
            "experiment": freeze["experiment"], "target_plan_schema": PLAN_SCHEMA_V4,
            "model": freeze["model"], "reasoning_effort": "medium",
            "case_ids": ["p04"], "scheduled_cases": 1,
            "trace_config": {"max_rounds": 2, "max_documents": 24,
                             "max_decomposition_calls": 24,
                             "experimental_force_rounds": False},
            "automatic_transport_retries": 0,
            "budget_per_case": freeze["budget_per_case"], "workers": 1,
            "target_structure_repairs": 1, "target_extension_repairs": 1,
            "target_extension_output_repairs": 1, "material_stage_repairs": 1,
            "max_inner_repairs": 1, "judgement_repairs": 1,
            "probe_result_structure_repairs": 1, "input_sha256": "a" * 64,
            "ordering_seed": 20260906, "gold_read_during_inference": False,
            "new_response_only": True, "provider": freeze["provider"],
            "provider_mode": "task_routed", "strict_retrieval_attribution": True,
            "retrieval_attribution_mode": "strict",
            "dataset_status": freeze["dataset_status"],
        }
        self.assertTrue(_exact_config(freeze, config)[0])
        config["trace_config"]["experimental_force_rounds"] = True
        self.assertFalse(_exact_config(freeze, config)[0])
        config["trace_config"]["experimental_force_rounds"] = False
        config["retrieval_attribution_mode"] = "legacy"
        self.assertFalse(_exact_config(freeze, config)[0])
        config["retrieval_attribution_mode"] = "strict"
        config["strict_retrieval_attribution"] = False
        self.assertFalse(_exact_config(freeze, config)[0])

    def test_p07_naming_evidence_must_share_one_sentence_or_line(self):
        valid = "On July 7, 2024, GOES-U was renamed GOES-19."
        self.assertTrue(_one_sentence_has_p07_naming_evidence(valid))
        for invalid in (
                "On July 7, 2024. GOES-U was renamed GOES-19.",
                "On July 7, 2024, GOES-U entered orbit. It was renamed GOES-19.",
                "On July 7, 2024, GOES-U was renamed GOES-190.",
                "On July 7, 2024, GOES-U reached orbit."):
            with self.subTest(invalid=invalid):
                self.assertFalse(_one_sentence_has_p07_naming_evidence(invalid))


if __name__ == "__main__":
    unittest.main()
