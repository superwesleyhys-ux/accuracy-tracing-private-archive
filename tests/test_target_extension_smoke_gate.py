"""Artifact-only tests for generic frozen target-extension smoke gates."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from target_extension_smoke_gate import (PLAN_SCHEMA_V3, _Checks,
                                         _canonical_identity_question,
                                         _checkpoint_audit, _exact_config,
                                         _execution_control_declarations,
                                         _frozen_baseline_audit,
                                         _one_sentence_has_p07_naming_evidence,
                                         _p02_checks, _p04_checks, _p07_checks, _p08_checks,
                                         _preserved_run_declarations, check_smoke_gate)


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
        self.assertEqual((34, 34, []), (result["checks_run"],
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
        self.assertEqual(5, len(v3))
        self.assertTrue(all(item["passed"] for item in v3), v3)

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
