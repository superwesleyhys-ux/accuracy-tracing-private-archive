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

from target_extension_compare_score import (_audit_plan, _canonical_digest,
                                             _plan_coverage, score)


class TargetExtensionCompareTests(unittest.TestCase):
    GOLD = ROOT / "experiments" / "proof_pilot" / "gold.json"
    ORIGINAL = ROOT / "reports" / "staged-baseline-01"
    STAGED = ROOT / "reports" / "staged-dev-04"
    EXTENSION_SMOKE = ROOT / "reports" / "target-extension-smoke-01"

    @staticmethod
    def _plan():
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
             ["atoms", "evidence"]),
            ("quantity", "quantity_unit", ["c1:dimension:quantity"], ["atoms", "evidence"]),
            ("lineage", "source_lineage", [], ["lineage", "world"]),
        )
        probes = [{"id": "c1:probe:" + name, "claim_id": claim["id"], "kind": kind,
                   "dimension_ids": dimension_ids,
                   "question": name + "?", "decision_impact": name + " impact",
                   "routes": routes, "gate": "always"}
                  for name, kind, dimension_ids, routes in probe_specs]
        base = {"target_signature": _canonical_digest(target), "logic": "single",
                "claims": [claim], "probes": probes, "notes": ""}
        checksum = _canonical_digest(base)
        projections = {}
        for stage in ("atoms", "lineage", "critic", "evidence", "world"):
            selected = probes if stage == "critic" else [p for p in probes if stage in p["routes"]]
            projections[stage] = {"target_signature": base["target_signature"],
                "plan_sha256": checksum, "stage": stage,
                "claims": [claim] if selected else [], "probes": selected}
        return target, {**base, "sha256": checksum, "projections": projections}

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
                                  result["routed_stage_slots_delivered_to_executed_stage"]))
        self.assertEqual(3, result["unique_probes_reviewed_by_an_accepted_judgement_critic"])

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
                                     totals["routed_stage_delivery_coverage"]))
        self.assertEqual(3, totals["projected_probes_critic"])

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
            self.assertEqual("target-extension-compare-score-v1", result["schema_version"])


if __name__ == "__main__":
    unittest.main()
