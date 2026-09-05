"""Synthetic, offline tests for the frozen v3 full-run artifact gate."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from target_extension_full_gate import (
    EXPERIMENT_V3,
    PLAN_SCHEMA_V3,
    PLAN_SCHEMA_V4,
    RUNTIME_EXPERIMENT_FILES,
    V4_EXPERIMENT_BY_PROVIDER,
    V4_PROVIDER_LABELS,
    _evaluate_gate,
    check_full_gate,
)


class FullGateFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.experiments = root / "experiments"
        self.reports = root / "reports"
        self.original = self.reports / "original"
        self.staged = self.reports / "staged"
        self.smoke = self.reports / "smoke"
        self.full = self.reports / "full"
        for path in (self.experiments / "proof_pilot", self.original, self.staged,
                     self.smoke, self.full):
            path.mkdir(parents=True, exist_ok=True)
        self.cases = [f"p{index:02d}" for index in range(1, 9)]
        self.gold_path = self.experiments / "gold.json"
        self.freeze_path = self.experiments / "freeze.json"
        self.input_path = self.experiments / "proof_pilot" / "inputs.json"
        self._write_json(self.gold_path, {"cases": [
            {"id": case_id, "decision": "true"} for case_id in self.cases]})
        self._write_json(self.input_path, {"cases": [
            {"target": {"id": case_id}} for case_id in self.cases]})
        self.input_hash = self._sha(self.input_path)
        self.gold_hash = self._sha(self.gold_path)
        self.runtime_hashes = self._runtime_files()
        self.baselines = self._baseline_files()
        prior = self.reports / "prior.json"
        self._write_json(prior, {"passed": True})
        self.freeze = {
            "status": "frozen_before_live_calls",
            "experiment": EXPERIMENT_V3,
            "target_plan_schema": PLAN_SCHEMA_V3,
            "model": "gpt-6-astra",
            "reasoning_effort": "medium",
            "outer_rounds": 2,
            "workers": 1,
            "automatic_transport_retries": 0,
            "smoke_cases": ["p04", "p07", "p08"],
            "full_cases": self.cases,
            "next_run": "reports/smoke",
            "follow_on_run": "reports/full",
            "max_documents": 24,
            "max_decomposition_calls": 24,
            "ordering_seed": 20260906,
            "provider": "fixed eligible snapshots, no open-web collection",
            "dataset_status": "previously_seen_development_cases_not_hidden_benchmark",
            "budget_per_case": {"calls": 64, "output_tokens": 64000,
                                "per_call_output_tokens": 4000, "seconds": 900},
            "max_target_structure_repairs": 1,
            "max_target_extension_repairs": 1,
            "max_extension_output_repairs": 1,
            "max_material_structure_repairs_per_return": 1,
            "max_material_semantic_repairs": 1,
            "max_judgement_repairs": 1,
            "max_probe_result_structure_repairs": 1,
            "inputs_sha256": self.input_hash,
            "provisional_reference_sha256": self.gold_hash,
            "files": self.runtime_hashes,
            "frozen_baselines": self.baselines,
            "prior_artifacts": {"reports/prior.json": self._sha(prior)},
            "preserved_runs": {},
        }
        self._full_artifacts()
        self._write_json(self.freeze_path, self.freeze)
        self.comparison = self._comparison()
        self.smoke_result = {
            "schema_version": "target-extension-smoke-gate-v1",
            "passed": True,
            "checks_run": 40,
            "checks_passed": 40,
            "failed_checks": [],
        }

    @staticmethod
    def _write_json(path: Path, value) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False) + "\n")

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _canonical_sha(value) -> str:
        payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    def _runtime_files(self):
        hashes = {}
        for relative in sorted(RUNTIME_EXPERIMENT_FILES):
            content = ("OpenAI(max_retries=0)\n" if relative == "experiments/model_io.py"
                       else f"# frozen {relative}\n")
            current = self.root / relative
            captured = self.full / "executed-code" / relative
            current.parent.mkdir(parents=True, exist_ok=True)
            captured.parent.mkdir(parents=True, exist_ok=True)
            current.write_text(content)
            captured.write_text(content)
            hashes[relative] = self._sha(current)
        return hashes

    def _baseline_files(self):
        result = {}
        required = {"config.json", "inputs.json", "results.json"}
        required |= {f"{case_id}-{suffix}.json" for case_id in self.cases
                     for suffix in ("calls", "report", "result")}
        for arm, directory in (("original", self.original), ("staged", self.staged)):
            arm_required = set(required)
            if arm == "staged":
                arm_required |= {f"{case_id}-{suffix}.json" for case_id in self.cases
                                 for suffix in ("psi-history", "verification-history")}
            files = {}
            for name in sorted(arm_required):
                self._write_json(directory / name, {"arm": arm, "file": name})
                files[name] = self._sha(directory / name)
            result[arm] = {"path": directory.relative_to(self.root).as_posix(),
                           "files": files}
        return result

    def _config(self):
        return {
            "experiment": EXPERIMENT_V3,
            "target_plan_schema": PLAN_SCHEMA_V3,
            "model": "gpt-6-astra",
            "reasoning_effort": "medium",
            "case_ids": self.cases,
            "scheduled_cases": 8,
            "trace_config": {"max_rounds": 2, "max_documents": 24,
                             "max_decomposition_calls": 24,
                             "experimental_force_rounds": True},
            "automatic_transport_retries": 0,
            "budget_per_case": self.freeze["budget_per_case"],
            "workers": 1,
            "target_extension": True,
            "target_structure_repairs": 1,
            "target_extension_repairs": 1,
            "target_extension_output_repairs": 1,
            "material_stage_repairs": 1,
            "max_inner_repairs": 1,
            "judgement_repairs": 1,
            "probe_result_structure_repairs": 1,
            "input_sha256": self.input_hash,
            "ordering_seed": 20260906,
            "gold_read_during_inference": False,
            "new_response_only": True,
            "provider": self.freeze["provider"],
            "dataset_status": self.freeze["dataset_status"],
            "source_sha256": self.runtime_hashes,
        }

    def _full_artifacts(self):
        config = self._config()
        self._write_json(self.full / "config.json", config)
        self._write_json(self.full / "inputs.json", {"cases": [
            {"target": {"id": case_id}} for case_id in self.cases]})
        rows = []
        for case_id in self.cases:
            checkpoints = []
            checkpoint_hashes = []
            for round_number in (1, 2):
                checkpoint = {"round": round_number, "decision": "true"}
                self._write_json(
                    self.full / f"{case_id}-checkpoint-{round_number}.json", checkpoint)
                checkpoints.append(checkpoint)
                checkpoint_hashes.append({"round": round_number,
                                          "sha256": self._canonical_sha(checkpoint)})
            usage = {"model_calls": 1, "input_tokens": 10,
                     "output_tokens": 2, "seconds": 1.0}
            row = {"id": case_id, "status": "completed", "prediction": "true",
                   "checkpoints": checkpoints, "checkpoint_hashes": checkpoint_hashes,
                   "new_api_attempts": 1, "returned_responses": 1, "usage": usage}
            rows.append(row)
            self._write_json(self.full / f"{case_id}-result.json", row)
            self._write_json(self.full / f"{case_id}-calls.json", [{
                "sequence": 1, "max_completion_tokens": 4000,
                "response_id": f"response-{case_id}", "actual_model": "gpt-6-astra",
                "error_type": None, "error_code": None, "has_refusal": False,
                "finish_reason": "stop", "reasoning_tokens": 1,
                "usage": {"input_tokens": 10, "output_tokens": 2}}])
            self._write_json(self.full / f"{case_id}-report.json",
                             {"assessment_valid": True, "errors": []})
            self._write_json(self.full / f"{case_id}-target-plan.json",
                             {"schema_version": PLAN_SCHEMA_V3})
            self._write_json(self.full / f"{case_id}-verification-history.json", [])
        self._write_json(self.full / "results.json", rows)
        self._write_json(self.full / "status.json", {
            "status": "completed", "scheduled_cases": 8, "completed": 8,
            "new_api_attempts": 8, "returned_responses": 8,
            "recorded_usage": {"model_calls": 8, "input_tokens": 80,
                               "output_tokens": 16}})

    def _comparison(self):
        usage = {"model_calls": 8, "input_tokens": 80, "output_tokens": 16,
                 "reasoning_tokens": 8, "visible_output_tokens": 8,
                 "service_seconds": 8.0, "reasoning_reporting_complete": True}
        extension = {
            "scheduled_cases": 8, "completed": 8, "execution_errors": 0,
            "label_evaluable_cases": 8, "correct": 8,
            "origins": {"evaluable_cases": 3, "matched_predictions": 3,
                        "predicted_items": 3, "precision": 1.0, "recall": 1.0,
                        "recall_denominator": 3},
            "edges": {"evaluable_cases": 4, "matched_predictions": 4,
                      "predicted_items": 4, "precision": 1.0, "recall": 1.0,
                      "gold_edges": 4, "recall_denominator": 4},
            "usage_total": usage,
        }
        baselines = {arm: {"scheduled_cases": 8, "completed": 8,
                           "execution_errors": 0}
                     for arm in ("original", "staged")}
        pairs = {f"extension_vs_{arm}": {"all_scheduled": {
            "case_pairs": 8, "case_ids": self.cases, "fixes": 1,
            "breaks": 0, "candidate_correct": 8}}
            for arm in ("original", "staged")}
        plan_cases = [{"id": case_id, "status": "completed", "plan_valid": True,
                       "plan_contract": PLAN_SCHEMA_V3,
                       "terminal_origin_audit": {"terminal_roots": True}}
                      for case_id in self.cases]
        result_cases = [{"id": case_id, "status": "completed", "available": True,
                         "accepted_judgement_cycles": 2,
                         "audited_judgement_cycles": 2,
                         "expected_result_slots": 4, "result_slots": 4,
                         "result_slot_coverage": 1.0, "results_with_basis": 4,
                         "grounded_conclusive_rate": 1.0}
                        for case_id in self.cases]
        artifacts = {
            arm: {name: self.baselines[arm]["files"][name]
                  for name in ("config.json", "inputs.json", "results.json")}
            for arm in ("original", "staged")}
        artifacts["extension"] = {name: self._sha(self.full / name)
                                  for name in ("config.json", "inputs.json", "results.json")}
        return {
            "selected_case_ids": self.cases,
            "scheduled_cases_per_arm": 8,
            "model": "gpt-6-astra",
            "reasoning_effort": "medium",
            "reference_sha256": self.gold_hash,
            "artifacts_sha256": artifacts,
            "arms": {**baselines, "extension": extension},
            "label_comparisons": pairs,
            "round1_to_final": {"extension": {
                "label_evaluable_scheduled_cases": 8,
                "paired_completed_cases_with_round1": 8,
                "final_correct": 8, "fixes": 1, "breaks": 0}},
            "target_plan_probe_coverage": {
                "totals": {"plans_present": 8, "plans_valid": 8,
                           "required_probe_coverage": 1.0,
                           "projection_coverage": 1.0,
                           "executed_stage_projection_coverage": 1.0},
                "cases": plan_cases},
            "target_probe_result_audit": {
                "totals": {"cases_with_results": 8, "scheduled_cases": 8,
                           "accepted_judgement_cycles": 16,
                           "audited_judgement_cycles": 16,
                           "expected_result_slots": 32, "result_slots": 32,
                           "results_with_basis": 32, "result_slot_coverage": 1.0,
                           "grounded_conclusive_rate": 1.0,
                           "by_stage": {"evidence": {"result_slot_coverage": 1.0},
                                        "world": {"result_slot_coverage": 1.0}}},
                "cases": result_cases},
        }

    def enable_v4(self, provider_mode="task_routed"):
        strict = provider_mode == "task_routed"
        self.freeze.update(
            experiment=V4_EXPERIMENT_BY_PROVIDER[provider_mode],
            target_plan_schema=PLAN_SCHEMA_V4,
            provider_mode=provider_mode,
            strict_retrieval_attribution=strict,
            retrieval_attribution_mode=("strict" if strict else "legacy"),
            provider=V4_PROVIDER_LABELS[provider_mode],
        )
        config_path = self.full / "config.json"
        config = json.loads(config_path.read_text())
        config.update(
            experiment=self.freeze["experiment"],
            target_plan_schema=PLAN_SCHEMA_V4,
            provider_mode=provider_mode,
            strict_retrieval_attribution=strict,
            retrieval_attribution_mode=("strict" if strict else "legacy"),
            provider=self.freeze["provider"],
        )
        config["trace_config"]["experimental_force_rounds"] = not strict
        self._write_json(config_path, config)
        rows = json.loads((self.full / "results.json").read_text())
        rows_by_id = {item["id"]: item for item in rows}
        for case_id in self.cases:
            self._write_json(self.full / f"{case_id}-psi-history.json", [])
            self._write_json(self.full / f"{case_id}-target-plan.json",
                             {"schema_version": PLAN_SCHEMA_V4})
            usage = {"rounds": 2, "verification_calls": 2}
            history = [{"round": 1}, {"round": 2}]
            if strict:
                origin = {"id": f"origin:{case_id}", "action": "search",
                          "probe_id": None}
                probe_task = {"id": f"probe:{case_id}", "action": "fetch",
                              "probe_id": f"frozen-probe:{case_id}",
                              "stage": "verification", "dimension": "evidence"}
                retrieval = [
                    {"round": 1, "tasks": [origin], "returned": ["m1"],
                     "attribution": [{"version_id": "m1",
                                      "task_ids": [origin["id"]]}],
                     "feedback": []},
                    {"round": 2, "tasks": [probe_task], "returned": ["m2"],
                     "attribution": [{"version_id": "m2",
                                      "task_ids": [probe_task["id"]]}],
                     "feedback": []},
                ]
                operations = [
                    {"sequence": 1, "round": 1, "action": "search",
                     "tasks": [origin]},
                    {"sequence": 2, "round": 1,
                     "action": "retrieval_attribution_validated", "version_id": "m1",
                     "trigger_task_ids": [origin["id"]]},
                    {"sequence": 3, "round": 1, "action": "snapshot_saved",
                     "version_id": "m1", "eligible": True},
                    {"sequence": 4, "round": 1, "action": "decompose_completed",
                     "version_id": "m1"},
                    {"sequence": 5, "round": 1, "action": "verification_started"},
                    {"sequence": 6, "round": 2, "action": "search",
                     "tasks": [probe_task]},
                    {"sequence": 7, "round": 2,
                     "action": "retrieval_attribution_validated", "version_id": "m2",
                     "trigger_task_ids": [probe_task["id"]]},
                    {"sequence": 8, "round": 2, "action": "snapshot_saved",
                     "version_id": "m2", "eligible": True},
                    {"sequence": 9, "round": 2, "action": "decompose_completed",
                     "version_id": "m2"},
                    {"sequence": 10, "round": 2,
                     "action": "probe_owned_novel_return_accepted",
                     "version_id": "m2", "task_ids": [probe_task["id"]],
                     "probe_ids": [probe_task["probe_id"]]},
                    {"sequence": 11, "round": 2, "action": "verification_started",
                     "probe_owned_novel_returns": [{
                         "version_id": "m2", "task_ids": [probe_task["id"]],
                         "probe_ids": [probe_task["probe_id"]]}]},
                ]
                report_mode = "strict"
            else:
                retrieval = [{"round": round_number} for round_number in (1, 2)]
                operations = []
                report_mode = "legacy-compatible"
            report = {"assessment_valid": True, "errors": [], "usage": usage,
                      "verification_history": history, "operations": operations,
                      "stop_reason": "round_budget",
                      "retrieval_attribution_mode": report_mode}
            self._write_json(self.full / f"{case_id}-retrieval.json", retrieval)
            self._write_json(self.full / f"{case_id}-report.json", report)
            rows_by_id[case_id]["engine_usage"] = usage
            self._write_json(self.full / f"{case_id}-result.json",
                             rows_by_id[case_id])
        self._write_json(self.full / "results.json",
                         [rows_by_id[case_id] for case_id in self.cases])
        self._write_json(self.freeze_path, self.freeze)
        comparison = self._comparison()
        comparison["artifacts_sha256"]["extension"]["config.json"] = self._sha(
            config_path)
        plan_audit = comparison["target_plan_probe_coverage"]
        for item in plan_audit["cases"]:
            item["plan_contract"] = PLAN_SCHEMA_V4
        round_rows = []
        for index, case_id in enumerate(self.cases):
            first_correct = not strict or index != 0
            round_rows.append({"id": case_id, "reference": "true",
                "first_round": "true" if first_correct else "false",
                "final": "true", "paired": True,
                "first_correct": first_correct, "final_correct": True})
        comparison["round1_to_final"]["extension"].update(
            first_round_correct=sum(item["first_correct"] for item in round_rows),
            fixes=sum(not item["first_correct"] for item in round_rows),
            cases=round_rows,
        )

        def blocks(multiplier=1, label_changes=0):
            returns = multiplier
            attributed = multiplier if strict else 0
            task_links = multiplier if strict else 0
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
                    "issued_tasks": 2 * multiplier, "provider_returns": returns,
                    "attributed_returns": attributed, "task_links": task_links,
                    "valid_task_links": task_links,
                    "later_probe_owned_tasks": multiplier if strict else 0,
                    "later_probe_owned_hit_tasks": multiplier if strict else 0,
                    "breaks": 0},
                "probe_delta_attribution": {
                    "round_transitions": multiplier if strict else 0,
                    "probe_slots_compared": 2 * multiplier if strict else 0,
                    "semantic_deltas": label_changes if strict else 0,
                    "basis_drifts": label_changes if strict else 0,
                    "traced_semantic_deltas": label_changes if strict else 0,
                    "graph_traced_semantic_deltas": 0,
                    "probe_owned_novel_second_pass_cases":
                        multiplier if strict else 0,
                    "label_changes": label_changes if strict else 0,
                    "label_changes_with_decisive_delta":
                        label_changes if strict else 0,
                    "breaks": 0},
            }

        plan_audit["v4_audit"] = {
            "applicable_cases": len(self.cases),
            "totals": blocks(len(self.cases), label_changes=int(strict)),
            "cases": [{"id": case_id,
                       **blocks(label_changes=int(strict and index == 0))}
                      for index, case_id in enumerate(self.cases)],
        }
        return comparison

    def make_adaptive_no_gap_and_no_hit(self, comparison):
        """Turn two routed cases into legitimate one-verification terminations."""
        rows = {item["id"]: item for item in
                json.loads((self.full / "results.json").read_text())}
        audit_cases = {item["id"]: item for item in comparison[
            "target_plan_probe_coverage"]["v4_audit"]["cases"]}
        result_cases = {item["id"]: item for item in comparison[
            "target_probe_result_audit"]["cases"]}
        for case_id, stop_kind in (("p01", "no_gap"), ("p02", "no_hit")):
            report_path = self.full / f"{case_id}-report.json"
            report = json.loads(report_path.read_text())
            retrieval_path = self.full / f"{case_id}-retrieval.json"
            retrieval = json.loads(retrieval_path.read_text())
            if stop_kind == "no_gap":
                report["usage"] = {"rounds": 1, "verification_calls": 1}
                report["verification_history"] = report["verification_history"][:1]
                report["operations"] = [item for item in report["operations"]
                                        if item["round"] == 1]
                report["stop_reason"] = "complete"
                retrieval = retrieval[:1]
                audit_cases[case_id]["retrieval_attribution"]["search_rounds"] = 1
                comparison["target_plan_probe_coverage"]["v4_audit"]["totals"][
                    "retrieval_attribution"]["search_rounds"] -= 1
                for field in ("later_probe_owned_tasks",
                              "later_probe_owned_hit_tasks"):
                    audit_cases[case_id]["retrieval_attribution"][field] = 0
                    comparison["target_plan_probe_coverage"]["v4_audit"]["totals"][
                        "retrieval_attribution"][field] -= 1
                delta = audit_cases[case_id]["probe_delta_attribution"]
                totals_delta = comparison["target_plan_probe_coverage"][
                    "v4_audit"]["totals"]["probe_delta_attribution"]
                for field, value in list(delta.items()):
                    totals_delta[field] -= value
                    delta[field] = 0
            else:
                task = retrieval[1]["tasks"][0]
                feedback = {"task_id": task["id"], "probe_id": task["probe_id"],
                            "status": "corpus_exhausted"}
                retrieval[1].update(returned=[], attribution=[], feedback=[feedback])
                report["usage"] = {"rounds": 2, "verification_calls": 1}
                report["verification_history"] = report["verification_history"][:1]
                report["operations"] = report["operations"][:5] + [
                    {"sequence": 6, "round": 2, "action": "search", "tasks": [task]},
                    {"sequence": 7, "round": 2, "action": "retrieval_feedback",
                     "feedback": feedback},
                ]
                report["stop_reason"] = "provider_exhausted"
                audit_cases[case_id]["retrieval_attribution"][
                    "later_probe_owned_hit_tasks"] = 0
                comparison["target_plan_probe_coverage"]["v4_audit"]["totals"][
                    "retrieval_attribution"]["later_probe_owned_hit_tasks"] -= 1
                delta = audit_cases[case_id]["probe_delta_attribution"]
                totals_delta = comparison["target_plan_probe_coverage"][
                    "v4_audit"]["totals"]["probe_delta_attribution"]
                for field in ("round_transitions", "probe_slots_compared",
                              "probe_owned_novel_second_pass_cases"):
                    totals_delta[field] -= delta[field]
                    delta[field] = 0
            self._write_json(report_path, report)
            self._write_json(retrieval_path, retrieval)
            rows[case_id]["engine_usage"] = report["usage"]
            rows[case_id]["checkpoints"] = rows[case_id]["checkpoints"][:1]
            rows[case_id]["checkpoint_hashes"] = rows[case_id]["checkpoint_hashes"][:1]
            (self.full / f"{case_id}-checkpoint-2.json").unlink()
            self._write_json(self.full / f"{case_id}-result.json", rows[case_id])
            result_cases[case_id].update(
                accepted_judgement_cycles=1, audited_judgement_cycles=1,
                expected_result_slots=2, result_slots=2, results_with_basis=2)

        self._write_json(self.full / "results.json",
                         [rows[case_id] for case_id in self.cases])
        comparison["artifacts_sha256"]["extension"]["results.json"] = self._sha(
            self.full / "results.json")
        result_totals = comparison["target_probe_result_audit"]["totals"]
        result_totals.update(
            accepted_judgement_cycles=14, audited_judgement_cycles=14,
            expected_result_slots=28, result_slots=28, results_with_basis=28)
        round_rows = comparison["round1_to_final"]["extension"]["cases"]
        for item in round_rows:
            if item["id"] in {"p01", "p02"}:
                item.update(first_round="true", first_correct=True)
        comparison["round1_to_final"]["extension"].update(
            first_round_correct=8, fixes=0)
        return comparison

    def evaluate(self, comparison=None):
        return _evaluate_gate(
            self.freeze, comparison or self.comparison, None,
            self.freeze_path, self.gold_path, self.original, self.staged, self.full,
            self.smoke_result, None)


class TargetExtensionFullGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = FullGateFixture(Path(self.temporary.name))

    def tearDown(self):
        self.temporary.cleanup()

    def test_complete_synthetic_full_run_passes_and_allows_fixes(self):
        result = self.fixture.evaluate()
        self.assertTrue(result["passed"], result["failed_checks"])
        self.assertEqual(1, self.fixture.comparison[
            "round1_to_final"]["extension"]["fixes"])
        self.assertEqual(8, result["reported_usage"]["model_calls"])
        self.assertIn("resources.reported_not_thresholded",
                      [item["id"] for item in result["checks"]])

    def test_v4_task_routed_full_gate_requires_all_four_independent_ledgers(self):
        comparison = self.fixture.enable_v4("task_routed")
        result = self.fixture.evaluate(comparison)
        self.assertTrue(result["passed"], result["failed_checks"])
        self.assertIn("probe_audit.eight_valid_v4_plans",
                      [item["id"] for item in result["checks"]])
        self.assertIn("probe_audit.v4_end_to_end_ledgers",
                      [item["id"] for item in result["checks"]])

        for block, field, value in (
                ("coverage_ledger", "ledger_entries", 31),
                ("material_probe_ledger", "probe_checks", 63),
                ("strict_followups", "covered_slots", 7),
                ("retrieval_attribution", "valid_task_links", 7)):
            with self.subTest(block=block, field=field):
                changed = deepcopy(comparison)
                changed["target_plan_probe_coverage"]["v4_audit"][
                    "totals"][block][field] = value
                failed = self.fixture.evaluate(changed)
                self.assertIn("probe_audit.v4_end_to_end_ledgers",
                              failed["failed_checks"])

    def test_v4_task_routed_full_gate_accepts_adaptive_no_gap_and_no_hit(self):
        comparison = self.fixture.enable_v4("task_routed")
        comparison = self.fixture.make_adaptive_no_gap_and_no_hit(comparison)
        result = self.fixture.evaluate(comparison)
        self.assertTrue(result["passed"], result["failed_checks"])
        loop = next(item for item in result["checks"]
                    if item["id"] == "execution.adaptive_task_routed_outer_loop")
        self.assertEqual(["p02", "p03", "p04", "p05", "p06", "p07", "p08"],
                         loop["actual"]["loop_opportunity_cases"])
        self.assertEqual(["p03", "p04", "p05", "p06", "p07", "p08"],
                         loop["actual"]["second_pass_cases"])
        cycles = next(item for item in result["checks"]
                      if item["id"] == "probe_audit.adaptive_cycles")
        self.assertEqual(14, cycles["actual"]["accepted_judgement_cycles"])

    def test_v4_fixed_reanalysis_is_distinct_and_cannot_claim_task_links(self):
        comparison = self.fixture.enable_v4("fixed_reanalysis")
        result = self.fixture.evaluate(comparison)
        self.assertTrue(result["passed"], result["failed_checks"])
        changed = deepcopy(comparison)
        retrieval = changed["target_plan_probe_coverage"]["v4_audit"][
            "cases"][0]["retrieval_attribution"]
        retrieval["attributed_returns"] = retrieval["provider_returns"]
        retrieval["task_links"] = retrieval["valid_task_links"] = 1
        failed = self.fixture.evaluate(changed)
        self.assertIn("probe_audit.v4_end_to_end_ledgers", failed["failed_checks"])

    def test_v4_schema_cannot_hide_under_the_v3_experiment_name(self):
        comparison = self.fixture.enable_v4("task_routed")
        self.fixture.freeze["experiment"] = EXPERIMENT_V3
        result = self.fixture.evaluate(comparison)
        self.assertIn("freeze.full_contract", result["failed_checks"])

    def test_one_baseline_or_round_break_fails(self):
        comparison = deepcopy(self.fixture.comparison)
        comparison["label_comparisons"]["extension_vs_staged"]["all_scheduled"][
            "breaks"] = 1
        comparison["round1_to_final"]["extension"]["breaks"] = 1
        result = self.fixture.evaluate(comparison)
        self.assertFalse(result["passed"])
        self.assertIn("labels.zero_baseline_breaks", result["failed_checks"])
        self.assertIn("labels.zero_round_breaks", result["failed_checks"])

    def test_aggregate_cycles_cannot_hide_one_bad_case(self):
        comparison = deepcopy(self.fixture.comparison)
        comparison["target_probe_result_audit"]["cases"][0][
            "audited_judgement_cycles"] = 1
        result = self.fixture.evaluate(comparison)
        self.assertFalse(result["passed"])
        self.assertIn("probe_audit.per_case_cycles_and_coverage",
                      result["failed_checks"])
        self.assertNotIn("probe_audit.sixteen_cycles", result["failed_checks"])

    def test_imperfect_origin_or_edge_metric_fails(self):
        comparison = deepcopy(self.fixture.comparison)
        comparison["arms"]["extension"]["origins"]["precision"] = 0.75
        comparison["arms"]["extension"]["edges"]["recall"] = 0.75
        result = self.fixture.evaluate(comparison)
        self.assertFalse(result["passed"])
        self.assertIn("provenance.origins", result["failed_checks"])
        self.assertIn("provenance.direct_edges", result["failed_checks"])

    def test_baseline_call_ledgers_must_be_in_frozen_manifest(self):
        del self.fixture.freeze["frozen_baselines"]["staged"]["files"]["p01-calls.json"]
        result = self.fixture.evaluate()
        self.assertFalse(result["passed"])
        self.assertIn("freeze.baseline_hashes", result["failed_checks"])

    def test_missing_captured_runtime_source_fails_closed(self):
        path = self.fixture.full / "executed-code" / "experiments" / "staged_run.py"
        path.unlink()
        result = self.fixture.evaluate()
        self.assertFalse(result["passed"])
        self.assertIn("execution.captured_runtime_hashes", result["failed_checks"])

    def test_public_gate_recomputes_comparison_and_smoke(self):
        with patch("target_extension_full_gate.comparison_score",
                   return_value=self.fixture.comparison) as compare, patch(
                       "target_extension_full_gate.check_smoke_gate",
                       return_value=self.fixture.smoke_result) as smoke:
            result = check_full_gate(
                self.fixture.freeze_path, self.fixture.gold_path,
                self.fixture.original, self.fixture.staged, self.fixture.full)
        self.assertTrue(result["passed"], result["failed_checks"])
        compare.assert_called_once()
        smoke.assert_called_once()

    def test_duplicate_json_key_returns_failed_json_gate(self):
        (self.fixture.full / "status.json").write_text(
            '{"status":"completed","status":"completed"}\n')
        with patch("target_extension_full_gate.comparison_score",
                   return_value=self.fixture.comparison), patch(
                       "target_extension_full_gate.check_smoke_gate",
                       return_value=self.fixture.smoke_result):
            result = check_full_gate(
                self.fixture.freeze_path, self.fixture.gold_path,
                self.fixture.original, self.fixture.staged, self.fixture.full)
        self.assertFalse(result["passed"])
        self.assertIn("execution.strict_json", result["failed_checks"])

    def test_missing_artifact_fails_without_traceback(self):
        (self.fixture.full / "p08-report.json").unlink()
        with patch("target_extension_full_gate.comparison_score",
                   side_effect=ValueError("missing report")), patch(
                       "target_extension_full_gate.check_smoke_gate",
                       return_value=self.fixture.smoke_result):
            result = check_full_gate(
                self.fixture.freeze_path, self.fixture.gold_path,
                self.fixture.original, self.fixture.staged, self.fixture.full)
        self.assertFalse(result["passed"])
        self.assertIn("execution.required_artifacts", result["failed_checks"])
        self.assertIn("comparison.independent_audit", result["failed_checks"])


if __name__ == "__main__":
    unittest.main()
