"""Artifact-only pairing tests; no inference client or credentials are used."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from newsverify.decisions import present_decision, round_decisions
from staged_compare_score import DATASET, EXPERIMENTS, PROVIDER, score


class PairedScoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.original, self.staged = self.root / "original", self.root / "staged"
        self.original.mkdir()
        self.staged.mkdir()
        self.gold = self.root / "gold.json"
        self.data = {"schema_version": 1, "cases": []}
        references = []
        for index in range(8):
            identifier = f"c{index}"
            label = ("true", "false", "true", "unverifiable")[index % 4]
            target = {"id": identifier, "text": f"claim {index}", "as_of": "2026-09-05T00:00:00Z",
                      "assessment_mode": "evidence", "source_version_id": "root", "evidence_scope": ["root"]}
            materials = [{"version_id": v, "url": f"https://example.com/{identifier}/{v}",
                          "content": f"Retained {v} text", "retrieved_at": target["as_of"], "published_at": None,
                          "available_at": target["as_of"], "availability_basis": "Fixed capture", "issuer": "Test"}
                         for v in ("root", "other")]
            self.data["cases"].append({"target": target, "materials": materials, "seed_ids": ["other"],
                                        "full_evidence_control": True})
            references.append({"id": identifier, "event_id": f"event-{index}", "assessment_mode": "evidence",
                               "decision": label, "origin_evaluable": index < 3, "origins": ["root"] if index < 3 else [],
                               "edges_evaluable": index < 4, "edges": [["other", "root", "cites"]] if index < 4 else []})
        self.write(self.gold, {"cases": references, "held_out": False, "independent_human_labeling": False})
        for arm, run in (("original", self.original), ("staged", self.staged)):
            config = {"experiment": EXPERIMENTS[arm], "case_ids": [f"c{i}" for i in range(8)], "scheduled_cases": 8,
                      "model": "gpt-6-astra", "reasoning_effort": "medium", "input_sha256": "a" * 64,
                      "budget_per_case": {"calls": 64, "output_tokens": 64000, "per_call_output_tokens": 4000, "seconds": 900},
                      "gold_read_during_inference": False, "new_response_only": True, "automatic_transport_retries": 0,
                      "provider": PROVIDER, "dataset_status": DATASET, "original_depth": 1,
                      "trace_config": {"max_rounds": 2, "max_documents": 24, "max_decomposition_calls": 24,
                                       "experimental_force_rounds": True}}
            self.write(run / "config.json", config)
            self.write(run / "inputs.json", self.data)
            rows = []
            for index, case in enumerate(self.data["cases"]):
                identifier = f"c{index}"
                failed = (arm == "original" and index == 1) or (arm == "staged" and index == 0)
                decision = "false" if arm == "original" and index == 2 else references[index]["decision"]
                row = {"id": identifier, "assessment_mode": "evidence", "status": "error" if failed else "completed",
                       "prediction": None, "usage": {"model_calls": 1, "input_tokens": 10, "output_tokens": 20, "seconds": 1},
                       "new_api_attempts": 1, "returned_responses": 1}
                call = {"sequence": 1, "response_id": f"{arm}-{identifier}", "actual_model": config["model"],
                        "max_completion_tokens": 4000, "finish_reason": "stop", "usage": {"input_tokens": 10, "output_tokens": 20}}
                if failed:
                    row["error_type"] = "ValueError"
                    call.update(error_type="ValueError", error_code="invalid_json")
                if arm == "original":
                    row["arm"] = arm
                    if not failed:
                        origins = ["other"] if index == 2 else ["root"]
                        row["prediction"] = {"decision": decision, "origins": origins, "edges": [],
                            "provenance_evaluable": True, "origin_evaluable": True, "edges_evaluable": False,
                            "edge_export_limit": "No native source-to-source edge representation"}
                        row["extractor_answer_text"] = f"Final conclusion: {decision}."
                        row["extraction"] = {"decision": decision, "extractable": True, "basis_quote": decision}
                        report = {"direct_response": row["extractor_answer_text"], "sources": [
                            {"url": f"https://example.com/{identifier}/{v}", "is_original": True} for v in origins]}
                else:
                    row["checkpoints"] = []
                    if not failed:
                        verdict = {"true": "supported", "false": "contradicted", "unverifiable": "unresolved"}[decision]
                        assessment = {"evidence": {"decision": verdict}, "world": {"decision": "unresolved"}}
                        report = {"target": case["target"], "config": config["trace_config"], "assessment_valid": True,
                                  "errors": [], "decision_status": verdict, "assessments": assessment,
                                  "provenance_status": "original_material_located", "stop_reason": "round_limit",
                                  "usage": {"verification_calls": 2, "rounds": 2}, "origins": [{"version_id": "root"}],
                                  "relations": [{"from_version": "other", "to_version": "root", "kind": "cites", "status": "direct"}],
                                  "operations": [{"action": "verification_completed", "round": r,
                                      "decision": "unresolved" if r == 1 else verdict,
                                      "assessments": assessment, "provenance_status": "original_material_located"} for r in (1, 2)]}
                        row.update(prediction=decision, native_prediction=present_decision(report),
                                   checkpoints=round_decisions(report), engine_usage=report["usage"])
                if not failed:
                    self.write(run / f"{identifier}-report.json", report)
                self.write(run / f"{identifier}-calls.json", [call])
                rows.append(row)
            self.write(run / "results.json", rows)

    @staticmethod
    def write(path, data):
        path.write_text(json.dumps(data) + "\n")

    @staticmethod
    def read(path):
        return json.loads(path.read_text())

    def mutate(self, path, action):
        data = self.read(path)
        action(data)
        self.write(path, data)

    def result(self):
        return score(self.gold, self.original, self.staged)

    def test_fixed_eight_case_denominator_and_shared_completed_comparisons(self):
        result = self.result()
        original, staged = result["arms"]["original"], result["arms"]["staged"]
        self.assertEqual((8, 8), (original["overall_task_success_denominator"], staged["overall_task_success_denominator"]))
        self.assertEqual((6 / 8, 7 / 8), (original["overall_task_success"], staged["overall_task_success"]))
        self.assertEqual((7, 7), (original["completed"], staged["completed"]))
        self.assertEqual((6 / 7, 1), (original["label_accuracy_among_completed"], staged["label_accuracy_among_completed"]))
        self.assertEqual(1, original["execution_errors"])
        self.assertEqual(8, original["usage_total"]["model_calls"])
        self.assertEqual(160, staged["usage_total"]["output_tokens"])
        comparison = result["all_scheduled_task_comparison"]
        self.assertEqual((2, 1, 1 / 8), (comparison["fixes"], comparison["breaks"], comparison["staged_minus_original"]))
        shared = result["shared_completed_label_comparison"]
        self.assertEqual((6, 1, 0, 1 / 6), (shared["case_pairs"], shared["fixes"], shared["breaks"], shared["staged_minus_original"]))
        self.assertEqual(7, result["staged_round1_to_final"]["paired_completed_cases"])
        round_tasks = result["staged_round1_to_final"]["all_scheduled_task_comparison"]
        self.assertEqual(8, round_tasks["scheduled_label_evaluable_cases"])
        self.assertEqual(7 / 8, round_tasks["final_task_success"])
        self.assertEqual(8, len(result["cases"]))
        self.assertEqual("error", result["cases"][0]["staged"]["status"])

    def test_origin_denominators_and_original_edges_are_unsupported(self):
        result = self.result()
        original, staged = result["arms"]["original"], result["arms"]["staged"]
        self.assertEqual((2, 2), (original["origins"]["recall_denominator"], staged["origins"]["recall_denominator"]))
        self.assertEqual((0.5, 1), (original["origins"]["recall"], staged["origins"]["recall"]))
        self.assertEqual(1, original["origins"]["exclusions"]["execution_error"])
        self.assertIsNone(original["edges"]["precision"])
        self.assertIsNone(original["edges"]["recall"])
        self.assertEqual("unsupported_native_export", original["edges"]["status"])
        shared = result["shared_completed_origins"]["arms"]
        self.assertEqual((1, 1), (shared["original"]["recall_denominator"], shared["staged"]["recall_denominator"]))
        self.assertEqual((0, 1), (shared["original"]["recall"], shared["staged"]["recall"]))

    def test_unmapped_native_original_urls_are_disclosed_without_label_rejection(self):
        self.mutate(self.original / "c0-report.json", lambda d: d["sources"].append({"url": "https://unmapped.example", "is_original": True}))
        result = self.result()
        self.assertEqual(1, result["arms"]["original"]["unmapped_native_original_source_count"])
        self.assertEqual(6, result["arms"]["original"]["correct"])

    def test_nonboolean_native_source_flags_exclude_only_origin_scoring(self):
        self.mutate(self.original / "c0-report.json", lambda d: d["sources"][0].update(is_original="yes"))
        result = self.result()
        self.assertEqual(6, result["arms"]["original"]["correct"])
        self.assertEqual(["c0"], result["arms"]["original"]["origin_unavailable_nonboolean_native_flags"])
        self.assertEqual(1, result["arms"]["original"]["origins"]["recall_denominator"])

    def test_matching_full_input_fingerprint_need_not_hash_rewritten_selection(self):
        self.assertNotEqual("a" * 64, hashlib.sha256((self.original / "inputs.json").read_bytes()).hexdigest())
        self.assertEqual("a" * 64, self.result()["input_sha256"])

    def test_pairing_rejects_each_unequal_config_dimension(self):
        path = self.original / "config.json"
        saved = self.read(path)
        variants = [("model", "different-model"), ("reasoning_effort", "high"), ("input_sha256", "b" * 64),
                    ("budget_per_case", {**saved["budget_per_case"], "calls": 63})]
        for key, value in variants:
            with self.subTest(key=key):
                self.write(path, {**saved, key: value})
                with self.assertRaisesRegex(ValueError, "unequal"):
                    self.result()
        self.write(path, saved)

    def test_same_id_different_scope_or_snapshot_is_rejected(self):
        path = self.staged / "inputs.json"
        saved = self.read(path)
        for edit in (lambda d: d["cases"][0]["target"].update(evidence_scope=["other"]),
                     lambda d: d["cases"][0]["materials"][0].update(content="Changed source text")):
            data = deepcopy(saved)
            edit(data)
            self.write(path, data)
            with self.assertRaisesRegex(ValueError, "different selected corpus"):
                self.result()

    def test_missing_duplicate_and_extra_rows_are_rejected(self):
        path = self.original / "results.json"
        saved = self.read(path)
        for rows in (saved[:-1], saved + [saved[0]], saved[:-1] + [saved[0]]):
            self.write(path, rows)
            with self.assertRaisesRegex(ValueError, "Every scheduled case"):
                self.result()

    def test_errors_cannot_be_relabeled_unverifiable(self):
        self.mutate(self.staged / "results.json", lambda d: d[0].update(prediction="unverifiable"))
        with self.assertRaisesRegex(ValueError, "Execution error"):
            self.result()

    def test_staged_label_cannot_disagree_with_native_report(self):
        self.mutate(self.staged / "results.json", lambda d: d[1].update(prediction="true"))
        with self.assertRaisesRegex(ValueError, "validated native decision"):
            self.result()

    def test_invented_staged_rounds_are_rejected(self):
        self.mutate(self.staged / "c1-report.json", lambda d: d["operations"].pop())
        with self.assertRaisesRegex(ValueError, "checkpoints"):
            self.result()

    def test_original_extraction_must_be_grounded_and_preserve_decision(self):
        path = self.original / "results.json"
        saved = self.read(path)
        edits = (lambda row: row["extraction"].update(extractable=False),
                 lambda row: row["extraction"].update(extractable=1),
                 lambda row: row["extraction"].update(basis_quote="invented conclusion"),
                 lambda row: row["extraction"].update(decision="false"),
                 lambda row: row.update(extractor_answer_text="Rewritten report"))
        for edit in edits:
            data = deepcopy(saved)
            edit(data[0])
            self.write(path, data)
            with self.assertRaisesRegex(ValueError, "Ungrounded or changed"):
                self.result()

    def test_original_origin_export_must_match_native_source_flags(self):
        self.mutate(self.original / "results.json", lambda d: d[0]["prediction"].update(origins=["other"]))
        with self.assertRaisesRegex(ValueError, "native source flags"):
            self.result()

    def test_native_original_edges_cannot_be_fabricated(self):
        self.mutate(self.original / "results.json", lambda d: d[0]["prediction"].update(edges_evaluable=True))
        with self.assertRaisesRegex(ValueError, "unsupported edges"):
            self.result()

    def test_duplicate_source_urls_are_rejected_even_when_both_inputs_agree(self):
        for run in (self.original, self.staged):
            self.mutate(run / "inputs.json", lambda d: d["cases"][0]["materials"][1].update(url=d["cases"][0]["materials"][0]["url"]))
        with self.assertRaisesRegex(ValueError, "Ambiguous duplicate"):
            self.result()

    def test_invalid_assumptions_and_gold_flags_are_rejected(self):
        self.mutate(self.original / "config.json", lambda d: d.update(new_response_only=False))
        with self.assertRaisesRegex(ValueError, "source/execution assumption"):
            self.result()
        self.mutate(self.original / "config.json", lambda d: d.update(new_response_only=True))
        self.mutate(self.gold, lambda d: d["cases"][0].update(origin_evaluable="yes"))
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            self.result()

    def test_invalid_usage_is_rejected_including_error_rows(self):
        path = self.original / "results.json"
        saved = self.read(path)
        for invalid in (-1, True, 1.5, float("nan"), float("inf")):
            data = deepcopy(saved)
            data[1]["usage"]["input_tokens"] = invalid
            self.write(path, data)
            with self.assertRaises(ValueError):
                self.result()

    def test_replayed_response_and_swallowed_call_errors_are_rejected(self):
        self.mutate(self.staged / "c1-calls.json", lambda d: d[0].update(response_id="original-c0"))
        with self.assertRaisesRegex(ValueError, "Duplicate response ID"):
            self.result()
        self.mutate(self.staged / "c1-calls.json", lambda d: d[0].update(response_id="staged-c1", error_code="invalid_json"))
        with self.assertRaisesRegex(ValueError, "failed or mismatched"):
            self.result()

    def test_actual_model_and_call_usage_are_checked(self):
        self.mutate(self.original / "c0-calls.json", lambda d: d[0].update(actual_model="another-model"))
        with self.assertRaisesRegex(ValueError, "failed or mismatched"):
            self.result()
        self.mutate(self.original / "c0-calls.json", lambda d: d[0].update(actual_model="gpt-6-astra"))
        self.mutate(self.original / "results.json", lambda d: d[0]["usage"].update(output_tokens=99))
        with self.assertRaisesRegex(ValueError, "Token usage"):
            self.result()

    def test_cli_writes_reviewable_json_without_api_environment(self):
        output = self.root / "comparison.json"
        import os
        env = {key: value for key, value in os.environ.items() if "OPENAI" not in key}
        command = [sys.executable, str(ROOT / "experiments" / "staged_compare_score.py"), "--gold", str(self.gold),
                   "--original-run", str(self.original), "--staged-run", str(self.staged), "--output", str(output)]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(8, self.read(output)["scheduled_cases_per_arm"])


if __name__ == "__main__":
    unittest.main()
