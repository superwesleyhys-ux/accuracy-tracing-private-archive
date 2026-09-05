"""Mock transport integration of the actual staged runner, schemas and engine.

These fixtures test execution and accounting, never accuracy or a live API.
"""
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
runner = importlib.import_module("staged_run")
stages = importlib.import_module("staged_semantic")
extension = importlib.import_module("extended_semantic")


class StagedRunMockTests(unittest.TestCase):
    def run_mock(self, *, include_failure=False, constructor_failure=False,
                 model_mismatch=False, target_extension=False):
        calls = []
        call_lock = threading.Lock()
        evidence_rounds = Counter()
        prompt_stages = [(getattr(stages, name.upper() + "_PROMPT"), name)
                         for name in ("atoms", "lineage", "critic", "evidence", "world")]
        prompt_stages += [(stages.JUDGMENT_CRITIC_PROMPT, "judgement_critic"),
                          (extension.CLAIM_CONTRACT_PROMPT, "claim_contract"),
                          (extension.EXTENSION_PROMPT, "extension")]
        claim_quote = "The bridge remains closed until 8 September 2026."
        origin_quote = "This is the original bridge closure record."
        citation_quote = "Source: https://example.org/a."

        def transport(**kwargs):
            # An unknown prompt fails immediately, including any LLM extractor.
            prompt = kwargs["messages"][0]["content"]
            stage = next(name for base, name in prompt_stages if prompt.startswith(base))
            payload = json.loads(kwargs["messages"][1]["content"])
            case_id = payload["target"]["id"]
            with call_lock:
                sequence = len(calls) + 1
                calls.append({"stage": stage, "case": case_id, "payload": payload,
                              "request": deepcopy(kwargs)})
                if stage == "evidence":
                    evidence_rounds[case_id] += 1
                evidence_round = evidence_rounds[case_id]
            if case_id == "failed-case" and stage == "evidence" and evidence_round == 2:
                raise RuntimeError("MOCK_SECRET transport detail must not be persisted")
            if stage == "claim_contract":
                text = payload["target"]["text"]
                answer = {"claims": [{"statement": text, "quote": text, "role": "main",
                    "dimensions": [{"kind": "subject", "quote": "bridge"},
                        {"kind": "predicate", "quote": "was open"},
                        {"kind": "time", "quote": "4 September 2026"}]}],
                    "logic": "single", "notes": ""}
            elif stage == "extension":
                claim = payload["claim_contract"]["claims"][0]
                dimensions = {item["kind"]: item["id"] for item in claim["dimensions"]}
                bindings = {
                    "semantic_core": [dimensions["subject"], dimensions["predicate"]],
                    "time_boundary": [dimensions["time"]],
                    "source_lineage": [],
                }
                answer = {"decision": "accept", "repair_quote": "", "repair_issue": "", "notes": "",
                    "probes": [{"claim_id": claim["id"], "kind": kind,
                        "dimension_ids": dimension_ids,
                        "question": "Check " + kind + ".", "decision_impact": "This can change the decision."}
                        for kind, dimension_ids in bindings.items()]}
            elif stage == "atoms":
                answer = {"atoms": [{"statement": claim_quote, "quote": claim_quote,
                                     "qualifier_quotes": ["until 8 September 2026"]}], "notes": ""}
            elif stage == "lineage":
                owner = payload["material"]["version_id"]
                answer = {"citations": [], "origin": None, "notes": ""}
                if owner == "a":
                    answer["origin"] = {"kind": "original_record", "quote": origin_quote,
                                        "rationale": "The source explicitly identifies its producing role."}
                elif payload["repair"] or payload["previous_analysis"]:
                    answer["citations"] = [{"locator": "https://example.org/a", "quote": citation_quote,
                        "kind": "cites", "rationale": "The source cites the original record.",
                        "decision_impact": "The original record establishes the cited provenance path."}]
            elif stage == "critic":
                missing = payload["material"]["version_id"] == "b" and not payload["drafts"]["lineage"]["citations"]
                answer = ({"decision": "repair", "stage": "lineage", "quote": citation_quote,
                           "issue": "The explicit source citation is missing from the lineage draft."}
                          if missing else {"decision": "accept", "stage": "none", "quote": "", "issue": ""})
            elif stage in ("evidence", "world"):
                basis = [{"version_id": "b", "quote": claim_quote}]
                answer = {"verdict": "unresolved", "basis": basis,
                          "rationale": "This fixture does not authenticate real world events.",
                          "gaps": [], "resolutions": []}
                if stage == "evidence" and evidence_round == 1:
                    answer["rationale"] = "Recheck the stated date condition before deciding entailment."
                    answer["gaps"] = [{"question": "Does the closure date include 4 September?",
                        "action": "reanalyse", "locator": "b", "blocking": True, "basis": basis,
                        "decision_impact": "The date condition determines whether the target is contradicted."}]
                elif stage == "evidence":
                    answer["verdict"] = "contradicted"
                    answer["rationale"] = "The source states the bridge is closed on the target date."
                    answer["resolutions"] = [{"gap_id": gap["id"], "basis": basis,
                        "rationale": "The quoted closure date includes the target date."}
                        for gap in payload["registered_gaps"]]
            else:
                answer = {"decision": "accept", "stage": "none", "probe_id": "",
                          "issue": "", "basis": []}
            return SimpleNamespace(id="mock-staged-response-" + str(sequence),
                model="unexpected-mock-model" if model_mismatch else kwargs["model"],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
                choices=[SimpleNamespace(finish_reason="stop",
                    message=SimpleNamespace(content=json.dumps(answer), refusal=None))])

        real_client = runner.BudgetClient

        class MockTransportClient(real_client):
            def __init__(self, model, budget, reasoning_effort=None):
                if constructor_failure:
                    raise RuntimeError("MOCK_SECRET constructor detail must not be persisted")
                super().__init__(model, budget, transport=transport, reasoning_effort=reasoning_effort)

        material_text = {
            "a": "City register: " + origin_quote + " " + claim_quote,
            "b": claim_quote + " " + citation_quote,
        }
        materials = [asdict(runner.p.MaterialVersion(owner, "https://example.org/" + owner, content,
            "2026-09-05T00:00:00Z", "2026-09-03T00:00:00Z", "2026-09-03T00:00:00Z",
            "Synthetic exact-version archive.")) for owner, content in material_text.items()]
        identifiers = ["success-case"] + (["failed-case"] if include_failure else [])
        data = {"schema_version": 1, "cases": [{
            "target": asdict(runner.p.Target(identifier, "The bridge was open on 4 September 2026.",
                "2026-09-04T20:00:00Z", source_version_id="b", assessment_mode="evidence",
                evidence_scope=("b",))), "materials": materials, "seed_ids": ["a", "b"],
            "full_evidence_control": False} for identifier in identifiers]}
        with tempfile.TemporaryDirectory(prefix="staged-mock-") as folder:
            temp = Path(folder)
            inputs = temp / "inputs.json"
            inputs.write_text(json.dumps(data))
            output = temp / "run"
            args = SimpleNamespace(inputs=str(inputs), output=str(output), cases=None,
                model="mock-model", reasoning_effort="medium", per_call_tokens=4000,
                output_tokens=64000, max_calls=64, seconds=900, rounds=2,
                max_repairs=1, workers=2, target_extension=target_extension)
            stdout = io.StringIO()
            # The dummy environment value only passes preflight. Every client
            # is constructed above with an explicit transport; no API is used.
            with patch.object(runner, "BudgetClient", MockTransportClient), \
                    patch.dict(runner.os.environ, {"OPENAI_API_KEY": "mock-only-no-api-access"}), \
                    redirect_stdout(stdout):
                status = runner.run(args)
            files = {path.name: json.loads(path.read_text()) for path in output.glob("*.json")}
            checkpoint_hashes = {path.name: hashlib.sha256(path.read_text().rstrip("\n").encode()).hexdigest()
                                 for path in output.glob("*-checkpoint-*.json")}
            copied_hashes = {path.relative_to(output / "executed-code").as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in (output / "executed-code").rglob("*.py")}
            original_input_hash = hashlib.sha256(inputs.read_bytes()).hexdigest()
            # Ensure neither arbitrary exception text nor the dummy key is
            # copied into any retained report, request log, or progress output.
            serialized = json.dumps(files) + stdout.getvalue()
            self.assertNotIn("MOCK_SECRET", serialized)
            self.assertNotIn("mock-only-no-api-access", serialized)
        return status, files, calls, checkpoint_hashes, copied_hashes, original_input_hash

    def test_two_materials_two_rounds_repair_native_verdict_and_frozen_evidence(self):
        status, files, calls, checkpoint_hashes, copied_hashes, input_hash = self.run_mock()
        self.assertEqual(0, status)
        row, = files["results.json"]
        self.assertEqual("completed", row["status"])
        self.assertEqual("false", row["prediction"])
        self.assertEqual(row["prediction"], row["native_prediction"]["decision"])
        self.assertEqual("unresolved", row["native_prediction"]["assessments"]["world"]["decision"])
        self.assertEqual(["unverifiable", "false"], [point["decision"] for point in row["checkpoints"]])
        self.assertEqual({"rounds": 2, "documents": 4, "unique_versions": 2,
            "decomposition_calls": 4, "verification_calls": 2}, row["engine_usage"])
        expected = ["atoms", "lineage", "critic", "atoms", "lineage", "critic", "lineage", "critic",
                    "evidence", "world", "atoms", "lineage", "critic", "atoms", "lineage", "critic",
                    "evidence", "world"]
        self.assertEqual(expected, [call["stage"] for call in calls])
        self.assertEqual(18, row["new_api_attempts"])
        self.assertEqual(18, row["returned_responses"])
        self.assertEqual(18, row["usage"]["model_calls"])
        self.assertEqual(180, row["usage"]["input_tokens"])
        self.assertEqual(360, row["usage"]["output_tokens"])
        self.assertTrue(all(call["request"]["max_completion_tokens"] == 4000 for call in calls))
        self.assertTrue(all(call["request"]["reasoning_effort"] == "medium" for call in calls))
        self.assertEqual(files["config.json"]["source_sha256"], copied_hashes)
        self.assertEqual(files["config.json"]["input_sha256"], input_hash)
        self.assertFalse(files["config.json"]["gold_read_during_inference"])
        self.assertEqual(0, files["config.json"]["automatic_transport_retries"])
        for point in row["checkpoint_hashes"]:
            self.assertEqual(point["sha256"], checkpoint_hashes[f"success-case-checkpoint-{point['round']}.json"])
        report = files["success-case-report.json"]
        self.assertEqual([], report["errors"])
        self.assertEqual("contradicted", report["verification_history"][-1]["evidence_verdict"])
        self.assertEqual("original_material_located", report["provenance_status"])
        self.assertEqual([], report["gaps"])
        self.assertEqual(4, len(report["analysis_history"]))
        self.assertEqual(report["analysis_history"][0]["analysis"]["fragments"],
                         report["analysis_history"][2]["analysis"]["fragments"])
        psi_history = files["success-case-psi-history.json"]
        self.assertEqual(1, sum(item["status"] == "repair_requested" for item in psi_history))
        self.assertEqual([1, 2], [point["round"] for point in files["success-case-retrieval.json"]])
        second_round_critic = calls[15]["payload"]
        self.assertIsNotNone(second_round_critic["previous_analysis"])
        self.assertEqual("b", second_round_critic["verifier_feedback"]["tasks"][0]["locator"])
        self.assertTrue(all({m["version_id"] for m in call["payload"]["materials"]} == {"b"}
                            for call in calls if call["stage"] == "evidence"))

    def test_failed_call_stays_in_matrix_with_prior_checkpoint_and_usage(self):
        status, files, calls, _, _, _ = self.run_mock(include_failure=True)
        self.assertEqual(1, status)
        rows = {row["id"]: row for row in files["results.json"]}
        self.assertEqual({"success-case", "failed-case"}, set(rows))
        self.assertEqual("completed", rows["success-case"]["status"])
        failed = rows["failed-case"]
        self.assertEqual("error", failed["status"])
        self.assertIsNone(failed["prediction"])
        self.assertEqual(17, failed["new_api_attempts"])
        self.assertEqual(16, failed["returned_responses"])
        self.assertEqual(17, failed["usage"]["model_calls"])
        self.assertEqual(160, failed["usage"]["input_tokens"])
        self.assertEqual(320, failed["usage"]["output_tokens"])
        self.assertEqual([1], [point["round"] for point in failed["checkpoint_hashes"]])
        self.assertEqual([1], [point["round"] for point in failed["checkpoints"]])
        self.assertEqual("sdk_error", files["failed-case-calls.json"][-1]["error_code"])
        self.assertEqual("RuntimeError", files["failed-case-calls.json"][-1]["error_type"])
        self.assertEqual("failed", files["failed-case-verification-history.json"][-1]["status"])
        self.assertEqual("verifier", files["failed-case-report.json"]["errors"][-1]["stage"])
        self.assertEqual("has_errors", files["status.json"]["status"])
        self.assertEqual(2, files["status.json"]["scheduled_cases"])
        self.assertEqual(1, files["status.json"]["completed"])
        self.assertEqual(len(calls), files["status.json"]["new_api_attempts"])
        self.assertEqual(35, files["status.json"]["new_api_attempts"])
        self.assertEqual(34, files["status.json"]["returned_responses"])

    def test_client_constructor_failure_retains_every_scheduled_case(self):
        status, files, calls, _, _, _ = self.run_mock(include_failure=True, constructor_failure=True)
        self.assertEqual(1, status)
        self.assertEqual([], calls)
        self.assertEqual(2, len(files["results.json"]))
        self.assertTrue(all(row["status"] == "error" and row["prediction"] is None
                            and row["new_api_attempts"] == 0 for row in files["results.json"]))
        self.assertEqual(2, files["status.json"]["scheduled_cases"])
        self.assertEqual(0, files["status.json"]["completed"])

    def test_unexpected_returned_model_retains_response_and_invalidates_case(self):
        status, files, calls, _, _, _ = self.run_mock(model_mismatch=True)
        self.assertEqual(1, status)
        row, = files["results.json"]
        self.assertEqual("error", row["status"])
        self.assertIsNone(row["prediction"])
        self.assertEqual(1, row["new_api_attempts"])
        self.assertEqual(1, row["returned_responses"])
        self.assertEqual(1, len(calls))
        self.assertEqual("unexpected_model", files["success-case-calls.json"][0]["error_code"])

    def test_target_extension_plans_once_routes_views_and_reviews_both_rounds(self):
        status, files, calls, checkpoint_hashes, copied_hashes, input_hash = self.run_mock(
            target_extension=True)
        self.assertEqual(0, status)
        self.assertEqual(["claim_contract", "extension"], [call["stage"] for call in calls[:2]])
        self.assertEqual(1, sum(call["stage"] == "claim_contract" for call in calls))
        self.assertEqual(1, sum(call["stage"] == "extension" for call in calls))
        self.assertEqual(2, sum(call["stage"] == "judgement_critic" for call in calls))
        self.assertEqual(22, len(calls))
        row, = files["results.json"]
        self.assertEqual("completed", row["status"])
        self.assertEqual(22, row["new_api_attempts"])
        self.assertEqual(files["success-case-target-plan.json"]["sha256"], row["target_plan_sha256"])
        self.assertEqual(["claim_contract", "extension"],
                         [item["stage"] for item in files["success-case-target-planner-history.json"]])
        self.assertTrue(files["config.json"]["target_extension"])
        self.assertIn("experiments/extended_semantic.py", copied_hashes)
        projections = files["success-case-target-plan.json"]["projections"]
        self.assertEqual({"atoms", "lineage", "critic", "evidence", "world"}, set(projections))
        self.assertEqual(1, len({view["plan_sha256"] for view in projections.values()}))
        for call in calls:
            if call["stage"] in projections:
                self.assertEqual(projections[call["stage"]], call["payload"]["target_plan"])


if __name__ == "__main__":
    unittest.main()
