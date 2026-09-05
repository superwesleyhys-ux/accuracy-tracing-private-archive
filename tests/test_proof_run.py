"""Mock-only experiment integration; no API access or accuracy measurement.

The original agent and every transport response are explicit synthetic fixtures.
The real runner, semantic codecs, checkpoint engine, and budget ledger execute.
"""

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
proof = importlib.import_module("proof_run")


@dataclass
class MockOriginalReport:
    direct_response: str
    sources: tuple = ()


class MockOriginalAgent:
    def __init__(self, client, **kwargs):
        self.client = client

    async def run(self, task):
        answer = await self.client.query("MOCK ORIGINAL", task)
        return MockOriginalReport(answer)


class ProofRunMockTests(unittest.TestCase):
    def run_mock(self, *, model_mismatch=False, seed_failure=False):
        transport_calls = []

        def transport(**kwargs):
            """Produce deterministic fixture JSON without creating an API client."""
            transport_calls.append(deepcopy(kwargs))
            spec = kwargs.get("response_format", {}).get("json_schema", {}).get("schema", {})
            properties = spec.get("properties", {})
            user = kwargs["messages"][1]["content"]
            if "fragments" in properties:
                payload = json.loads(user)
                material = proof.p.MaterialVersion(**payload["material"])
                target = proof.p.Target(**payload["target"])
                analysis = proof.p.ConservativeDecomposer().decompose(target, material, {})
                if material.version_id == target.source_version_id:
                    basis = (proof.p.Span(material.version_id, 0, len(material.content), material.content),)
                    analysis = proof.p.replace(analysis,
                        origins=(proof.p.OriginFinding(target.id, material.version_id, basis,
                            "original_record", "Mock original-record finding."),),
                        resolutions=(proof.p.Resolution("origin:" + target.id, basis,
                            "Mock origin resolution."),))
                answer = asdict(analysis)
            elif "evidence_verdict" in properties:
                payload = json.loads(user)
                material = proof.p.MaterialVersion(**payload["context"]["materials"][0])
                basis = (proof.p.Span(material.version_id, 0, len(material.content), material.content),)
                answer = asdict(proof.p.VerificationResult("supported", basis,
                    "Mock evidence finding.", evidence_verdict="supported", world_verdict="supported",
                    world_basis=basis, world_rationale="Mock world finding."))
            elif "selected_trial" in properties:
                answer = {"selected_trial": 2, "rationale": "Mock selection of third independent trial."}
            elif "extractable" in properties:
                answer_text = json.loads(user)["answer_text"]
                answer = {"decision": "true", "extractable": True, "basis_quote": answer_text}
            else:
                answer = "Mock original conclusion: the claim is supported."
            content = json.dumps(answer) if spec else answer
            return SimpleNamespace(id="mock-response-" + str(len(transport_calls)),
                model="mock-unexpected-model" if model_mismatch else kwargs["model"],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
                choices=[SimpleNamespace(finish_reason="stop",
                    message=SimpleNamespace(content=content, refusal=None))])

        real_client = proof.BudgetClient

        class MockTransportClient(real_client):
            def __init__(self, model, budget):
                super().__init__(model, budget, transport=transport)

        agent_package, agent_module = ModuleType("agent"), ModuleType("agent.core")
        agent_module.NewsTracingAgent = MockOriginalAgent
        agent_package.core = agent_module
        rich_package, rich_module = ModuleType("rich"), ModuleType("rich.console")
        rich_module.Console = lambda **kwargs: None
        rich_package.console = rich_module

        with tempfile.TemporaryDirectory(prefix="proof-mock-") as temporary:
            temp = Path(temporary)
            original = temp / "original"
            (original / "agent").mkdir(parents=True)
            (original / "agent" / "core.py").write_text("# Mock original module for source manifest.\n")
            materials = [asdict(proof.p.MaterialVersion(identifier, "https://example.org/" + identifier,
                "Synthetic source " + identifier + " supports the fixture.",
                "2026-09-05T00:00:00Z", "2026-08-01T00:00:00Z",
                "2026-08-01T00:00:00Z", "Mock exact-version archive.")) for identifier in ("a", "b", "c")]
            case = {"target": asdict(proof.p.Target("mock-case", "Synthetic fixture claim.",
                "2026-09-04T00:00:00Z", source_version_id="a", evidence_scope=("a",))),
                "materials": materials, "seed_ids": ["a", "b", "c"], "full_evidence_control": False}
            inputs = temp / "inputs.json"
            inputs.write_text(json.dumps({"schema_version": 1, "cases": [case]}))
            output = temp / "run"
            args = SimpleNamespace(inputs=str(inputs), original=str(original), output=str(output),
                                   model="mock-model", workers=1)
            seed = proof.seed_client

            def seed_adapter(*args):
                if seed_failure:
                    raise ValueError("Mock invalid shared prefix")
                return seed(*args)

            # The dummy environment value only passes the runner's preflight.
            # The patched constructor unconditionally injects transport above.
            with patch.object(proof, "BudgetClient", MockTransportClient), \
                    patch.object(proof, "seed_client", seed_adapter), \
                    patch.dict(proof.os.environ, {"OPENAI_API_KEY": "mock-only-no-api-access"}), \
                    patch.dict(sys.modules, {"agent": agent_package, "agent.core": agent_module,
                                             "rich": rich_package, "rich.console": rich_module}), \
                    redirect_stdout(io.StringIO()):
                status = proof.run(args)
            artifacts = {file.name: json.loads(file.read_text()) for file in output.glob("*.json")}
            checkpoint_text = (output / "mock-case-shared-checkpoint.json").read_text().rstrip("\n") \
                if (output / "mock-case-shared-checkpoint.json").exists() else None
        return status, artifacts, transport_calls, checkpoint_text

    def test_mocked_five_arm_run_preserves_prefix_and_accounts_for_real_rounds(self):
        status, files, calls, checkpoint_text = self.run_mock()
        self.assertEqual(0, status)
        rows = {row["arm"]: row for row in files["results.json"]}
        self.assertEqual(set(proof.ARMS), set(rows))
        self.assertTrue(all(row["status"] == "completed" for row in rows.values()))
        expected_logical = {"original": 2, "single": 5, "loop_psi": 13,
                            "loop_frozen": 7, "independent": 14}
        expected_new = {"original": 2, "single": 5, "loop_psi": 9,
                        "loop_frozen": 3, "independent": 10}
        self.assertEqual(29, len(calls))
        for arm in proof.ARMS:
            with self.subTest(arm=arm):
                self.assertEqual(expected_logical[arm], rows[arm]["usage"]["model_calls"])
                self.assertEqual(expected_new[arm], rows[arm]["actual_new_api_usage"]["model_calls"])
                self.assertEqual(expected_logical[arm] * 20, rows[arm]["usage"]["output_tokens"])
                self.assertLessEqual(rows[arm]["usage"]["model_calls"], proof.BUDGET.calls)
        self.assertEqual(29, files["status.json"]["actual_new_api_usage"]["model_calls"])
        self.assertEqual(41, files["status.json"]["logical_arm_calls"])

        prefix = files["mock-case-single-calls.json"][:4]
        prefix_report = files["mock-case-single-report.json"]
        self.assertEqual("complete", prefix_report["stop_reason"])
        digest = hashlib.sha256(checkpoint_text.encode()).hexdigest()
        self.assertEqual(digest, rows["single"]["state_sha256"])
        for arm in ("loop_psi", "loop_frozen", "independent"):
            self.assertEqual(digest, rows[arm]["shared_state_sha256"])
            shared = files["mock-case-" + arm + "-calls.json"][:4]
            self.assertTrue(all(record["shared_first_round"] for record in shared))
            self.assertEqual(prefix, [{k: v for k, v in record.items() if k != "shared_first_round"}
                                      for record in shared])

        for arm in ("loop_psi", "loop_frozen"):
            report = files["mock-case-" + arm + "-report.json"]
            self.assertEqual(3, report["usage"]["verification_calls"])
            self.assertEqual([1, 2, 3], [item["round"] for item in report["verification_history"]])
            self.assertEqual([2, 3], [item["round"] for item in files["mock-case-" + arm + "-retrieval.json"]])
            self.assertEqual(prefix_report["operations"][:-1],
                             report["operations"][:len(prefix_report["operations"]) - 1])
        self.assertEqual(6, rows["loop_frozen"]["frozen_psi_adapter_returns"])
        frozen = files["mock-case-loop_frozen-report.json"]
        self.assertEqual(prefix_report["analyses"], frozen["analyses"])
        fresh_records = files["mock-case-loop_frozen-calls.json"][4:]
        self.assertEqual(2, sum("previous_verification" in json.loads(record["user"])
                                for record in fresh_records))
        self.assertTrue(all("material" not in json.loads(record["user"]) for record in fresh_records))

        independent_records = files["mock-case-independent-calls.json"][4:]
        decomposition = [json.loads(record["user"]) for record in independent_records
                         if "material" in json.loads(record["user"])]
        self.assertEqual(6, len(decomposition))
        self.assertTrue(all(item["context"]["round"] == 1 for item in decomposition))
        self.assertTrue(all(item["previous_verification"] == [] for item in decomposition))
        self.assertTrue(all(item["previous_analysis"] is None for item in decomposition))
        for index in (1, 2):
            self.assertEqual(1, files["mock-case-independent-trial-" + str(index) + ".json"]
                             ["usage"]["verification_calls"])

    def test_mocked_invalid_prefix_is_recorded_per_branch_without_aborting_run(self):
        status, files, calls, _ = self.run_mock(seed_failure=True)
        self.assertEqual(1, status)
        rows = {row["arm"]: row for row in files["results.json"]}
        self.assertEqual(set(proof.ARMS), set(rows))
        self.assertEqual("completed", rows["single"]["status"])
        for arm in ("loop_psi", "loop_frozen", "independent"):
            self.assertEqual("error", rows[arm]["status"])
            self.assertEqual("ValueError", rows[arm]["error_type"])
        self.assertEqual(7, len(calls))
        self.assertEqual("has_errors", files["status.json"]["status"])

    def test_mocked_model_mismatch_never_counts_as_completed_prediction(self):
        status, files, calls, _ = self.run_mock(model_mismatch=True)
        self.assertEqual(1, status)
        rows = files["results.json"]
        self.assertEqual(5, len(rows))
        self.assertTrue(all(row["status"] == "error" for row in rows))
        self.assertTrue(all(row["prediction"] is None for row in rows))
        self.assertEqual(2, len(calls))
        for arm in ("original", "single"):
            records = files["mock-case-" + arm + "-calls.json"]
            self.assertTrue(records[0].get("error_type"))
        by_arm = {row["arm"]: row for row in rows}
        for arm in ("loop_psi", "loop_frozen", "independent"):
            self.assertEqual(by_arm["single"]["usage"], by_arm[arm]["usage"])
            self.assertTrue(by_arm[arm]["inherited_failed_prefix"])
            self.assertEqual(0, by_arm[arm]["actual_new_api_usage"]["model_calls"])


if __name__ == "__main__":
    unittest.main()
