"""Baseline accounting/contract tests using only an injected mock transport."""
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
runner = importlib.import_module("staged_baseline_run")


@dataclass
class Report:
    direct_response: str
    sources: list


class BaselineRunTests(unittest.TestCase):
    def run_mock(self, failure=None, cases=None, unsafe=False):
        calls, depths = [], []

        def transport(**kwargs):
            calls.append(kwargs)
            extraction = kwargs["messages"][0]["content"].startswith("Extract the existing conclusion")
            if failure == "call" and not extraction:
                raise RuntimeError("MOCK_SECRET remote detail")
            answer = {"decision": "true", "extractable": True, "basis_quote": "The claim is supported."}
            if failure == "extractor" and extraction:
                answer["basis_quote"] = "Not actually present in the report"
            return SimpleNamespace(id=f"mock-{len(calls)}", model="wrong" if failure == "model" else kwargs["model"],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
                choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
                    content=json.dumps(answer) if extraction else "The claim is supported.", refusal=None))])

        base = runner.BudgetClient

        class Client(base):
            def __init__(self, model, budget, reasoning_effort=None):
                if failure == "constructor":
                    raise RuntimeError("MOCK_SECRET constructor detail")
                super().__init__(model, budget, transport=transport, reasoning_effort=reasoning_effort)

        class Agent:
            def __init__(self, llm, max_depth, console):
                depths.append(max_depth)
                self.llm = llm

            async def run(self, task):
                try:
                    answer = await self.llm.query("Mock original agent prompt", task, web_search=True)
                except RuntimeError:
                    # Exercise original-agent swallowed-error protection.
                    answer = "The claim is supported."
                return Report(answer, [{"url": "https://example.org/a", "is_original": True},
                                       {"url": "https://example.org/b", "is_original": False}])

        material = asdict(runner.p.MaterialVersion("a", "https://example.org/a", "The source supports the claim.",
            "2026-09-05T00:00:00Z", "2026-09-03T00:00:00Z", "2026-09-03T00:00:00Z", "Archive"))
        future = {**material, "version_id": "future", "url": "https://example.org/future",
                  "published_at": "2026-09-06T00:00:00Z"}
        identifiers = ["../unsafe"] if unsafe else ["c1", "c2"]
        data = {"schema_version": 1, "cases": [{"target": asdict(runner.p.Target(identifier,
            "The claim holds.", "2026-09-04T00:00:00Z", assessment_mode="evidence", evidence_scope=("a",))),
            "materials": [material, future], "seed_ids": ["a"], "full_evidence_control": False}
            for identifier in identifiers]}
        with tempfile.TemporaryDirectory() as folder:
            temp = Path(folder)
            inputs = temp / "inputs.json"
            inputs.write_text(json.dumps(data))
            original = temp / "original"
            (original / "agent").mkdir(parents=True)
            (original / "agent" / "core.py").write_text("# Mock original module\n")
            output = temp / "run"
            args = SimpleNamespace(inputs=str(inputs), output=str(output), original=str(original), cases=cases,
                model="mock-model", reasoning_effort="medium", max_calls=64, output_tokens=64000,
                per_call_tokens=4000, seconds=900, workers=2)
            stdout = io.StringIO()
            with patch.object(runner, "BudgetClient", Client), patch.object(runner, "load_original", return_value=Agent), \
                    patch.dict(sys.modules, {"rich.console": SimpleNamespace(Console=lambda **kw: None)}), \
                    patch.dict(runner.os.environ, {"OPENAI_API_KEY": "mock-only-no-api-access"}), redirect_stdout(stdout):
                status = runner.run(args)
                with self.assertRaises(FileExistsError):
                    runner.run(args)
            files = {p.name: json.loads(p.read_text()) for p in output.glob("*.json")}
            hashes = {p.relative_to(output / "executed-code").as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (output / "executed-code").rglob("*.py")}
            serialized = json.dumps(files) + stdout.getvalue()
            self.assertNotIn("MOCK_SECRET", serialized)
            self.assertNotIn("mock-only-no-api-access", serialized)
        return status, files, calls, depths, hashes

    def test_complete_corpus_contract_medium_caps_and_counted_report_only_extractor(self):
        status, files, calls, depths, hashes = self.run_mock()
        self.assertEqual(0, status)
        self.assertEqual([1, 1], depths)
        self.assertEqual(4, len(calls))
        self.assertTrue(all(c["reasoning_effort"] == "medium" and c["max_completion_tokens"] == 4000 for c in calls))
        for row in files["results.json"]:
            self.assertEqual("completed", row["status"])
            self.assertEqual("true", row["prediction"]["decision"])
            self.assertEqual(["a"], row["prediction"]["origins"])
            self.assertFalse(row["prediction"]["edges_evaluable"])
            self.assertEqual((2, 2, 2, 20, 40), (row["new_api_attempts"], row["returned_responses"],
                row["usage"]["model_calls"], row["usage"]["input_tokens"], row["usage"]["output_tokens"]))
            self.assertEqual(["a"], [m["version_id"] for m in files[row["id"] + "-input.json"]["materials"]])
        for call in calls:
            user = call["messages"][1]["content"]
            if call["messages"][0]["content"].startswith("Extract the existing conclusion"):
                self.assertEqual({"target", "answer_text"}, set(json.loads(user)))
            else:
                payload = json.loads(user.split("FIXED TASK CONTRACT AND COMPLETE CORPUS:\n")[1])
                self.assertEqual(["a"], [m["version_id"] for m in payload["materials"]])
        config = files["config.json"]
        self.assertEqual(["c1", "c2"], config["case_ids"])
        self.assertFalse(config["gold_read_during_inference"])
        self.assertEqual(0, config["automatic_transport_retries"])
        expected = {"accuracy/" + k: v for k, v in config["source_sha256"].items()}
        expected.update({"original/" + k: v for k, v in config["original_source_sha256"].items()})
        self.assertEqual(expected, hashes)

    def test_constructor_failure_preserves_all_scheduled_rows_without_calls(self):
        status, files, calls, _, _ = self.run_mock("constructor")
        self.assertEqual(1, status)
        self.assertEqual([], calls)
        self.assertEqual(2, len(files["results.json"]))
        self.assertTrue(all(r["status"] == "error" and r["prediction"] is None and
                            r["new_api_attempts"] == 0 for r in files["results.json"]))

    def test_swallowed_failure_and_bad_extractor_never_become_answers(self):
        for failure in ("call", "extractor", "model"):
            with self.subTest(failure=failure):
                status, files, _, _, _ = self.run_mock(failure)
                self.assertEqual(1, status)
                self.assertEqual(2, files["status.json"]["scheduled_cases"])
                self.assertTrue(all(r["status"] == "error" and r["prediction"] is None for r in files["results.json"]))
                if failure == "model":
                    self.assertEqual("unexpected_model", files["c1-calls.json"][0]["error_code"])

    def test_selection_and_safe_ids(self):
        _, files, _, _, _ = self.run_mock(cases="c2")
        self.assertEqual(["c2"], files["config.json"]["case_ids"])
        with self.assertRaises(ValueError):
            self.run_mock(cases="not-present")
        with self.assertRaises(ValueError):
            self.run_mock(unsafe=True)


if __name__ == "__main__":
    unittest.main()
