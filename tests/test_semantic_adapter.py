from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from model_io import Budget, BudgetClient, PriorResponseCache, digest, schema
from semantic_adapter import Decomposer, Verifier, compact_context
from loop_compare import load_inputs, score
from newsverify import provenance as p
from newsverify.decisions import present_decision
from newsverify.retrieval import SnapshotSearchProvider


class AdapterTests(unittest.TestCase):
    def test_scoring_rejects_mixed_question_modes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "gold.json").write_text(json.dumps({"cases": [{"id": "x", "decision": "true", "assessment_mode": "evidence"}]}))
            (root / "results.json").write_text(json.dumps([{"id": "x", "variant": "loop", "status": "completed",
                "prediction": {"decision": "true", "assessment_mode": "world"}, "checkpoints": []}]))
            with self.assertRaises(ValueError): score(NS(gold=str(root / "gold.json"), run=str(root)))

    def test_no_evidence_unknown_does_not_create_a_false_loop_gain(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "gold.json").write_text(json.dumps({"cases": [{"id": "x", "decision": "unverifiable", "assessment_mode": "world"}]}))
            (root / "results.json").write_text(json.dumps([{"id": "x", "variant": "loop", "status": "completed",
                "prediction": {"decision": "unverifiable", "assessment_mode": "world"}, "checkpoints": []}]))
            self.assertEqual(0, score(NS(gold=str(root / "gold.json"), run=str(root))))
            result = json.loads((root / "scores.json").read_text())
            self.assertEqual(result["overall"]["first_correct"], result["overall"]["last_correct"])

    def test_cache_reuses_exact_request_and_preserves_original_usage(self):
        record = {"actual_model": "m", "response_id": "recorded-response", "output": "{}",
                  "usage": {"input_tokens": 10, "output_tokens": 7}, "seconds": 3,
                  "request_digest": digest(["s", "u"]), "max_completion_tokens": 2500}
        live = []
        cache = PriorResponseCache([record], "m", lambda **kw: live.append(kw) or "live")
        kwargs = {"model": "m", "messages": [{"content": "s"}, {"content": "u"}], "max_completion_tokens": 2500}
        result = cache(**kwargs)
        self.assertEqual("recorded-response", result.id)
        self.assertEqual(7, result.usage.completion_tokens)
        self.assertEqual([], live)
        kwargs["messages"] = [{"content": "s"}, {"content": "changed"}]
        self.assertEqual("live", cache(**kwargs))
        self.assertEqual(1, len(live))

    def test_cache_does_not_reuse_failed_response(self):
        cache = PriorResponseCache([{"actual_model": "m", "error_type": "RuntimeError"}], "m", lambda **kw: "live")
        self.assertEqual("live", cache(model="m", messages=[{"content": "s"}, {"content": "u"}], max_completion_tokens=2500))

    def test_new_gap_and_verdict_values_are_constrained_before_live_call(self):
        gap = schema(p.Gap)["properties"]
        self.assertEqual(["fetch", "search", "reanalyse"], gap["action"]["enum"])
        self.assertNotIn("auto", gap["dimension"]["enum"])
        verification = schema(p.VerificationResult)["properties"]
        self.assertEqual("string", verification["world_verdict"]["type"])
        self.assertEqual(["evidence", "world"], verification["gaps"]["items"]["properties"]["dimension"]["enum"])

    def test_actual_semantic_adapter_engine_and_final_mapping_with_explicit_fake(self):
        value = p.MaterialVersion("m", "https://example.org/m", "The bridge remains closed.",
            "2026-09-04T20:00:00Z", "2026-09-04T10:00:00Z", "2026-09-04T10:00:00Z", "Fixture")
        target = p.Target("c", "The bridge is open.", "2026-09-04T20:00:00Z", assessment_mode="evidence", evidence_scope=("m",))
        basis = p.Span("m", 0, len(value.content), value.content)
        calls = []
        def fake(**kwargs):
            calls.append(kwargs)
            verification = "Return TWO distinct assessments" in kwargs["messages"][0]["content"]
            output = (p.VerificationResult("contradicted", (basis,), "Negation", evidence_verdict="contradicted", world_verdict="unresolved")
                      if verification else p.Analysis(fragments=(p.Fragment("f", "Closed", basis, target.id),)))
            return NS(id="fake", model="fake", usage=NS(prompt_tokens=1, completion_tokens=1),
                choices=[NS(finish_reason="stop", message=NS(content=json.dumps(asdict(output)), refusal=None))])
        client = BudgetClient("fake", Budget(), transport=fake)
        trace = p.run_provenance(target, SnapshotSearchProvider([value], ["m"]), Decomposer(client), Verifier(client))
        self.assertEqual([], trace["errors"])
        self.assertEqual("false", present_decision(trace)["decision"])
        self.assertEqual("unresolved", trace["fact_status"])
        self.assertEqual(2, len(calls))
        self.assertTrue(all("earlier_report" not in c["messages"][1]["content"] for c in calls))

    def test_frozen_inputs_have_no_gold_labels(self):
        path = Path(__file__).resolve().parents[1] / "experiments/inputs-v03.json"
        self.assertEqual(5, len(load_inputs(path)["cases"]))

    def test_compact_context_does_not_duplicate_full_analysis_history(self):
        full = {"materials": [], "relations": [], "origins": [], "gaps": [], "usage": {"rounds": 1},
                "analyses": {"sensitive-to-cost": "long output"}, "verification_history": ["long history"]}
        result = compact_context(full)
        self.assertNotIn("analyses", result)
        self.assertNotIn("verification_history", result)


if __name__ == "__main__": unittest.main()
