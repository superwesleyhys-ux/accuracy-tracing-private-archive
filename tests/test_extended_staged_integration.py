"""Offline integration of target planning with staged interpretation and review."""
from copy import deepcopy
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
e = importlib.import_module("extended_semantic")
s = importlib.import_module("staged_semantic")
p = s.p


class PlanClient:
    def __init__(self): self.calls = []
    def call(self, system, user, schema):
        payload = json.loads(user); self.calls.append((system, payload))
        if system == e.CLAIM_CONTRACT_PROMPT:
            text = payload["target"]["text"]
            return {"claims": [{"statement": text, "quote": text, "role": "main",
                "dimensions": [{"kind": "subject", "quote": "Output"},
                    {"kind": "predicate", "quote": "rose"},
                    {"kind": "baseline_scope", "quote": "above the 2024 baseline"}]}],
                "logic": "single", "notes": ""}
        claim = payload["claim_contract"]["claims"][0]
        return {"decision": "accept", "repair_quote": "", "repair_issue": "", "notes": "",
            "probes": [{"claim_id": claim["id"], "kind": kind,
                         "question": "Check " + kind + ".",
                         "decision_impact": "A mismatch changes the decision."}
                       for kind in ("semantic_core", "baseline_scope", "source_lineage")]}


class SemanticClient:
    def __init__(self, responses): self.responses = list(responses); self.calls = []
    def call(self, system, user, schema):
        prompts = (("judgement_critic", s.JUDGMENT_CRITIC_PROMPT),
                   ("atoms", s.ATOMS_PROMPT), ("lineage", s.LINEAGE_PROMPT),
                   ("critic", s.CRITIC_PROMPT), ("evidence", s.EVIDENCE_PROMPT),
                   ("world", s.WORLD_PROMPT))
        stage = next(name for name, prompt in prompts if system.startswith(prompt))
        expected, response = self.responses.pop(0)
        if stage != expected: raise AssertionError(f"expected {expected}, received {stage}")
        self.calls.append({"stage": stage, "payload": json.loads(user)})
        return deepcopy(response)


class ExtendedStagedIntegrationTests(unittest.TestCase):
    def fixture(self):
        target = p.Target("target", "Output rose above the 2024 baseline.",
            "2026-09-04T20:00:00Z", source_version_id="a",
            assessment_mode="evidence", evidence_scope=("a",))
        material = p.MaterialVersion("a", "https://example.org/a",
            "Output did not rise above the 2024 baseline.",
            "2026-09-04T20:00:00Z", "2026-09-03T00:00:00Z",
            "2026-09-03T00:00:00Z", "Synthetic exact archive.")
        context = {"target": asdict(target), "materials": [asdict(material)], "analyses": {},
            "fragments": [], "relations": [], "origins": [], "gaps": [], "gap_registry": [],
            "verification_history": [], "usage": {"rounds": 1}}
        plan = e.TargetPlanner(PlanClient()).prepare(target)
        views = {stage: e.project_plan(plan, stage).to_payload()
                 for stage in ("atoms", "lineage", "critic", "evidence", "world")}
        return target, material, context, plan, views

    @staticmethod
    def layer(version, quote, verdict):
        return {"verdict": verdict, "basis": [{"version_id": version, "quote": quote}],
            "rationale": "The exact source resolves this layer.", "gaps": [], "resolutions": []}

    def test_stage_projections_are_shared_immutable_and_filtered(self):
        target, material, context, plan, views = self.fixture()
        quote = material.content
        client = SemanticClient([
            ("atoms", {"atoms": [{"statement": quote, "quote": quote,
                                   "qualifier_quotes": ["did not", "above the 2024 baseline"]}], "notes": ""}),
            ("lineage", {"citations": [], "origin": None, "notes": ""}),
            ("critic", {"decision": "accept", "stage": "none", "quote": "", "issue": ""}),
            ("evidence", self.layer("a", quote, "contradicted")),
            ("world", self.layer("a", quote, "unresolved")),
            ("judgement_critic", {"decision": "accept", "stage": "none", "probe_id": "",
                                    "issue": "", "basis": []}),
        ])
        analysis = s.StagedDecomposer(client, target_plan=views).decompose(target, material, context)
        context["fragments"] = [asdict(item) for item in analysis.fragments]
        result = s.StagedVerifier(client, target_plan=views).verify(target, context)
        self.assertEqual("contradicted", result.evidence_verdict)
        self.assertEqual(["atoms", "lineage", "critic", "evidence", "world", "judgement_critic"],
                         [call["stage"] for call in client.calls])
        for call in client.calls[:-1]:
            self.assertEqual(call["stage"], call["payload"]["target_plan"]["stage"])
            self.assertEqual(plan.sha256, call["payload"]["target_plan"]["plan_sha256"])
        atom_kinds = {item["kind"] for item in client.calls[0]["payload"]["target_plan"]["probes"]}
        lineage_kinds = {item["kind"] for item in client.calls[1]["payload"]["target_plan"]["probes"]}
        self.assertEqual({"semantic_core", "baseline_scope"}, atom_kinds)
        self.assertEqual({"source_lineage"}, lineage_kinds)
        self.assertEqual("critic", client.calls[-1]["payload"]["target_plan"]["stage"])

    def test_judgement_repair_reruns_only_named_layer_then_reviews_again(self):
        target, material, context, plan, views = self.fixture()
        quote = material.content
        evidence_probe = next(item.id for item in plan.probes if item.kind == "baseline_scope")
        repair = {"decision": "repair", "stage": "evidence", "probe_id": evidence_probe,
                  "issue": "The baseline probe conflicts with the first verdict.",
                  "basis": [{"version_id": "a", "quote": quote}]}
        accept = {"decision": "accept", "stage": "none", "probe_id": "", "issue": "", "basis": []}
        client = SemanticClient([
            ("evidence", self.layer("a", quote, "supported")),
            ("world", self.layer("a", quote, "unresolved")),
            ("judgement_critic", repair),
            ("evidence", self.layer("a", quote, "contradicted")),
            ("judgement_critic", accept),
        ])
        verifier = s.StagedVerifier(client, target_plan=views, max_repairs=1)
        result = verifier.verify(target, context)
        self.assertEqual("contradicted", result.evidence_verdict)
        self.assertEqual(["evidence", "world", "judgement_critic", "evidence", "judgement_critic"],
                         [call["stage"] for call in client.calls])
        self.assertEqual(evidence_probe, client.calls[3]["payload"]["repair"]["probe_id"])
        self.assertEqual(1, sum(item["status"] == "repair_requested" for item in verifier.history))
        self.assertEqual("accepted", verifier.history[-1]["status"])

    def test_changed_target_and_cross_routed_repair_fail_closed(self):
        target, material, context, plan, views = self.fixture()
        changed = p.Target(target.id, "Output fell above the 2024 baseline.", target.as_of,
                           target.source_version_id, target.assessment_mode, target.evidence_scope)
        client = SemanticClient([])
        with self.assertRaisesRegex(s.StagedSemanticError, "immutable target"):
            s.StagedDecomposer(client, target_plan=views).decompose(changed, material, context)
        self.assertEqual([], client.calls)

        quote = material.content
        lineage_probe = next(item.id for item in plan.probes if item.kind == "source_lineage")
        bad = {"decision": "repair", "stage": "evidence", "probe_id": lineage_probe,
               "issue": "Wrongly route lineage into evidence.", "basis": []}
        verifier_client = SemanticClient([
            ("evidence", self.layer("a", quote, "contradicted")),
            ("world", self.layer("a", quote, "unresolved")),
            ("judgement_critic", bad),
        ])
        with self.assertRaisesRegex(s.StagedSemanticError, "unrelated stage"):
            s.StagedVerifier(verifier_client, target_plan=views).verify(target, context)
        self.assertEqual("judgement_critic", verifier_client.calls[-1]["stage"])


if __name__ == "__main__":
    unittest.main()
