"""Offline integration of target planning with staged interpretation and review."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
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
        dimensions = {item["kind"]: item["id"] for item in claim["dimensions"]}
        bindings = {
            "semantic_core": [dimensions["subject"], dimensions["predicate"]],
            "baseline_scope": [dimensions["baseline_scope"]],
            "source_lineage": [],
        }
        return {"decision": "accept", "repair_quote": "", "repair_issue": "", "notes": "",
            "probes": [{"claim_id": claim["id"], "kind": kind,
                         "dimension_ids": dimension_ids,
                         "question": "Check " + kind + ".",
                         "decision_impact": "A mismatch changes the decision."}
                       for kind, dimension_ids in bindings.items()]}


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
        payload = self.calls[-1]["payload"]
        return deepcopy(response(payload) if callable(response) else response)


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
        def response(payload):
            results = []
            for probe in payload["target_plan"]["probes"]:
                policy = probe["match_policy"]
                relation = "not_applicable"
                if policy == "same_referent":
                    relation = {"supported": "description", "contradicted": "different",
                                "conflicting": "ambiguous", "unresolved": "unresolved"}[verdict]
                elif policy == "exact_designation":
                    relation = {"supported": "exact", "contradicted": "different",
                                "conflicting": "ambiguous", "unresolved": "unresolved"}[verdict]
                results.append({"probe_id": probe["id"], "status": verdict,
                    "basis_indexes": [0], "rationale": "The shared passage answers this probe.",
                    "referent_relation": relation})
            return {"basis": [{"version_id": version, "quote": quote}],
                "probe_results": results,
                "rationale": "The exact source resolves this layer.",
                "gaps": [], "resolutions": []}
        return response

    @staticmethod
    def probe_view(logic, probes, claim_ids=("claim-1",)):
        """Minimal v2 projection for testing Python-owned layer assembly."""
        return {
            "schema_version": "decision-probe-v2",
            "logic": logic,
            "claims": [{"id": claim_id} for claim_id in claim_ids],
            "probes": probes,
        }

    @staticmethod
    def probe(identifier, claim_id="claim-1", *, policy="semantic_constraint"):
        return {
            "id": identifier,
            "claim_id": claim_id,
            "gate": "always",
            "match_policy": policy,
        }

    @staticmethod
    def raw_probe_layer(version, quote, results, *, gaps=None):
        return {
            "basis": [{"version_id": version, "quote": quote}],
            "probe_results": results,
            "rationale": "Each result is assembled by Python.",
            "gaps": [] if gaps is None else gaps,
            "resolutions": [],
        }

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

    def test_plan_aware_draft_grounding_gets_one_bounded_stage_repair(self):
        target, material, context, plan, views = self.fixture()
        invalid = {"atoms": [{"statement": "Issuer metadata names the publisher.",
            "quote": "Synthetic exact archive.", "qualifier_quotes": []}], "notes": ""}
        valid = {"atoms": [{"statement": material.content, "quote": material.content,
            "qualifier_quotes": ["did not", "above the 2024 baseline"]}], "notes": ""}
        no_lineage = {"citations": [], "origin": None, "notes": ""}
        accept = {"decision": "accept", "stage": "none", "quote": "", "issue": ""}
        client = SemanticClient([
            ("atoms", invalid), ("atoms", valid),
            ("lineage", no_lineage), ("critic", accept),
        ])
        decomposer = s.StagedDecomposer(client, target_plan=views, max_repairs=1)
        analysis = decomposer.decompose(target, material, context)
        self.assertTrue(analysis.fragments)
        self.assertEqual(["atoms", "atoms", "lineage", "critic"],
                         [call["stage"] for call in client.calls])
        self.assertEqual("structure_repair_requested", decomposer.history[0]["status"])
        repair = client.calls[1]["payload"]["repair"]
        self.assertIn("source quote is absent or not unique", repair["issue"])
        self.assertIn("material.content", repair["instruction"])
        self.assertNotIn("Synthetic exact archive.", json.dumps(repair))

        failing = SemanticClient([("atoms", invalid), ("atoms", invalid)])
        exhausted = s.StagedDecomposer(failing, target_plan=views, max_repairs=1)
        with self.assertRaisesRegex(s.StagedSemanticError,
                                    "source quote is absent or not unique"):
            exhausted.decompose(target, material, context)
        self.assertEqual(["atoms", "atoms"], [call["stage"] for call in failing.calls])
        self.assertEqual("structure_repair_exhausted", exhausted.history[-1]["status"])

    def test_disconnected_related_origin_is_repaired_out_of_whole_target(self):
        target, material, context, plan, views = self.fixture()
        other = p.MaterialVersion("b", "https://example.org/b",
            "Another agency produced a related estimate using a different baseline.",
            "2026-09-04T20:00:00Z", "2026-09-03T00:00:00Z",
            "2026-09-03T00:00:00Z", "Synthetic exact archive.")
        context["materials"].append(asdict(other))
        bad_origin = {"citations": [], "origin": {"kind": "original_record",
            "quote": other.content,
            "rationale": "This agency produced its different estimate, not the target report."},
            "notes": "The material is related but does not source the target."}
        no_lineage = {"citations": [], "origin": None,
            "notes": "Removed the unrelated whole-target origin."}
        accept = {"decision": "accept", "stage": "none", "quote": "", "issue": ""}
        client = SemanticClient([
            ("atoms", {"atoms": [{"statement": other.content, "quote": other.content,
                                    "qualifier_quotes": []}], "notes": ""}),
            ("lineage", bad_origin), ("lineage", no_lineage), ("critic", accept),
        ])
        decomposer = s.StagedDecomposer(client, target_plan=views, max_repairs=1)
        analysis = decomposer.decompose(target, other, context)
        self.assertEqual((), analysis.origins)
        self.assertEqual(["atoms", "lineage", "lineage", "critic"],
                         [call["stage"] for call in client.calls])
        self.assertEqual("structure_repair_requested", decomposer.history[1]["status"])
        repair = client.calls[2]["payload"]["repair"]
        self.assertIn("disconnected", repair["issue"])
        self.assertIn("complete replacement", repair["instruction"])

    def test_connected_upstream_origin_may_be_outside_evidence_scope(self):
        target, material, context, plan, views = self.fixture()
        other = p.MaterialVersion("b", "https://example.org/b",
            "This is the producing record for the target source's cited result.",
            "2026-09-04T20:00:00Z", "2026-09-03T00:00:00Z",
            "2026-09-03T00:00:00Z", "Synthetic exact archive.")
        context["materials"].append(asdict(other))
        context["relations"] = [asdict(p.Relation("a-to-b", "a", None, "cites", "declared",
            (p.Span("a", 0, len(material.content), material.content),),
            "The target source explicitly cites the producing record.", other.url))]
        origin = {"citations": [], "origin": {"kind": "original_record",
            "quote": other.content,
            "rationale": "The connected upstream explicitly identifies its producing role."},
            "notes": ""}
        accept = {"decision": "accept", "stage": "none", "quote": "", "issue": ""}
        client = SemanticClient([
            ("atoms", {"atoms": [{"statement": other.content, "quote": other.content,
                                    "qualifier_quotes": []}], "notes": ""}),
            ("lineage", origin), ("critic", accept),
        ])
        analysis = s.StagedDecomposer(client, target_plan=views).decompose(target, other, context)
        self.assertEqual(("b",), tuple(item.version_id for item in analysis.origins))
        self.assertNotIn("b", target.evidence_scope)
        self.assertEqual(["atoms", "lineage", "critic"],
                         [call["stage"] for call in client.calls])

    def test_evidence_repair_basis_cannot_escape_evidence_scope(self):
        target, material, context, plan, views = self.fixture()
        other = p.MaterialVersion("b", "https://example.org/b",
            "An outside report says output rose above its own baseline.",
            "2026-09-04T20:00:00Z", "2026-09-03T00:00:00Z",
            "2026-09-03T00:00:00Z", "Synthetic exact archive.")
        context["materials"].append(asdict(other))
        evidence_probe = next(item.id for item in plan.probes if item.kind == "baseline_scope")
        bad_repair = {"decision": "repair", "stage": "evidence", "probe_id": evidence_probe,
            "issue": "Use an out-of-scope report to change the evidence verdict.",
            "basis": [{"version_id": "b", "quote": other.content}]}
        client = SemanticClient([
            ("evidence", self.layer("a", material.content, "contradicted")),
            ("world", self.layer("a", material.content, "unresolved")),
            ("judgement_critic", bad_repair),
        ])
        verifier = s.StagedVerifier(client, target_plan=views)
        with self.assertRaisesRegex(s.StagedSemanticError,
                                    "outside the requested stage's permitted materials"):
            verifier.verify(target, context)
        self.assertEqual(["evidence", "world", "judgement_critic"],
                         [call["stage"] for call in client.calls])
        self.assertEqual("failed", verifier.history[-1]["status"])

    def test_world_repair_basis_may_use_visible_material_outside_evidence_scope(self):
        target, material, context, plan, views = self.fixture()
        other = p.MaterialVersion("b", "https://example.org/b",
            "An independent report addresses the same output claim.",
            "2026-09-04T20:00:00Z", "2026-09-03T00:00:00Z",
            "2026-09-03T00:00:00Z", "Synthetic exact archive.")
        context["materials"].append(asdict(other))
        world_probe = next(item.id for item in plan.probes if item.kind == "source_lineage")
        repair = {"decision": "repair", "stage": "world", "probe_id": world_probe,
            "issue": "Recheck whether the visible outside report changes world corroboration.",
            "basis": [{"version_id": "b", "quote": other.content}]}
        accept = {"decision": "accept", "stage": "none", "probe_id": "", "issue": "", "basis": []}
        client = SemanticClient([
            ("evidence", self.layer("a", material.content, "contradicted")),
            ("world", self.layer("a", material.content, "unresolved")),
            ("judgement_critic", repair),
            ("world", self.layer("b", other.content, "unresolved")),
            ("judgement_critic", accept),
        ])
        verifier = s.StagedVerifier(client, target_plan=views)
        result = verifier.verify(target, context)
        self.assertEqual("contradicted", result.evidence_verdict)
        self.assertEqual(["evidence", "world", "judgement_critic", "world", "judgement_critic"],
                         [call["stage"] for call in client.calls])
        self.assertEqual("b", client.calls[3]["payload"]["repair"]["basis"][0]["version_id"])

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

    def test_v2_requires_exactly_one_grounded_result_per_projected_probe(self):
        target, material, _, _, views = self.fixture()
        allowed = {material.version_id: asdict(material)}
        view = views["evidence"]
        raw = self.layer(material.version_id, material.content, "supported")(
            {"target_plan": view})
        assembled = s.StagedVerifier._assemble_layer(
            target, "evidence", raw, allowed, {}, [], view)
        self.assertEqual(
            {item["id"] for item in view["probes"]},
            {item.probe_id for item in assembled["probe_results"]},
        )

        malformed = {}
        unknown = deepcopy(raw)
        unknown["probe_results"][0]["probe_id"] = "not-in-plan"
        malformed["unknown"] = (unknown, "unknown or duplicate probe")

        duplicate = deepcopy(raw)
        duplicate["probe_results"][1]["probe_id"] = duplicate["probe_results"][0]["probe_id"]
        malformed["duplicate"] = (duplicate, "unknown or duplicate probe")

        missing = deepcopy(raw)
        missing["probe_results"].pop()
        malformed["missing"] = (missing, "exactly one result")

        outside = deepcopy(raw)
        outside["probe_results"][0]["basis_indexes"] = [len(outside["basis"])]
        malformed["outside_basis"] = (outside, "invalid shared-basis indexes")

        repeated = deepcopy(raw)
        repeated["probe_results"][0]["basis_indexes"] = [0, 0]
        malformed["duplicate_basis"] = (repeated, "invalid shared-basis indexes")

        for name, (candidate, message) in malformed.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                s.StagedVerifier._assemble_layer(
                    target, "evidence", candidate, allowed, {}, [], view)

    def test_v2_identity_policy_accepts_coreference_without_relaxing_exact_designation(self):
        target, material, _, _, _ = self.fixture()
        allowed = {material.version_id: asdict(material)}
        for relation in ("description", "anaphora"):
            with self.subTest(relation=relation):
                view = self.probe_view("single", [
                    self.probe("identity", policy="same_referent"),
                ])
                raw = self.raw_probe_layer(material.version_id, material.content, [{
                    "probe_id": "identity",
                    "status": "supported",
                    "basis_indexes": [0],
                    "rationale": "The passage unambiguously binds the same object.",
                    "referent_relation": relation,
                }])
                layer = s.StagedVerifier._assemble_layer(
                    target, "evidence", raw, allowed, {}, [], view)
                self.assertEqual("supported", layer["verdict"])
                self.assertEqual(relation, layer["probe_results"][0].referent_relation)

        designation_view = self.probe_view("single", [
            self.probe("designation", policy="exact_designation"),
        ])
        descriptive_only = self.raw_probe_layer(material.version_id, material.content, [{
            "probe_id": "designation",
            "status": "unresolved",
            "basis_indexes": [0],
            "rationale": "A description identifies the object but does not establish its exact title.",
            "referent_relation": "unresolved",
        }])
        layer = s.StagedVerifier._assemble_layer(
            target, "evidence", descriptive_only, allowed, {}, [], designation_view)
        self.assertEqual("unresolved", layer["verdict"])

        wrongly_relaxed = deepcopy(descriptive_only)
        wrongly_relaxed["probe_results"][0].update(
            status="supported", referent_relation="description")
        with self.assertRaisesRegex(
                ValueError,
                "exact-designation result with status supported requires referent_relation=exact"):
            s.StagedVerifier._assemble_layer(
                target, "evidence", wrongly_relaxed, allowed, {}, [], designation_view)
        self.assertIn("exact_designation -> supported: exact", s.PROBED_LAYER_RULE)
        self.assertIn("not_applicable is invalid for exact_designation",
                      s.JUDGMENT_CRITIC_PROMPT)

    def test_v2_python_aggregates_claims_with_frozen_and_or_logic(self):
        probes = [
            self.probe("left", "left-claim"),
            self.probe("right", "right-claim"),
        ]

        def assessments(left, right):
            return (
                p.ProbeAssessment("left", "left-claim", "evidence", left, (), "Left result."),
                p.ProbeAssessment("right", "right-claim", "evidence", right, (), "Right result."),
            )

        cases = (
            ("and", "supported", "supported", "supported"),
            ("and", "supported", "contradicted", "contradicted"),
            ("and", "supported", "unresolved", "unresolved"),
            ("or", "contradicted", "supported", "supported"),
            ("or", "contradicted", "contradicted", "contradicted"),
            ("or", "contradicted", "unresolved", "unresolved"),
        )
        for logic, left, right, expected in cases:
            with self.subTest(logic=logic, left=left, right=right):
                view = self.probe_view(logic, probes, ("left-claim", "right-claim"))
                self.assertEqual(expected, s._aggregate_probe_results(
                    view, assessments(left, right)))

    def test_v2_relational_logic_never_ands_separate_operands(self):
        statuses = ("supported", "contradicted", "conflicting", "unresolved")
        for logic in ("conditional", "comparison", "causal"):
            for status in statuses:
                with self.subTest(logic=logic, status=status):
                    result = p.ProbeAssessment("left", "left-claim", "evidence",
                                               status, (), "Relation assessed intact.")
                    single = self.probe_view(logic, [self.probe("left", "left-claim")],
                                             ("left-claim",))
                    self.assertEqual(status, s._aggregate_probe_results(single, (result,)))
                    multiple = self.probe_view(logic, [self.probe("left", "left-claim"),
                        self.probe("right", "right-claim")], ("left-claim", "right-claim"))
                    other = p.ProbeAssessment("right", "right-claim", "evidence",
                                              status, (), "Separate operand.")
                    self.assertEqual("unresolved", s._aggregate_probe_results(
                        multiple, (result, other)))
        for status in statuses:
            mixed = self.probe_view("mixed", [self.probe("left", "left-claim")],
                                    ("left-claim",))
            self.assertEqual("unresolved", s._aggregate_probe_results(mixed, (
                p.ProbeAssessment("left", "left-claim", "evidence", status, (), "Mixed."),)))

    def test_designation_relation_blocks_disconnected_dimension_support(self):
        probes = [
            self.probe("semantic"),
            self.probe("old-identity", policy="same_referent"),
            self.probe("new-designation", policy="exact_designation"),
            self.probe("designation-relation"),
        ]
        view = self.probe_view("single", probes)

        def result(identifier, status):
            return p.ProbeAssessment(identifier, "claim-1", "evidence", status,
                                     (), "Synthetic relation check.")

        disconnected = tuple(result(probe["id"],
            "unresolved" if probe["id"] == "designation-relation" else "supported")
            for probe in probes)
        self.assertEqual("unresolved",
                         s._aggregate_probe_results(view, disconnected))

        wrong_direction = tuple(result(probe["id"],
            "contradicted" if probe["id"] == "designation-relation" else "supported")
            for probe in probes)
        self.assertEqual("contradicted",
                         s._aggregate_probe_results(view, wrong_direction))

    def test_runtime_accepts_p07_shaped_renamed_plan_with_direction_probe(self):
        target = p.Target("p07",
            "NOAA renamed GOES-U to GOES-19 on June 25, 2024.",
            "2026-09-05T00:00:00Z", source_version_id="m12",
            assessment_mode="evidence", evidence_scope=("m13",))

        class RenameClient:
            def call(self, system, user, schema):
                payload = json.loads(user)
                if system == e.CLAIM_CONTRACT_PROMPT:
                    return {"claims": [{"statement": target.text,
                        "quote": target.text, "role": "main", "dimensions": [
                            {"kind": "subject", "quote": "NOAA"},
                            {"kind": "predicate", "quote": "renamed"},
                            {"kind": "entity_identity", "quote": "NOAA"},
                            {"kind": "entity_identity", "quote": "GOES-U"},
                            {"kind": "exact_designation", "quote": "GOES-19"},
                            {"kind": "time", "quote": "June 25, 2024"},
                        ]}], "logic": "single", "notes": ""}
                claim = payload["claim_contract"]["claims"][0]
                by_kind = {}
                for item in claim["dimensions"]:
                    by_kind.setdefault(item["kind"], []).append(item["id"])
                bindings = [
                    ("semantic_core", by_kind["subject"] + by_kind["predicate"]),
                    ("source_lineage", []),
                    ("time_boundary", by_kind["time"]),
                    ("designation_relation", [item["id"]
                                               for item in claim["dimensions"]]),
                ]
                bindings.extend(("entity_identity", [identifier])
                                for identifier in by_kind["entity_identity"])
                bindings.extend(("exact_designation", [identifier])
                                for identifier in by_kind["exact_designation"])
                return {"decision": "accept", "repair_quote": "",
                    "repair_issue": "", "notes": "", "probes": [{
                        "claim_id": claim["id"], "kind": kind,
                        "dimension_ids": dimensions,
                        "question": "Check " + kind + ".",
                        "decision_impact": "A mismatch changes the decision.",
                    } for kind, dimensions in bindings]}

        plan = e.TargetPlanner(RenameClient()).prepare(target)
        views = {stage: plan.projection(stage).to_payload()
                 for stage in ("atoms", "lineage", "critic", "evidence", "world")}
        s._validate_target_plan(views, target)
        self.assertEqual(e._DESIGNATION_ASSERTION_CUE.pattern,
                         s._PLAN_DESIGNATION_CUE.pattern)
        for predicate in ("rename", "renamed", "renames", "renaming"):
            with self.subTest(predicate=predicate):
                self.assertIsNotNone(s._PLAN_DESIGNATION_CUE.fullmatch(predicate))
        relation, = [probe for probe in plan.probes
                     if probe.kind == "designation_relation"]
        self.assertEqual({item.id for item in plan.claims[0].dimensions},
                         set(relation.dimension_ids))

        evidence_view = views["evidence"]

        def raw_layer(content, relation_status):
            results = []
            for probe in evidence_view["probes"]:
                relation_value = "not_applicable"
                if probe["match_policy"] == "same_referent":
                    relation_value = "exact"
                elif probe["match_policy"] == "exact_designation":
                    relation_value = "exact"
                status = (relation_status if probe["kind"] == "designation_relation"
                          else "supported")
                results.append({"probe_id": probe["id"], "status": status,
                    "basis_indexes": [0], "rationale": "Synthetic claimed support.",
                    "referent_relation": relation_value})
            return {"basis": [{"version_id": "m13", "quote": content}],
                "probe_results": results, "rationale": "Synthetic layer.",
                "gaps": [], "resolutions": []}

        disconnected_cases = (
            "NOAA operates satellites. GOES-U launched. GOES-19 appears in a catalogue. "
            "June 25, 2024 was a Tuesday.",
            "The archive renamed another satellite to TEST-1. NOAA operates GOES-U. "
            "GOES-19 appears in a catalogue dated June 25, 2024.",
        )
        for disconnected in disconnected_cases:
            allowed = {"m13": {"version_id": "m13", "content": disconnected}}
            for status in ("supported", "contradicted"):
                with self.subTest(disconnected=disconnected,
                                  relation_status=status), self.assertRaisesRegex(
                        ValueError, "naming-predicate basis span"):
                    s.StagedVerifier._assemble_layer(target, "evidence",
                        raw_layer(disconnected, status), allowed, {}, False,
                        plan_view=evidence_view)

        connected = "NOAA RENAMED GOES-U to goes-19 on July 7, 2024."
        accepted = s.StagedVerifier._assemble_layer(target, "evidence",
            raw_layer(connected, "contradicted"),
            {"m13": {"version_id": "m13", "content": connected}}, {}, False,
            plan_view=evidence_view)
        self.assertEqual("contradicted", accepted["verdict"])

        actual_p07 = raw_layer(connected, "contradicted")
        probes_by_id = {probe["id"]: probe for probe in evidence_view["probes"]}
        for result in actual_p07["probe_results"]:
            kind = probes_by_id[result["probe_id"]]["kind"]
            if kind == "time_boundary":
                result["status"] = "contradicted"
            elif kind == "semantic_core":
                result["status"] = "unresolved"
                result["basis_indexes"] = []
        assembled = s.StagedVerifier._assemble_layer(
            target, "evidence", actual_p07,
            {"m13": {"version_id": "m13", "content": connected}}, {}, False,
            plan_view=evidence_view)
        self.assertEqual("contradicted", assembled["verdict"])

        broken_protocol = deepcopy(actual_p07)
        exact_probe_id = next(probe["id"] for probe in evidence_view["probes"]
                              if probe["kind"] == "exact_designation")
        next(result for result in broken_protocol["probe_results"]
             if result["probe_id"] == exact_probe_id)["referent_relation"] = "not_applicable"
        with self.assertRaisesRegex(
                ValueError,
                "exact-designation result with status supported requires referent_relation=exact"):
            s.StagedVerifier._assemble_layer(
                target, "evidence", broken_protocol,
                {"m13": {"version_id": "m13", "content": connected}}, {}, False,
                plan_view=evidence_view)

    def test_runtime_keeps_p08_time_dimensions_atomic_under_partial_support(self):
        target = p.Target("p08",
            "The USGS Unified Geologic Map of the Moon released in 2020 combined six "
            "Apollo-era regional maps with newer lunar-mission data.",
            "2026-09-05T00:00:00Z", source_version_id="m14",
            assessment_mode="evidence", evidence_scope=("m15",))

        class MultiTimeClient:
            def call(self, system, user, schema):
                payload = json.loads(user)
                if system == e.CLAIM_CONTRACT_PROMPT:
                    return {"claims": [
                        {"statement": "The map was released in 2020.",
                         "quote": ("The USGS Unified Geologic Map of the Moon released "
                                   "in 2020"),
                         "role": "conjunct", "dimensions": [
                             {"kind": "subject", "quote":
                              "The USGS Unified Geologic Map of the Moon"},
                             {"kind": "predicate", "quote": "released"},
                             {"kind": "time", "quote": "in 2020"},
                         ]},
                        {"statement": ("The map combined six Apollo-era regional maps "
                                       "with newer lunar-mission data."),
                         "quote": target.text, "role": "conjunct", "dimensions": [
                             {"kind": "subject", "quote":
                              "The USGS Unified Geologic Map of the Moon"},
                             {"kind": "predicate", "quote": "combined"},
                             {"kind": "quantity_unit", "quote":
                              "six Apollo-era regional maps"},
                             {"kind": "time", "quote":
                              "Apollo-era regional maps"},
                             {"kind": "time", "quote":
                              "newer lunar-mission data"},
                         ]},
                    ], "logic": "and", "notes": ""}
                probes = []
                for claim in payload["claim_contract"]["claims"]:
                    by_kind = {}
                    for dimension in claim["dimensions"]:
                        by_kind.setdefault(dimension["kind"], []).append(dimension["id"])
                    bindings = [
                        ("semantic_core", by_kind["subject"] + by_kind["predicate"]),
                        ("source_lineage", []),
                    ]
                    bindings.extend(("time_boundary", [identifier])
                                    for identifier in by_kind.get("time", ()))
                    if "quantity_unit" in by_kind:
                        bindings.append(("quantity_unit", by_kind["quantity_unit"]))
                    probes.extend({
                        "claim_id": claim["id"], "kind": kind,
                        "dimension_ids": dimensions,
                        "question": "Check " + kind + ".",
                        "decision_impact": "A mismatch changes the decision.",
                    } for kind, dimensions in bindings)
                return {"decision": "accept", "repair_quote": "",
                    "repair_issue": "", "notes": "", "probes": probes}

        plan = e.TargetPlanner(MultiTimeClient()).prepare(target)
        views = {stage: plan.projection(stage).to_payload()
                 for stage in ("atoms", "lineage", "critic", "evidence", "world")}
        s._validate_target_plan(views, target)

        combination = next(claim for claim in plan.claims
                           if any(item.anchor.quote == "combined"
                                  for item in claim.dimensions))
        time_by_quote = {item.anchor.quote: item.id for item in combination.dimensions
                         if item.kind == "time"}
        combination_time_probes = [probe for probe in plan.probes
                                   if (probe.claim_id == combination.id and
                                       probe.kind == "time_boundary")]
        self.assertEqual({(identifier,) for identifier in time_by_quote.values()},
                         {probe.dimension_ids for probe in combination_time_probes})

        malformed = deepcopy(views)
        combined_ids = sorted(time_by_quote.values())
        for view in malformed.values():
            matching = [probe for probe in view["probes"]
                        if (probe["claim_id"] == combination.id and
                            probe["kind"] == "time_boundary")]
            if matching:
                matching[0]["dimension_ids"] = combined_ids
                removed_ids = {probe["id"] for probe in matching[1:]}
                view["probes"] = [probe for probe in view["probes"]
                                  if probe["id"] not in removed_ids]
        critic = malformed["critic"]
        canonical = {key: critic[key] for key in (
            "schema_version", "target_signature", "logic", "claims", "probes", "notes")}
        malformed_sha = hashlib.sha256(json.dumps(canonical, sort_keys=True,
            ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        for view in malformed.values():
            view["plan_sha256"] = malformed_sha
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            s._validate_target_plan(malformed, target)

        evidence_view = views["evidence"]
        newer_id = time_by_quote["newer lunar-mission data"]
        raw_results = []
        for probe in evidence_view["probes"]:
            unresolved = (probe["kind"] == "time_boundary" and
                          tuple(probe["dimension_ids"]) == (newer_id,))
            raw_results.append({
                "probe_id": probe["id"],
                "status": "unresolved" if unresolved else "supported",
                "basis_indexes": [] if unresolved else [0],
                "rationale": ("The newer-data boundary is not resolved."
                              if unresolved else "The passage supports this probe."),
                "referent_relation": "not_applicable",
            })
        assembled = s.StagedVerifier._assemble_layer(target, "evidence", {
            "basis": [{"version_id": "m15", "quote": target.text}],
            "probe_results": raw_results,
            "rationale": "The two time boundaries have independent results.",
            "gaps": [], "resolutions": [],
        }, {"m15": {"version_id": "m15", "content": target.text}}, {}, False,
            plan_view=evidence_view)
        self.assertEqual("unresolved", assembled["verdict"])
        status_by_probe = {item.probe_id: item.status
                           for item in assembled["probe_results"]}
        time_status_by_dimension = {
            probe["dimension_ids"][0]: status_by_probe[probe["id"]]
            for probe in evidence_view["probes"]
            if probe["kind"] == "time_boundary"
        }
        self.assertEqual("supported", time_status_by_dimension[
            time_by_quote["Apollo-era regional maps"]])
        self.assertEqual("unresolved", time_status_by_dimension[newer_id])

    def test_runtime_keeps_v2_archive_readable_but_rejects_v3_disguise(self):
        target = p.Target("archive",
            "Alpha is called Item Three.", "2026-09-05T00:00:00Z",
            source_version_id="m1", assessment_mode="evidence",
            evidence_scope=("m1",))

        class IdentityClient:
            def call(self, system, user, schema):
                payload = json.loads(user)
                if system == e.CLAIM_CONTRACT_PROMPT:
                    return {"claims": [{"statement": target.text,
                        "quote": target.text, "role": "main", "dimensions": [
                            {"kind": "subject", "quote": "Alpha"},
                            {"kind": "predicate", "quote": "called"},
                            {"kind": "exact_designation", "quote": "Item Three"},
                        ]}], "logic": "single", "notes": ""}
                claim = payload["claim_contract"]["claims"][0]
                by_kind = {item["kind"]: item["id"] for item in claim["dimensions"]}
                bindings = [
                    ("semantic_core", [by_kind["subject"], by_kind["predicate"]]),
                    ("exact_designation", [by_kind["exact_designation"]]),
                    ("designation_relation", [item["id"]
                                                for item in claim["dimensions"]]),
                    ("source_lineage", []),
                ]
                return {"decision": "accept", "repair_quote": "",
                    "repair_issue": "", "notes": "", "probes": [{
                        "claim_id": claim["id"], "kind": kind,
                        "dimension_ids": dimensions,
                        "question": "Did Alpha perform the calling action?",
                        "decision_impact": "A mismatch changes the decision.",
                    } for kind, dimensions in bindings]}

        plan = e.TargetPlanner(IdentityClient()).prepare(target)
        generated = {stage: plan.projection(stage).to_payload()
                     for stage in ("atoms", "lineage", "critic", "evidence", "world")}
        self.assertEqual({"decision-probe-v3"},
                         {view["schema_version"] for view in generated.values()})

        def rehash(views, schema_version):
            for view in views.values():
                view["schema_version"] = schema_version
            critic = views["critic"]
            canonical = {key: critic[key] for key in (
                "schema_version", "target_signature", "logic", "claims", "probes", "notes")}
            digest = hashlib.sha256(json.dumps(canonical, sort_keys=True,
                ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
            for view in views.values():
                view["plan_sha256"] = digest

        archived_v2 = deepcopy(generated)
        for view in archived_v2.values():
            for probe in view["probes"]:
                if probe["kind"] == "exact_designation":
                    probe["question"] = "Did Alpha perform the calling action?"
        rehash(archived_v2, "decision-probe-v2")
        s._validate_target_plan(archived_v2, target)

        disguised_v3 = deepcopy(archived_v2)
        rehash(disguised_v3, "decision-probe-v3")
        with self.assertRaisesRegex(ValueError, "identity question.*canonical"):
            s._validate_target_plan(disguised_v3, target)

    def test_v2_plan_tampering_is_rejected_before_any_model_call(self):
        target, material, context, plan, original = self.fixture()
        s._validate_target_plan(original, target)
        critic = original["critic"]
        canonical = {key: critic[key] for key in (
            "schema_version", "target_signature", "logic", "claims", "probes", "notes")}
        digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False,
                                           separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(plan.sha256, digest)

        def changed_claim(views):
            for view in views.values():
                view["claims"][0]["statement"] = "Tampered proposition."

        def policy(field, value):
            def change(views):
                for view in views.values():
                    for probe in view["probes"]:
                        if probe["kind"] == "semantic_core":
                            probe[field] = value
            return change

        def changed_projection(views):
            views["evidence"]["claims"][0]["statement"] = "Projection-only change."

        def changed_hash(views):
            for view in views.values():
                view["plan_sha256"] = "0" * 64

        def changed_notes(views):
            views["world"]["notes"] = "Unshared notes."

        mutations = (changed_claim, changed_projection, changed_hash, changed_notes,
            policy("routes", ["world"]), policy("gate", "positive_world_only"),
            policy("match_policy", "exact_designation"))
        for change in mutations:
            for operation in ("decompose", "verify"):
                with self.subTest(mutation=change.__name__, operation=operation):
                    views = deepcopy(original)
                    change(views)
                    client = SemanticClient([])
                    with self.assertRaises(ValueError):
                        if operation == "decompose":
                            s.StagedDecomposer(client, target_plan=views).decompose(
                                target, material, context)
                        else:
                            s.StagedVerifier(client, target_plan=views).verify(target, context)
                    self.assertEqual([], client.calls)

    def test_v2_probe_gap_is_linked_and_same_visible_url_gets_bounded_repair(self):
        target, material, context, _, views = self.fixture()

        def with_gap(locator):
            def response(payload):
                raw = self.layer(material.version_id, material.content, "unresolved")(payload)
                probe_id = payload["target_plan"]["probes"][0]["id"]
                raw["gaps"] = [{
                    "probe_id": probe_id,
                    "question": "Retrieve a distinct snapshot that resolves this probe.",
                    "action": "fetch",
                    "locator": locator,
                    "basis": [{"version_id": material.version_id, "quote": material.content}],
                    "decision_impact": "Without it this probe remains unresolved.",
                }]
                return raw
            return response

        repaired_url = "https://example.org/a?version=2026-09-03"
        accept = {"decision": "accept", "stage": "none", "probe_id": "",
                  "issue": "", "basis": []}
        client = SemanticClient([
            ("evidence", with_gap(material.url)),
            ("evidence", with_gap(repaired_url)),
            ("world", self.layer(material.version_id, material.content, "unresolved")),
            ("judgement_critic", accept),
        ])
        verifier = s.StagedVerifier(client, target_plan=views, max_structure_repairs=1)
        result = verifier.verify(target, context)
        self.assertEqual(["evidence", "evidence", "world", "judgement_critic"],
                         [call["stage"] for call in client.calls])
        self.assertEqual("structure_repair_requested", verifier.history[0]["status"])
        self.assertIn("already-visible snapshot URL",
                      client.calls[1]["payload"]["repair"]["issue"])
        self.assertEqual(1, len(result.gaps))
        self.assertEqual(repaired_url, result.gaps[0].locator)
        unresolved = {item.probe_id for item in result.evidence_probe_results
                      if item.status == "unresolved"}
        self.assertIn(result.gaps[0].probe_id, unresolved)


if __name__ == "__main__":
    unittest.main()
