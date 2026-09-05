"""Deterministic stage-contract regressions; every model response is synthetic."""
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
s = importlib.import_module("staged_semantic")
e = importlib.import_module("extended_semantic")
p = s.p

ACCEPT = {"decision": "accept", "stage": "none", "quote": "", "issue": ""}
NO_LINEAGE = {"citations": [], "origin": None, "notes": ""}


def atoms(*items, notes=""):
    return {"atoms": [{"statement": statement, "quote": quote, "qualifier_quotes": list(qualifiers)}
                      for statement, quote, qualifiers in items], "notes": notes}


def layer(version, quote, verdict="contradicted"):
    return {"verdict": verdict, "basis": [{"version_id": version, "quote": quote}],
            "rationale": "Synthetic source-backed layer assessment.", "gaps": [], "resolutions": []}


class ScriptClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def call(self, system, user, json_schema):
        names = ("atoms", "lineage", "critic", "evidence", "world")
        stage = next(name for name in names if system == getattr(s, name.upper() + "_PROMPT"))
        expected, result = self.responses.pop(0)
        if stage != expected:
            raise AssertionError(f"expected {expected}, received {stage}")
        spec = getattr(s, stage.upper() + "_SCHEMA") if stage in names[:3] else s.LAYER_SCHEMA
        if json_schema != spec:
            raise AssertionError("stage schema mismatch")
        self.calls.append({"stage": stage, "payload": json.loads(user)})
        return deepcopy(result)


class V4PlanClient:
    """Return one minimal valid model draft; Python owns the final v4 plan."""

    def call(self, system, user, json_schema):
        payload = json.loads(user)
        if system == e.CLAIM_CONTRACT_PROMPT:
            text = payload["target"]["text"]
            return {"claims": [{
                "statement": "ignored advisory text",
                "quote": text,
                "role": "main",
                "dimensions": [
                    {"kind": "subject", "quote": "Output"},
                    {"kind": "predicate", "quote": "rose"},
                ],
            }], "logic": "single", "notes": "ignored model notes"}
        claim = payload["claim_contract"]["claims"][0]
        dimensions = {item["kind"]: item for item in claim["dimensions"]}
        bindings = (
            ("predicate_core", [dimensions["predicate"]["id"]]),
            ("claim_composition", [item["id"] for item in claim["dimensions"]]),
            ("source_lineage", []),
        )
        ledger = []
        for segment in payload["coverage_segments"]:
            overlapping = sorted(item["id"] for item in claim["dimensions"]
                if max(segment["anchor"]["start"], item["anchor"]["start"]) <
                   min(segment["anchor"]["end"], item["anchor"]["end"]))
            if overlapping:
                status = "covered_by_dimension"
            elif segment["cue_kind"] == "logic_connector":
                status = "logic_connector"
            else:
                status = "context_only"
            ledger.append({"segment_id": segment["id"], "status": status,
                           "dimension_ids": overlapping})
        return {
            "decision": "accept", "repair_quote": "", "repair_issue": "",
            "probes": [{"claim_id": claim["id"], "kind": kind,
                        "dimension_ids": dimension_ids,
                        "question": "model text must be discarded",
                        "decision_impact": "model text must be discarded"}
                       for kind, dimension_ids in bindings],
            "coverage_ledger": ledger, "notes": "ignored model notes",
        }


class StagedSemanticTests(unittest.TestCase):
    def fixture(self, content="Output did not rise above the 2024 baseline.", owner="a"):
        material = p.MaterialVersion(owner, "https://example.org/" + owner, content,
            "2026-09-04T20:00:00Z", "2026-09-03T00:00:00Z", "2026-09-03T00:00:00Z", "Synthetic archive.")
        target = p.Target("target", "Output rose above the 2024 baseline.", "2026-09-04T20:00:00Z",
                          source_version_id=owner, assessment_mode="evidence", evidence_scope=(owner,))
        context = {"target": asdict(target), "materials": [asdict(material)], "analyses": {},
                   "gaps": [], "gap_registry": [], "relations": [], "origins": [], "usage": {"rounds": 1}}
        return target, material, context

    def v4_fixture(self):
        target = p.Target("target", "Output rose.", "2026-09-04T20:00:00Z",
                          source_version_id="a", assessment_mode="evidence",
                          evidence_scope=("a",))
        plan = e.TargetPlanner(V4PlanClient()).prepare(target)
        views = {stage: e.project_plan(plan, stage).to_payload()
                 for stage in ("atoms", "lineage", "critic", "evidence", "world")}
        return target, plan, views

    @staticmethod
    def reproject_v4(views):
        """Recompute a caller-controlled hash and all projections after critic edits."""
        critic = views["critic"]
        canonical = {key: deepcopy(critic[key]) for key in (
            "schema_version", "target_signature", "logic", "claims", "probes",
            "coverage_ledger", "notes")}
        sha = hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False,
            separators=(",", ":")).encode()).hexdigest()
        for stage, view in views.items():
            probes = deepcopy(critic["probes"] if stage == "critic" else [
                item for item in critic["probes"] if stage in item["routes"]])
            claim_ids = {item["claim_id"] for item in probes}
            claims = deepcopy(critic["claims"] if stage == "critic" else [
                item for item in critic["claims"] if item["id"] in claim_ids])
            dimension_ids = {dimension["id"] for claim in claims
                             for dimension in claim["dimensions"]}
            ledger = deepcopy(critic["coverage_ledger"] if stage == "critic" else [
                item for item in critic["coverage_ledger"]
                if item["claim_id"] in claim_ids or
                   set(item["dimension_ids"]) & dimension_ids])
            view.update(schema_version="decision-probe-v4",
                        target_signature=critic["target_signature"],
                        plan_sha256=sha, stage=stage, logic=critic["logic"],
                        claims=claims, probes=probes, coverage_ledger=ledger,
                        notes=critic["notes"])

    def test_missing_negation_or_baseline_repairs_only_atoms_and_reviews_both_drafts(self):
        target, material, context = self.fixture()
        quote = material.content
        corrected = atoms((quote, quote, ("did not", "2024 baseline")))
        for missing, statement in (("negation", "Output rose above the 2024 baseline."),
                                   ("comparison baseline", "Output did not rise.")):
            with self.subTest(missing=missing):
                repair = {"decision": "repair", "stage": "atoms", "quote": quote,
                          "issue": "Restore the missing " + missing + "."}
                client = ScriptClient(("atoms", atoms((statement, quote, ()))),
                    ("lineage", NO_LINEAGE), ("critic", repair), ("atoms", corrected), ("critic", ACCEPT))
                psi = s.StagedDecomposer(client, max_repairs=1)
                result = psi.decompose(target, material, context)
                self.assertEqual(["atoms", "lineage", "critic", "atoms", "critic"],
                                 [call["stage"] for call in client.calls])
                self.assertEqual(quote, result.fragments[0].text)
                self.assertEqual(("did not", "2024 baseline"), result.fragments[0].qualifiers)
                self.assertEqual({"quote": quote, "issue": repair["issue"]}, client.calls[3]["payload"]["repair"])
                self.assertEqual(corrected, client.calls[4]["payload"]["drafts"]["atoms"])
                self.assertEqual(NO_LINEAGE, client.calls[4]["payload"]["drafts"]["lineage"])

    def test_second_repair_request_exhausts_budget_without_returning_candidate(self):
        target, material, context = self.fixture()
        original = deepcopy(context)
        draft = atoms((material.content, material.content, ()))
        repair = {"decision": "repair", "stage": "atoms", "quote": material.content,
                  "issue": "The baseline qualifier remains missing."}
        client = ScriptClient(("atoms", draft), ("lineage", NO_LINEAGE), ("critic", repair),
                              ("atoms", draft), ("critic", repair))
        psi = s.StagedDecomposer(client, max_repairs=1)
        with self.assertRaisesRegex(s.StagedSemanticError, "repair budget exhausted"):
            psi.decompose(target, material, context)
        self.assertEqual("repair_exhausted", psi.history[-1]["status"])
        self.assertEqual(5, len(client.calls))
        self.assertEqual(original, context)
        self.assertEqual([], client.responses)

    def test_unknown_or_out_of_scope_basis_and_unregistered_resolution_fail_closed(self):
        target, material, context = self.fixture()
        _, other, _ = self.fixture(owner="b")
        context["materials"].append(asdict(other))
        unknown_resolution = layer("a", material.content)
        unknown_resolution["resolutions"] = [{"gap_id": "never-registered", "basis": unknown_resolution["basis"],
                                               "rationale": "A model cannot invent registration."}]
        provenance_resolution = deepcopy(unknown_resolution)
        provenance_resolution["resolutions"][0]["gap_id"] = "origin:target"
        context["gap_registry"] = [asdict(p.Gap("origin:target", "Find the original record."))]
        for label, response in (("unknown version", layer("missing", material.content)),
                                ("outside scope", layer("b", material.content)),
                                ("unknown resolution", unknown_resolution),
                                ("wrong stage resolution", provenance_resolution)):
            with self.subTest(label=label):
                client = ScriptClient(("evidence", response))
                verifier = s.StagedVerifier(client)
                with self.assertRaises(s.StagedSemanticError):
                    verifier.verify(target, context)
                self.assertEqual("failed", verifier.history[-1]["status"])
                self.assertEqual(["a"], [m["version_id"] for m in client.calls[0]["payload"]["materials"]])
                self.assertEqual([], client.calls[0]["payload"]["registered_gaps"])

    def test_absent_exact_quote_stops_before_lineage_or_critic(self):
        target, material, context = self.fixture()
        client = ScriptClient(("atoms", atoms(("Output rose.", "Output rose.", ()))))
        psi = s.StagedDecomposer(client)
        with self.assertRaisesRegex(s.StagedSemanticError, "source quote is absent or not unique"):
            psi.decompose(target, material, context)
        self.assertEqual(["atoms"], [call["stage"] for call in client.calls])
        self.assertEqual("failed", psi.history[-1]["status"])

    def test_two_round_replacement_retains_supported_atom_and_withdraws_old_id(self):
        quote = "The bridge is closed."
        target, material, _ = self.fixture(quote)
        first = atoms((quote, quote, ()), ("The bridge is open.", quote, ()))
        second = atoms((quote, quote, ()), notes="Withdraw the unsupported open interpretation.")
        client = ScriptClient(("atoms", first), ("lineage", NO_LINEAGE), ("critic", ACCEPT),
            ("evidence", layer("a", quote)), ("world", layer("a", quote, "unresolved")),
            ("atoms", second), ("lineage", NO_LINEAGE), ("critic", ACCEPT),
            ("evidence", layer("a", quote)), ("world", layer("a", quote, "unresolved")))

        class Provider:
            def search(self, target, tasks, round_number, limit):
                return iter((material,))

        report = p.run_provenance(target, Provider(), s.StagedDecomposer(client), s.StagedVerifier(client),
                                  p.TraceConfig(max_rounds=2, experimental_force_rounds=True))
        self.assertEqual([], report["errors"])
        old = {f["text"]: f["id"] for f in report["analysis_history"][0]["analysis"]["fragments"]}
        self.assertEqual([old[quote]], [f["id"] for f in report["fragments"]])
        self.assertNotIn(old["The bridge is open."], {f["id"] for f in report["fragments"]})
        self.assertEqual(2, len(report["analysis_history"]))
        self.assertEqual(json.loads(json.dumps(report["analysis_history"][0]["analysis"])),
                         client.calls[5]["payload"]["previous_analysis"])
        self.assertEqual([], report["resolutions"])
        self.assertEqual([], report["relations"])
        self.assertEqual(2, report["usage"]["verification_calls"])

    def test_qualifier_repeated_elsewhere_resolves_inside_unique_parent_quote(self):
        quote = "The 2024 output did not rise."
        target, material, context = self.fixture("Archive 2024 summary. " + quote)
        client = ScriptClient(("atoms", atoms((quote, quote, ("2024", "did not")))),
                              ("lineage", NO_LINEAGE), ("critic", ACCEPT))
        result = s.StagedDecomposer(client).decompose(target, material, context)
        fragment, = result.fragments
        year, negation = fragment.qualifier_spans
        self.assertEqual(material.content.index("2024", fragment.span.start), year.start)
        self.assertEqual("2024", material.content[year.start:year.end])
        self.assertTrue(fragment.span.start <= negation.start < negation.end <= fragment.span.end)

    def test_qualifier_outside_parent_or_ambiguous_inside_parent_is_rejected(self):
        for content, quote, qualifier in (("2024 summary. Output did not rise.", "Output did not rise.", "2024"),
                                          ("2024 output equals the 2024 baseline.",
                                           "2024 output equals the 2024 baseline.", "2024")):
            with self.subTest(content=content):
                target, material, context = self.fixture(content)
                client = ScriptClient(("atoms", atoms((quote, quote, (qualifier,)))))
                psi = s.StagedDecomposer(client)
                with self.assertRaises(s.StagedSemanticError):
                    psi.decompose(target, material, context)
                self.assertEqual("failed", psi.history[-1]["status"])
                self.assertEqual(1, len(client.calls))

    def test_distinct_atomic_statements_sharing_quote_get_stable_distinct_owned_ids(self):
        quote = "Output fell and delays rose."
        target, material, context = self.fixture(quote)
        items = (("Output fell.", quote, ()), ("Delays rose.", quote, ()))
        outputs = []
        for ordered in (items, tuple(reversed(items))):
            client = ScriptClient(("atoms", atoms(*ordered)), ("lineage", NO_LINEAGE), ("critic", ACCEPT))
            outputs.append(s.StagedDecomposer(client).decompose(target, material, context))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(2, len({f.id for f in outputs[0].fragments}))
        self.assertTrue(all(f.id.startswith("a:atom:") for f in outputs[0].fragments))
        _, other, other_context = self.fixture(quote, owner="b")
        client = ScriptClient(("atoms", atoms(*items)), ("lineage", NO_LINEAGE), ("critic", ACCEPT))
        other_result = s.StagedDecomposer(client).decompose(target, other, other_context)
        self.assertTrue({f.id for f in outputs[0].fragments}.isdisjoint({f.id for f in other_result.fragments}))

    def test_v4_material_probe_ledger_requires_one_grounded_slot_per_probe(self):
        plan = {"schema_version": "decision-probe-v4",
                "probes": [{"id": "probe-a"}, {"id": "probe-b"}]}
        raw = {"atoms": [
            {"statement": "A", "quote": "A", "qualifier_quotes": []},
            {"statement": "B", "quote": "B", "qualifier_quotes": []},
        ], "probe_checks": [
            {"probe_id": "probe-a", "status": "addressed",
             "finding_indexes": [0], "origin_used": False, "rationale": "A."},
            {"probe_id": "probe-b", "status": "ambiguous",
             "finding_indexes": [1], "origin_used": False, "rationale": "B."},
        ], "notes": ""}
        s.StagedDecomposer._check_probe_ledger("atoms", raw, plan)

        malformed = []
        missing = deepcopy(raw)
        missing["probe_checks"].pop()
        malformed.append(missing)
        unknown = deepcopy(raw)
        unknown["probe_checks"][1]["probe_id"] = "unknown"
        malformed.append(unknown)
        unused = deepcopy(raw)
        unused["probe_checks"][1].update(
            status="absent", finding_indexes=[], origin_used=False)
        malformed.append(unused)
        bad_index = deepcopy(raw)
        bad_index["probe_checks"][1]["finding_indexes"] = [2]
        malformed.append(bad_index)
        atoms_origin = deepcopy(raw)
        atoms_origin["probe_checks"][0]["origin_used"] = True
        malformed.append(atoms_origin)
        for candidate in malformed:
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError):
                    s.StagedDecomposer._check_probe_ledger("atoms", candidate, plan)

    def test_v4_lineage_probe_ledger_can_use_only_a_present_origin(self):
        plan = {"schema_version": "decision-probe-v4",
                "probes": [{"id": "lineage"}]}
        origin = {"kind": "original_record", "quote": "Original record.",
                  "rationale": "Producing role is explicit."}
        raw = {"citations": [], "origin": origin, "probe_checks": [{
            "probe_id": "lineage", "status": "addressed", "finding_indexes": [],
            "origin_used": True, "rationale": "The origin finding addresses lineage.",
        }], "notes": ""}
        s.StagedDecomposer._check_probe_ledger("lineage", raw, plan)
        raw["origin"] = None
        with self.assertRaisesRegex(ValueError, "absent origin"):
            s.StagedDecomposer._check_probe_ledger("lineage", raw, plan)

    def test_v4_runtime_validator_reconstructs_canonical_plan_and_projections(self):
        target, plan, views = self.v4_fixture()
        s._validate_target_plan(views, target)
        critic = views["critic"]
        claim, = critic["claims"]
        self.assertEqual(claim["anchor"]["quote"], claim["statement"])
        self.assertEqual(e.PLAN_NOTES, critic["notes"])
        self.assertEqual({"predicate_core", "claim_composition", "source_lineage"},
                         {item["kind"] for item in critic["probes"]})
        self.assertTrue(all(set(item) == s._PLAN_COVERAGE_FIELDS_V4
                            for item in critic["coverage_ledger"]))
        self.assertEqual(plan.sha256, critic["plan_sha256"])

    def test_v4_runtime_validator_rejects_rehashed_model_owned_contract_text(self):
        target, _, original = self.v4_fixture()

        def statement(critic):
            critic["claims"][0]["statement"] = "model-authored substitute"

        def question(critic):
            critic["probes"][0]["question"] = "A rewritten question?"

        def impact(critic):
            critic["probes"][0]["decision_impact"] = "A rewritten consequence."

        def legacy_core(critic):
            probe = next(item for item in critic["probes"]
                         if item["kind"] == "predicate_core")
            probe["kind"] = "semantic_core"

        for label, mutate in (("statement", statement), ("question", question),
                              ("impact", impact), ("legacy semantic core", legacy_core)):
            with self.subTest(label=label):
                views = deepcopy(original)
                mutate(views["critic"])
                self.reproject_v4(views)
                with self.assertRaises(ValueError):
                    s._validate_target_plan(views, target)

    def test_v4_runtime_validator_recomputes_seven_field_coverage_ledger(self):
        target, _, original = self.v4_fixture()

        def missing_segment(critic):
            critic["coverage_ledger"].pop()

        def changed_anchor(critic):
            critic["coverage_ledger"][0]["anchor"]["quote"] = "forged"

        def missing_dimension(critic):
            entry = next(item for item in critic["coverage_ledger"]
                         if item["status"] == "covered_by_dimension")
            entry["status"] = "context_only"
            entry["dimension_ids"] = []

        def eighth_field(critic):
            critic["coverage_ledger"][0]["model_comment"] = "not contractual"

        for label, mutate in (("missing segment", missing_segment),
                              ("changed segment anchor", changed_anchor),
                              ("missing exact dimension", missing_dimension),
                              ("extra ledger field", eighth_field)):
            with self.subTest(label=label):
                views = deepcopy(original)
                mutate(views["critic"])
                self.reproject_v4(views)
                with self.assertRaises(ValueError):
                    s._validate_target_plan(views, target)

    def test_v4_unresolved_probe_requires_exactly_one_task_or_probe_stop(self):
        _, material, _ = self.fixture("Source passage.")
        allowed = {material.version_id: asdict(material)}
        plan = {
            "schema_version": "decision-probe-v4", "logic": "single",
            "claims": [{"id": "claim"}],
            "probes": [{"id": "probe", "claim_id": "claim", "kind": "predicate_core",
                        "gate": "always", "match_policy": "semantic_constraint",
                        "question": "Find the decisive record.",
                        "decision_impact": "It decides the probe."}],
        }
        base = {
            "basis": [{"version_id": material.version_id, "quote": material.content}],
            "probe_results": [{"probe_id": "probe", "status": "unresolved",
                               "basis_indexes": [0], "rationale": "The visible lead is incomplete.",
                               "referent_relation": "not_applicable"}],
            "rationale": "The probe remains unresolved.", "resolutions": [],
        }
        task = {"probe_id": "probe", "question": "Find the decisive record.",
                "action": "search", "locator": "publisher record",
                "basis": base["basis"], "decision_impact": "It decides the probe."}
        stop = {"probe_id": "probe", "reason": "no_source_lead",
                "rationale": "No concrete locator is visible."}

        with_task = {**deepcopy(base), "gaps": [task], "no_leads": []}
        assembled = s.StagedVerifier._assemble_layer(
            p.Target("target", "Target.", "2026-09-04T20:00:00Z"),
            "evidence", with_task, allowed, {}, [], plan)
        self.assertEqual(("probe",), tuple(item.probe_id for item in assembled["gaps"]))
        self.assertEqual((), assembled["probe_stops"])

        with_stop = {**deepcopy(base), "gaps": [], "no_leads": [stop]}
        assembled = s.StagedVerifier._assemble_layer(
            p.Target("target", "Target.", "2026-09-04T20:00:00Z"),
            "evidence", with_stop, allowed, {}, [], plan)
        self.assertEqual((), assembled["gaps"])
        self.assertEqual(("probe",), tuple(item.probe_id
                                           for item in assembled["probe_stops"]))
        self.assertIsInstance(assembled["probe_stops"][0], p.ProbeStop)

        for label, gaps, stops in (("neither", [], []),
                                   ("both", [task], [stop]),
                                   ("duplicate stops", [], [stop, stop])):
            with self.subTest(label=label):
                raw = {**deepcopy(base), "gaps": deepcopy(gaps),
                       "no_leads": deepcopy(stops)}
                with self.assertRaises(ValueError):
                    s.StagedVerifier._assemble_layer(
                        p.Target("target", "Target.", "2026-09-04T20:00:00Z"),
                        "evidence", raw, allowed, {}, [], plan)

    def test_v4_runtime_rejects_rehashed_omitted_boolean_branch(self):
        target = p.Target("target", "Alpha launched Orion and Beta cancelled Nova.",
            "2026-09-04T20:00:00Z", source_version_id="a",
            assessment_mode="evidence", evidence_scope=("a",))
        target_payload = asdict(target)
        target_payload["evidence_scope"] = list(target_payload["evidence_scope"])
        signature = e._digest(target_payload)
        quote = "Alpha launched Orion"
        anchor = e.TextAnchor(0, len(quote), quote)
        provisional = (("subject", "Alpha"), ("predicate", "launched"),
                       ("entity_identity", "Orion"))
        predicate_start = target.text.index("launched")
        claim_id = target.id + ":claim:" + e._digest([
            signature, 0, len(quote), "main", predicate_start,
            predicate_start + len("launched")])[:20]
        dimensions = []
        for kind, text in provisional:
            start = target.text.index(text)
            dimension_id = target.id + ":dimension:" + e._digest([
                signature, claim_id, kind, start, start + len(text)])[:20]
            dimensions.append(e.ClaimDimension(
                dimension_id, kind, e.TextAnchor(start, start + len(text), text)))
        dimensions.sort(key=lambda item: (item.anchor.start, item.anchor.end,
                                           item.kind, item.id))
        claim = e.TargetClaim(claim_id, quote, anchor, "main", tuple(dimensions))
        probes = []
        owned = {item.id: item for item in claim.dimensions}
        known = {claim.id: claim}
        for kind, dimension_ids in e._required_probe_bindings(
                claim, target.assessment_mode, "single"):
            dimension_ids = tuple(dimension_ids)
            routes, gate = e._routes_and_gate(kind)
            probes.append(e.DecisionProbe(
                target.id + ":probe:" + e._digest(
                    [signature, claim.id, kind, *dimension_ids])[:20],
                claim.id, kind, dimension_ids,
                e._canonical_probe_question(
                    claim, kind, dimension_ids, owned, known),
                e._canonical_decision_impact(kind), e._match_policy(kind), routes, gate))
        probes.sort(key=lambda item: (item.claim_id, item.kind,
                                      item.dimension_ids, item.id))
        ledger = []
        for segment in e._target_segments(target_payload, signature, (claim,)):
            overlapping = tuple(sorted(item.id for item in dimensions
                if max(segment.anchor.start, item.anchor.start) <
                   min(segment.anchor.end, item.anchor.end)))
            status = ("covered_by_dimension" if overlapping else
                      "logic_connector" if segment.cue_kind == "logic_connector"
                      else "context_only")
            ledger.append(e.CoverageLedgerEntry(segment.id, segment.claim_id,
                segment.anchor, segment.cue_kind, segment.high_signal, status, overlapping))
        ledger.sort(key=lambda item: (item.claim_id, item.anchor.start,
                                      item.anchor.end, item.cue_kind, item.segment_id))
        draft = e.TargetPlan(signature, "single", (claim,), tuple(probes),
                             tuple(ledger), e.PLAN_NOTES, "")
        plan = e.TargetPlan(signature, "single", draft.claims, draft.probes,
            draft.coverage_ledger, draft.notes, e._digest(e._plan_payload(draft)))
        views = {stage: e.project_plan(plan, stage).to_payload()
                 for stage in ("atoms", "lineage", "critic", "evidence", "world")}
        with self.assertRaisesRegex(ValueError, "Boolean connector"):
            s._validate_target_plan(views, target)

    def test_v4_followup_task_fields_and_locators_are_frozen_or_grounded(self):
        target, material, _ = self.fixture("Lead supports output record.")
        allowed = {material.version_id: asdict(material)}
        question = "Does the output record resolve this probe?"
        impact = "Without it, the output claim remains unresolved."
        plan = {"schema_version": "decision-probe-v4", "logic": "single",
            "claims": [{"id": "claim"}], "probes": [{
                "id": "probe", "claim_id": "claim", "kind": "predicate_core",
                "gate": "always", "match_policy": "semantic_constraint",
                "question": question, "decision_impact": impact}]}
        base = {"basis": [{"version_id": material.version_id,
                            "quote": material.content}],
            "probe_results": [{"probe_id": "probe", "status": "unresolved",
                "basis_indexes": [0], "rationale": "More evidence is needed.",
                "referent_relation": "not_applicable"}],
            "rationale": "Unresolved.", "no_leads": [], "resolutions": []}
        task = {"probe_id": "probe", "question": question, "action": "search",
                "locator": "output record", "basis": base["basis"],
                "decision_impact": impact}
        accepted = s.StagedVerifier._assemble_layer(
            target, "evidence", {**deepcopy(base), "gaps": [task]},
            allowed, {}, [], plan)
        self.assertEqual(question, accepted["gaps"][0].question)
        self.assertEqual(impact, accepted["gaps"][0].decision_impact)
        for field, value in (("question", "Banana aliens?"),
                             ("decision_impact", "Bananas decide everything."),
                             ("locator", "banana aliens")):
            with self.subTest(field=field):
                bad = deepcopy(task)
                bad[field] = value
                with self.assertRaises(ValueError):
                    s.StagedVerifier._assemble_layer(
                        target, "evidence", {**deepcopy(base), "gaps": [bad]},
                        allowed, {}, [], plan)

        cited = p.MaterialVersion("cited", "https://example.org/cited",
            "The record cites https://upstream.example/source for the output record.",
            "2026-09-04T20:00:00Z", "2026-09-03T00:00:00Z",
            "2026-09-03T00:00:00Z", "Archive.")
        other = p.MaterialVersion("other", "https://example.org/other",
            "Another visible record.", "2026-09-04T20:00:00Z",
            "2026-09-03T00:00:00Z", "2026-09-03T00:00:00Z", "Archive.")
        task_allowed = {item.version_id: asdict(item)
                        for item in (cited, other)}
        task_base = deepcopy(base)
        task_base["basis"] = [{"version_id": cited.version_id,
                                "quote": cited.content}]
        task_base["probe_results"][0]["basis_indexes"] = [0]

        fetch = {**deepcopy(task), "action": "fetch",
                 "locator": "https://upstream.example/source",
                 "basis": deepcopy(task_base["basis"])}
        accepted_fetch = s.StagedVerifier._assemble_layer(
            target, "evidence", {**deepcopy(task_base), "gaps": [fetch]},
            task_allowed, {}, [], plan)
        self.assertEqual(fetch["locator"], accepted_fetch["gaps"][0].locator)
        bad_fetch = {**deepcopy(fetch),
                     "locator": "https://banana.example/unseen"}
        with self.assertRaisesRegex(ValueError, "declared by its basis"):
            s.StagedVerifier._assemble_layer(
                target, "evidence", {**deepcopy(task_base), "gaps": [bad_fetch]},
                task_allowed, {}, [], plan)

        reanalyse = {**deepcopy(task), "action": "reanalyse",
                     "locator": cited.version_id,
                     "basis": deepcopy(task_base["basis"])}
        accepted_reanalysis = s.StagedVerifier._assemble_layer(
            target, "evidence", {**deepcopy(task_base), "gaps": [reanalyse]},
            task_allowed, {}, [], plan)
        self.assertEqual(cited.version_id,
                         accepted_reanalysis["gaps"][0].locator)
        bad_reanalysis = {**deepcopy(reanalyse), "locator": other.version_id}
        with self.assertRaisesRegex(ValueError, "visible basis version"):
            s.StagedVerifier._assemble_layer(
                target, "evidence",
                {**deepcopy(task_base), "gaps": [bad_reanalysis]},
                task_allowed, {}, [], plan)

    def test_v4_loop_receipt_uses_exact_issued_tasks_and_prior_probe_result(self):
        target, material, context = self.fixture("Returned output record.")
        span = p.Span(material.version_id, 0, len(material.content), material.content)
        provenance = p.Gap(
            "fetch-upstream", "Fetch the cited upstream record.",
            stage="provenance", dimension="provenance", target_id=target.id,
            basis=(span,), decision_impact="This determines the lineage path.",
            action="fetch", locator="https://upstream.example/original")
        probe_task = p.Gap(
            "probe-task", "Resolve the frozen output probe.",
            stage="verification", dimension="world", target_id=target.id,
            basis=(span,), decision_impact="This changes the world decision.",
            action="search", locator="output record", probe_id="probe-7")
        old_task = p.Gap(
            "old-task", "An older unrelated task.", stage="verification",
            dimension="evidence", target_id=target.id, basis=(span,),
            decision_impact="Old task.", action="search", locator="old lead",
            probe_id="old-probe")
        assessment = p.ProbeAssessment(
            "probe-7", "claim-7", "world", "unresolved", (span,),
            "The prior world pass could not resolve this probe.")
        context.update(
            gaps=[asdict(provenance), asdict(probe_task), asdict(old_task)],
            gap_registry=[asdict(provenance), asdict(probe_task), asdict(old_task)],
            verification_history=[{"round": 1, **asdict(p.VerificationResult(
                verdict="unresolved", rationale="Prior aggregate unresolved.",
                world_verdict="unresolved",
                world_rationale="The world layer needs another source.",
                world_probe_results=(assessment,))) }],
            current_return={
                "version_id": material.version_id,
                "trigger_task_ids": [provenance.id, probe_task.id],
                "trigger_probe_ids": [assessment.probe_id],
                "issued_tasks": [asdict(provenance), asdict(probe_task)],
            },
        )
        receipt = s._feedback(context, target, material)
        self.assertEqual("loop-receipt-v1", receipt["schema_version"])
        self.assertEqual([provenance.id, probe_task.id],
                         [item["id"] for item in receipt["tasks"]])
        self.assertEqual("https://upstream.example/original",
                         receipt["tasks"][0]["locator"])
        self.assertTrue(all(item["active_at_decomposition"]
                            for item in receipt["tasks"]))
        self.assertNotIn(old_task.id,
                         {item["id"] for item in receipt["tasks"]})
        result, = receipt["probe_results"]
        self.assertEqual({"probe_id": "probe-7", "layer": "world",
                          "status": "unresolved",
                          "aggregate_verdict": "unresolved",
                          "task_ids": [probe_task.id]},
                         {key: result[key] for key in (
                             "probe_id", "layer", "status",
                             "aggregate_verdict", "task_ids")})
        self.assertEqual([asdict(span)], result["basis"])
        self.assertEqual(assessment.rationale, result["rationale"])
        self.assertEqual([], s._project_loop_receipts((receipt,), "evidence"))
        world_projection, = s._project_loop_receipts((receipt,), "world")
        self.assertEqual([probe_task.id],
                         [item["id"] for item in world_projection["tasks"]])
        self.assertNotIn(provenance.id,
                         world_projection["return_attribution"]
                         ["trigger_task_ids"])

        # A later return from the same round remains attributable from the
        # frozen issued snapshot even if the first return closed the live gap.
        inactive = deepcopy(context)
        inactive["gaps"] = [asdict(old_task)]
        later_receipt = s._feedback(inactive, target, material)
        self.assertFalse(any(item["active_at_decomposition"]
                             for item in later_receipt["tasks"]))

        # A dependency revisit may still share a version or basis with an old
        # task, but without current_return it is not a provider-hit receipt.
        revisit = deepcopy(context)
        revisit.pop("current_return")
        self.assertEqual({"schema_version": "loop-receipt-v1",
                          "target_id": target.id,
                          "return_attribution": None,
                          "tasks": [], "probe_results": []},
                         s._feedback(revisit, target, material))

    def test_v4_loop_receipt_tampering_fails_before_any_model_stage(self):
        target, _, views = self.v4_fixture()
        material = p.MaterialVersion(
            "a", "https://example.org/a", target.text,
            "2026-09-04T20:00:00Z", "2026-09-03T00:00:00Z",
            "2026-09-03T00:00:00Z", "Archive.")
        task = p.Gap("origin:target", "Find the original record.")
        base = {"target": asdict(target), "materials": [asdict(material)],
                "analyses": {}, "fragments": [], "relations": [], "origins": [],
                "gaps": [asdict(task)], "gap_registry": [asdict(task)],
                "verification_history": [], "usage": {"rounds": 1},
                "current_return": {"version_id": material.version_id,
                    "trigger_task_ids": [task.id], "trigger_probe_ids": [],
                    "issued_tasks": [asdict(task)]}}

        bad_contexts = []
        missing_snapshot = deepcopy(base)
        missing_snapshot["current_return"].pop("issued_tasks")
        bad_contexts.append(("snapshot", missing_snapshot))
        unknown = deepcopy(base)
        unknown["current_return"]["trigger_task_ids"] = ["not-issued"]
        unknown["current_return"]["issued_tasks"][0]["id"] = "not-issued"
        bad_contexts.append(("gap registry", unknown))
        wrong_version = deepcopy(base)
        wrong_version["current_return"]["version_id"] = "other"
        bad_contexts.append(("wrong target or return", wrong_version))
        wrong_probe = deepcopy(base)
        wrong_probe["current_return"]["trigger_probe_ids"] = ["invented-probe"]
        bad_contexts.append(("probe IDs", wrong_probe))
        wrong_identity = deepcopy(base)
        wrong_identity["current_return"]["issued_tasks"][0]["locator"] = "other"
        bad_contexts.append(("registered identity", wrong_identity))

        class NeverClient:
            def __init__(self): self.calls = 0
            def call(self, *args, **kwargs):
                self.calls += 1
                raise AssertionError("receipt validation must precede model stages")

        for message, context in bad_contexts:
            with self.subTest(message=message):
                client = NeverClient()
                with self.assertRaisesRegex(s.StagedSemanticError, message):
                    s.StagedDecomposer(client, target_plan=views).decompose(
                        target, material, context)
                self.assertEqual(0, client.calls)

    def test_v4_graph_owned_lineage_and_independence_cannot_be_rubber_stamped(self):
        target = p.Target("target", "Output rose.", "2026-09-04T20:00:00Z",
                          source_version_id="a", assessment_mode="world")
        materials = {
            key: asdict(p.MaterialVersion(key, "https://example.org/" + key,
                "Passage " + key + ".", "2026-09-04T20:00:00Z",
                "2026-09-03T00:00:00Z", "2026-09-03T00:00:00Z", "Archive."))
            for key in ("a", "b", "root", "root2")}

        def assemble(kind, versions, context):
            question = "Check " + kind + "."
            impact = "This graph check affects the decision."
            plan = {"schema_version": "decision-probe-v4", "logic": "single",
                "claims": [{"id": "claim"}], "probes": [{
                    "id": "probe", "claim_id": "claim", "kind": kind,
                    "gate": ("provenance" if kind == "source_lineage"
                             else "positive_world_only"),
                    "match_policy": "semantic_constraint", "question": question,
                    "decision_impact": impact}]}
            basis = [{"version_id": version, "quote": materials[version]["content"]}
                     for version in versions]
            raw = {"basis": basis, "probe_results": [{"probe_id": "probe",
                "status": "supported", "basis_indexes": list(range(len(basis))),
                "rationale": "The model claims the graph condition is met.",
                "referent_relation": "not_applicable"}], "rationale": "Claimed.",
                "gaps": [], "no_leads": [], "resolutions": []}
            return s.StagedVerifier._assemble_layer(
                target, "world", raw, materials, {}, [], plan, context)

        with self.assertRaisesRegex(ValueError, "confirmed terminal origin"):
            assemble("source_lineage", ("a",), {"relations": [], "origins": []})
        with self.assertRaisesRegex(ValueError, "confirmed terminal origin"):
            assemble("source_lineage", ("a",), {
                "relations": [],
                "origins": [{"target_id": "target", "version_id": "root"}],
            })
        self.assertTrue(assemble("source_lineage", ("a",), {
            "relations": [], "origins": [{"target_id": "target", "version_id": "a"}]
        })["probe_results"])
        direct = {"kind": "cites", "from_version": "a", "to_version": "root",
                  "status": "direct"}
        self.assertTrue(assemble("source_lineage", ("a",), {
            "relations": [direct],
            "origins": [{"target_id": "target", "version_id": "root"}]
        })["probe_results"])

        same_root = [direct, {"kind": "cites", "from_version": "b",
                             "to_version": "root", "status": "direct"}]
        with self.assertRaisesRegex(ValueError, "graph-independent"):
            assemble("source_independence", ("a",), {
                "relations": [],
                "origins": [{"target_id": "target", "version_id": "a"}]})
        with self.assertRaisesRegex(ValueError, "graph-independent"):
            assemble("source_independence", ("a", "b"), {
                "relations": same_root,
                "origins": [{"target_id": "target", "version_id": "root"}]})
        independent = assemble("source_independence", ("a", "b"), {
            "relations": [direct, {"kind": "cites", "from_version": "b",
                "to_version": "root2", "status": "direct"}],
            "origins": [{"target_id": "target", "version_id": "root"},
                        {"target_id": "target", "version_id": "root2"}]})
        self.assertTrue(independent["probe_results"])


if __name__ == "__main__":
    unittest.main()
