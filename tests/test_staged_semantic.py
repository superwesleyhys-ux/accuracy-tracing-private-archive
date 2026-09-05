"""Deterministic stage-contract regressions; every model response is synthetic."""
from copy import deepcopy
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
s = importlib.import_module("staged_semantic")
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


class StagedSemanticTests(unittest.TestCase):
    def fixture(self, content="Output did not rise above the 2024 baseline.", owner="a"):
        material = p.MaterialVersion(owner, "https://example.org/" + owner, content,
            "2026-09-04T20:00:00Z", "2026-09-03T00:00:00Z", "2026-09-03T00:00:00Z", "Synthetic archive.")
        target = p.Target("target", "Output rose above the 2024 baseline.", "2026-09-04T20:00:00Z",
                          source_version_id=owner, assessment_mode="evidence", evidence_scope=(owner,))
        context = {"target": asdict(target), "materials": [asdict(material)], "analyses": {},
                   "gaps": [], "gap_registry": [], "relations": [], "origins": [], "usage": {"rounds": 1}}
        return target, material, context

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


if __name__ == "__main__":
    unittest.main()
