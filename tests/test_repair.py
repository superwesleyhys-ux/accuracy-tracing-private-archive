"""Regression cases for the observed failure mechanisms; no real-news accuracy claims."""
from dataclasses import replace
import unittest

from newsverify.provenance import (Analysis, Fragment, Gap, MaterialVersion, ReplayTraceProvider, Relation,
    Span, Target, TraceConfig, VerificationResult, run_provenance)
from newsverify.decisions import present_decision, round_decisions
from newsverify.retrieval import SnapshotSearchProvider


def document(id="notice", content="The bridge remains closed."):
    return MaterialVersion(id, "https://example.org/" + id, content,
        "2026-09-04T20:00:00Z", "2026-09-04T10:00:00Z", "2026-09-04T10:00:00Z", "Fixture snapshot")


def span(doc): return Span(doc.version_id, 0, len(doc.content), doc.content)


class Psi:
    def __init__(self, fn): self.fn = fn
    def decompose(self, *args): return self.fn(*args)


class Verify:
    def __init__(self, fn): self.fn = fn
    def verify(self, *args): return self.fn(*args)


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.doc = document()
        self.target = Target("claim", "The bridge has reopened.", "2026-09-04T20:00:00Z",
                             assessment_mode="evidence", evidence_scope=("notice",))

    def run_check(self, check, mode="evidence"):
        return run_provenance(replace(self.target, assessment_mode=mode), ReplayTraceProvider(((self.doc,),)),
                              verifier=Verify(lambda *_: check))

    def gap(self, dimension="world", blocking=True, **changes):
        return replace(Gap("g", "Check actual bridge condition", "verification", dimension,
            blocking, "claim", (span(self.doc),), "Could change the actual-world claim", "search", "bridge actual condition"), **changes)

    def check(self, gaps=(), evidence="contradicted", world="unresolved"):
        return VerificationResult("unresolved", (span(self.doc),), "Notice contradicts reopening", gaps,
            evidence_verdict=evidence, world_verdict=world,
            world_basis=(span(self.doc),) if world != "unresolved" else (),
            world_rationale="Fixture world annotation" if world != "unresolved" else "No world authentication")

    def test_world_gap_does_not_erase_contradicting_evidence(self):
        report = self.run_check(self.check((self.gap(),)))
        self.assertEqual("contradicted", report["assessments"]["evidence"]["raw_verdict"])
        self.assertEqual("false", present_decision(report)["decision"])
        self.assertEqual("unresolved", report["fact_status"])

    def test_conflict_is_preserved_alongside_unknown_world_state(self):
        report = self.run_check(self.check((self.gap(),), evidence="conflicting"))
        self.assertEqual("disputed", present_decision(report)["decision"])
        self.assertEqual("unresolved", report["assessments"]["world"]["decision"])

    def test_critical_world_gap_still_blocks_a_positive_world_claim(self):
        report = self.run_check(self.check((self.gap(),), world="supported"), mode="world")
        self.assertEqual("unverifiable", present_decision(report)["decision"])
        self.assertEqual("supported", report["assessments"]["world"]["raw_verdict"])

    def test_nonblocking_gap_does_not_blanket_override(self):
        report = self.run_check(self.check((self.gap(blocking=False),), world="supported"), mode="world")
        self.assertEqual("true", present_decision(report)["decision"])

    def test_critical_evidence_gap_blocks_only_evidence_decision(self):
        report = self.run_check(self.check((self.gap(dimension="evidence"),)))
        self.assertEqual("unverifiable", present_decision(report)["decision"])
        self.assertEqual("contradicted", report["assessments"]["evidence"]["raw_verdict"])

    def test_unjustified_explicit_blocking_gap_is_rejected(self):
        report = self.run_check(self.check((self.gap(basis=()),)))
        self.assertEqual("verifier_error", report["stop_reason"])

    def test_gap_for_another_target_is_rejected(self):
        report = self.run_check(self.check((self.gap(target_id="another"),)))
        self.assertEqual("verifier_error", report["stop_reason"])

    def test_claim_cannot_be_supported_from_out_of_scope_document(self):
        target = replace(self.target, evidence_scope=("different",))
        report = run_provenance(target, ReplayTraceProvider(((self.doc,),)), verifier=Verify(lambda *_: self.check()))
        self.assertEqual("verifier_error", report["stop_reason"])

    def test_invented_world_basis_cannot_pass(self):
        check = replace(self.check(world="supported"), world_basis=(Span("notice", 0, 8, "invented"),))
        self.assertEqual("verifier_error", self.run_check(check)["stop_reason"])

    def test_every_round_and_final_share_the_same_decision_mapping(self):
        report = run_provenance(self.target, ReplayTraceProvider(((self.doc,),) * 4),
            verifier=Verify(lambda *_: self.check()), config=TraceConfig(max_rounds=5))
        self.assertTrue(all(r["decision"] == present_decision(report)["decision"] for r in round_decisions(report)))

    def test_cosmetic_fragment_changes_do_not_extend_loop(self):
        count = []
        def fn(target, material, context):
            count.append(1)
            return Analysis(fragments=(Fragment("generated-"+str(len(count)), "paraphrase-"+str(len(count)),
                span(material), target.id, ("wording-"+str(len(count)),)),))
        report = run_provenance(self.target, ReplayTraceProvider(((self.doc,),) * 8), Psi(fn),
                               config=TraceConfig(max_rounds=8))
        self.assertEqual(2, report["usage"]["decomposition_calls"])

    def test_self_revisit_does_not_invent_a_news_gap(self):
        report = run_provenance(self.target, ReplayTraceProvider(((self.doc,),)),
            Psi(lambda t, m, c: Analysis(revisit_versions=(m.version_id,))))
        self.assertFalse(any(g["id"].startswith("revisit:") for g in report["gaps"]))
        self.assertEqual(1, report["usage"]["decomposition_calls"])

    def test_new_upstream_reanalyses_old_declared_edge_without_model_request(self):
        article = document("article", "According to https://example.org/notice")
        seen = []
        def fn(t, m, c):
            seen.append(m.version_id)
            if m.version_id == "article":
                available = any(x["version_id"] == "notice" for x in c["materials"])
                return Analysis(relations=(Relation("citation", "article", "notice" if available else None,
                    "cites", "direct" if available else "declared", (span(m),), "Visible citation", self.doc.url),))
            return Analysis()
        report = run_provenance(self.target, ReplayTraceProvider(((article, self.doc),)), Psi(fn))
        self.assertEqual(["article", "notice", "article"], seen)
        self.assertEqual("direct", report["relations"][0]["status"])
        self.assertTrue(any(o["action"] == "upstream_arrival_reanalysis" for o in report["operations"]))

    def test_nested_fragments_must_connect_to_current_target(self):
        for parent in ["missing", "f"]:
            report = run_provenance(self.target, ReplayTraceProvider(((self.doc,),)),
                Psi(lambda t, m, c: Analysis(fragments=(Fragment("f", "x", span(m), parent),))))
            self.assertEqual("decomposer_error", report["stop_reason"])
        valid = run_provenance(self.target, ReplayTraceProvider(((self.doc,),)),
            Psi(lambda t, m, c: Analysis(fragments=(Fragment("f", "x", span(m), t.id),
                                                    Fragment("child", "y", span(m), "f")))))
        self.assertEqual([], valid["errors"])

    def test_future_versions_do_not_enter_stateful_semantic_plugin(self):
        seen = []
        future = replace(self.doc, version_id="future", available_at="2026-09-05T00:00:00Z")
        report = run_provenance(self.target, ReplayTraceProvider(((future, self.doc),)),
            Psi(lambda t, m, c: seen.append(m.version_id) or Analysis()))
        self.assertEqual(["notice"], seen)
        self.assertEqual(2, report["usage"]["decomposition_calls"])

    def test_fetch_task_retrieves_its_upstream_not_same_seed_again(self):
        seed = document("article", "According to https://example.org/notice")
        provider = SnapshotSearchProvider([seed, self.doc], ["article"])
        self.assertEqual([seed], list(provider.search(self.target, (), 1, 5)))
        tasks = (Gap("upstream", "Read the cited notice", action="fetch", locator=self.doc.url),)
        self.assertEqual([self.doc], list(provider.search(self.target, tasks, 2, 5)))
        self.assertEqual([], list(provider.search(self.target, tasks, 3, 5)))
        self.assertEqual("unavailable", provider.last_feedback[0]["status"])

    def test_query_changes_actual_search_results(self):
        library = document("library", "The library closes at nine.")
        provider = SnapshotSearchProvider([self.doc, library], [])
        provider.search(self.target, (), 1, 5)
        found = list(provider.search(self.target, (Gap("q", "library", locator="library"),), 2, 5))
        self.assertEqual([library], found)

    def test_exhausted_search_does_not_rerun_verifier_without_new_input(self):
        provider = SnapshotSearchProvider([self.doc], ["notice"])
        count = []
        report = run_provenance(self.target, provider,
            verifier=Verify(lambda *_: count.append(1) or self.check()), config=TraceConfig(max_rounds=8))
        self.assertEqual(1, len(count))
        self.assertEqual("provider_exhausted", report["stop_reason"])

    def test_discovered_upstreams_pass_each_return_through_psi(self):
        a = document("a", "https://example.org/b")
        b = document("b", "https://example.org/notice")
        provider = SnapshotSearchProvider([a, b, self.doc], ["a"])
        seen = []
        def fn(t, m, c):
            seen.append(m.version_id)
            locator = {"a": b.url, "b": self.doc.url}.get(m.version_id)
            return Analysis(gaps=(Gap("next:"+m.version_id, "Read upstream", action="fetch", locator=locator),) if locator else ())
        report = run_provenance(self.target, provider, Psi(fn), config=TraceConfig(max_rounds=5))
        self.assertEqual("a", seen[0])
        self.assertEqual({"a", "b", "notice"}, set(seen))
        self.assertEqual(3, report["usage"]["decomposition_calls"])


if __name__ == "__main__": unittest.main()
