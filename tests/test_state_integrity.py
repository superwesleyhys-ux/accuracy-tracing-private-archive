"""Offline adversarial state tests; these are not model-accuracy measurements."""

from copy import deepcopy
from dataclasses import asdict, replace
import unittest

from newsverify.provenance import (
    Analysis, Fragment, Gap, MaterialVersion, OriginFinding, Relation, Resolution,
    ReplayTraceProvider, Span, Target, TraceConfig, VerificationResult,
    canonical_checkpoint_json, checkpoint_sha256, run_provenance,
)


TARGET = Target("bridge", "What does the bridge notice establish?",
                "2026-09-04T20:00:00Z", "a", assessment_mode="evidence")


def material(identifier):
    return MaterialVersion(identifier, "https://example.org/" + identifier,
        "Notice " + identifier + " records the bridge status.",
        "2026-09-04T12:00:00Z", "2026-09-04T10:00:00Z",
        "2026-09-04T10:00:00Z", "Synthetic exact-version archive fixture")


def span(value):
    return Span(value.version_id, 0, len(value.content), value.content)


def fragment(value, identifier=None):
    return Fragment(identifier or value.version_id + ":notice", value.content,
                    span(value), TARGET.id)


def relation(value, identifier="shared"):
    return Relation(identifier, value.version_id, value.version_id, "supports", "direct",
                    (span(value),), "The notice supports its own recorded statement.")


def resolution(identifier, value=None):
    return Resolution(identifier, (span(value or material("a")),), "Source-backed closure.")


class ScriptedDecomposer:
    def __init__(self, function):
        self.function = function
        self.contexts = []

    def decompose(self, target, value, context):
        self.contexts.append(deepcopy(context))
        return self.function(value, context)


class ScriptedVerifier:
    def __init__(self, function=None):
        self.function = function
        self.contexts = []

    def verify(self, target, context):
        self.contexts.append(deepcopy(context))
        if self.function:
            return self.function(context)
        value = MaterialVersion(**context["materials"][0])
        return VerificationResult("supported", (span(value),), "Synthetic supported statement.")


def run(rounds, function, verifier=None, checkpoints=None):
    return run_provenance(TARGET, ReplayTraceProvider(rounds), ScriptedDecomposer(function),
        verifier, config=TraceConfig(max_rounds=len(rounds), experimental_force_rounds=True),
        checkpoint_callback=None if checkpoints is None else checkpoints.append)


class StateIntegrityTests(unittest.TestCase):
    def assert_invalid(self, report, stage, message):
        self.assertFalse(report["assessment_valid"])
        self.assertEqual(stage + "_error", report["stop_reason"])
        self.assertIn(message, report["errors"][-1]["message"])

    def test_conflicting_ids_within_one_response_reject_either_order(self):
        a = material("a")
        cases = {
            "fragments": (fragment(a, "same"), replace(fragment(a, "same"), text="Different claim")),
            "relations": (relation(a), replace(relation(a), kind="contradicts")),
            "gaps": (Gap("same", "Find upstream"), Gap("same", "Find a different upstream")),
        }
        for field, findings in cases.items():
            for ordered in (findings, tuple(reversed(findings))):
                with self.subTest(field=field, reversed=ordered != findings):
                    report = run([[a]], lambda value, context: Analysis(**{field: ordered}))
                    self.assert_invalid(report, "decomposer", "conflicting")
                    self.assertEqual({}, report["analyses"])
                    self.assertEqual([], report["analysis_history"])
                    self.assertEqual([], report["eligible_version_ids"])
                    rejected = [event for event in report["operations"] if event["action"] == "analysis_rejected"]
                    self.assertEqual(asdict(Analysis(**{field: ordered})), rejected[0]["analysis"])

    def test_conflicting_ids_across_owners_reject_either_material_order(self):
        a, b = material("a"), material("b")
        functions = {
            "fragment": lambda value: Analysis(fragments=(fragment(value, "shared"),)),
            "relation": lambda value: Analysis(relations=(relation(value),)),
            "gap": lambda value: Analysis(gaps=(Gap("shared", "Find source for " + value.version_id),)),
        }
        for kind, function in functions.items():
            for ordered in ([a, b], [b, a]):
                with self.subTest(kind=kind, first=ordered[0].version_id):
                    report = run([ordered, [ordered[0]]], lambda value, context: function(value))
                    self.assert_invalid(report, "decomposer", "conflicting")
                    self.assertEqual([ordered[0].version_id], list(report["analyses"]))
                    self.assertEqual(1, len(report["analysis_history"]))
                    self.assertEqual(1, report["usage"]["rounds"])

    def test_exact_shared_definitions_are_idempotent_and_cross_material_bases_are_valid(self):
        a, b = material("a"), material("b")
        shared = Analysis(fragments=(fragment(a, "shared"), fragment(a, "shared")),
                          gaps=(Gap("shared-task", "Identify the bridge source"),))
        report = run([[a, b], [a]], lambda value, context: shared)
        self.assertTrue(report["assessment_valid"])
        self.assertEqual([asdict(fragment(a, "shared"))], report["fragments"])
        self.assertEqual({"a", "b"}, set(report["analyses"]))
        cross = Relation("b:cites-a", "b", "a", "cites", "direct",
                         (span(a), span(b)), "B cites the exact A notice.")
        report = run([[a, b]], lambda value, context: Analysis(
            fragments=(fragment(value),), relations=(cross,) if value.version_id == "b" else ()))
        self.assertTrue(report["assessment_valid"])
        self.assertEqual([asdict(cross)], report["relations"])

    def test_same_owner_revisions_can_change_findings_and_improve_gap_description(self):
        a = material("a")
        old_gap = Gap("a:source", "Identify source")
        new_gap = replace(old_gap, question="Locate the producing record", basis=(span(a),),
                          decision_impact="More precise follow-up", blocking=False)
        first = Analysis(fragments=(fragment(a),), relations=(relation(a),), gaps=(old_gap,))
        second = Analysis(fragments=(replace(fragment(a), text="A more precise interpretation"),),
                          relations=(replace(relation(a), kind="contradicts"),), gaps=(new_gap,))
        report = run([[a], [a]], lambda value, context: first if context["usage"]["rounds"] == 1 else second)
        self.assertTrue(report["assessment_valid"])
        self.assertEqual(asdict(second), report["analyses"]["a"])
        self.assertIn(asdict(new_gap), report["gap_registry"])
        self.assertNotIn(asdict(old_gap), report["gap_registry"])

    def test_shared_gap_can_be_revised_after_other_current_owner_withdraws(self):
        a, b = material("a"), material("b")
        old = Gap("shared", "Find the notice source")
        updated = replace(old, question="Find the exact producing record", basis=(span(a),))
        def analyze(value, context):
            current_round = context["usage"]["rounds"]
            if current_round == 1:
                return Analysis(gaps=(old,))
            return Analysis() if value.version_id == "b" else Analysis(gaps=(updated,))
        report = run([[a, b], [b], [a]], analyze)
        self.assertTrue(report["assessment_valid"])
        self.assertIn(asdict(updated), report["gap_registry"])
        self.assertEqual(asdict(Analysis()), report["analyses"]["b"])
        conflicting = run([[a, b], [a]], analyze)
        self.assert_invalid(conflicting, "decomposer", "conflicting gap")
        self.assertIn(asdict(old), conflicting["gap_registry"])

    def test_gap_lifecycle_identity_cannot_change_after_closure(self):
        a = material("a")
        original = Gap("a:source", "Find the original")
        variations = (replace(original, stage="verification"),
                      replace(original, action="fetch", locator=a.url),
                      replace(original, locator="different search hint"))
        for changed in variations:
            with self.subTest(changed=changed):
                report = run([[a], [a]], lambda value, context: (
                    Analysis(gaps=(original,), resolutions=(resolution(original.id),))
                    if context["usage"]["rounds"] == 1 else Analysis(gaps=(changed,))))
                self.assert_invalid(report, "decomposer", "gap lifecycle identity")
                self.assertEqual([asdict(resolution(original.id))], report["resolutions"])
                self.assertIn(asdict(original), report["gap_registry"])

    def test_unknown_resolution_rejected_by_both_stages_even_if_name_looks_valid(self):
        a = material("a")
        for unknown in ("missing", "verification:missing", "origin:missing"):
            with self.subTest(stage="decomposer", unknown=unknown):
                report = run([[a]], lambda value, context: Analysis(resolutions=(resolution(unknown),)))
                self.assert_invalid(report, "decomposer", "unknown gap")
                self.assertEqual([], report["resolutions"])
            with self.subTest(stage="verifier", unknown=unknown):
                report = run([[a]], lambda value, context: Analysis(), ScriptedVerifier(
                    lambda context: VerificationResult(resolutions=(resolution(unknown),))))
                self.assert_invalid(report, "verifier", "unknown gap")
                self.assertEqual([], report["resolutions"])
                self.assertEqual([], report["verification_history"])

    def test_verifier_cannot_resolve_closed_or_withdrawn_provenance_gap(self):
        a = material("a")
        provenance_gap = Gap("a:lineage", "Find the lineage")
        for closed in (True, False):
            with self.subTest(closed=closed):
                def analyze(value, context):
                    if context["usage"]["rounds"] == 1:
                        return Analysis(gaps=(provenance_gap,),
                            resolutions=(resolution(provenance_gap.id),) if closed else ())
                    return Analysis()
                verifier = ScriptedVerifier(lambda context: VerificationResult(
                    resolutions=(resolution(provenance_gap.id),) if context["usage"]["rounds"] == 2 else ()))
                report = run([[a], [a]], analyze, verifier)
                self.assert_invalid(report, "verifier", "may not resolve provenance")
                self.assertEqual(1, len(report["verification_history"]))
                self.assertNotIn(provenance_gap.id, {gap["id"] for gap in report["gaps"]})

    def test_verifier_same_response_gap_resolution_requires_consistent_stage(self):
        a = material("a")
        verification_gap = Gap("verify:status", "Check bridge status", "verification")
        good = run([[a]], lambda value, context: Analysis(), ScriptedVerifier(lambda context:
            VerificationResult(gaps=(verification_gap,), resolutions=(resolution(verification_gap.id),))))
        self.assertTrue(good["assessment_valid"])
        self.assertIn(asdict(verification_gap), good["gap_registry"])
        self.assertNotIn(verification_gap.id, {gap["id"] for gap in good["gaps"]})
        for wrong in (replace(verification_gap, stage="provenance"),
                      replace(verification_gap, dimension="provenance")):
            bad = run([[a]], lambda value, context: Analysis(), ScriptedVerifier(lambda context:
                VerificationResult(gaps=(wrong,), resolutions=(resolution(wrong.id),))))
            self.assertFalse(bad["assessment_valid"])
            self.assertEqual([], bad["resolutions"])
            self.assertNotIn(wrong.id, {gap["id"] for gap in bad["gap_registry"]})

    def test_psi_can_register_and_resolve_a_consistent_gap_in_one_response(self):
        a = material("a")
        for stage in ("provenance", "verification"):
            task = Gap("a:task", "Inspect the notice", stage)
            report = run([[a]], lambda value, context: Analysis(gaps=(task,), resolutions=(resolution(task.id),)))
            self.assertTrue(report["assessment_valid"])
            self.assertIn(asdict(task), report["gap_registry"])
            self.assertNotIn(task.id, {gap["id"] for gap in report["gaps"]})

    def test_full_replacement_can_withdraw_prior_origin_and_resolution(self):
        a = material("a")
        first = Analysis(fragments=(fragment(a),),
            origins=(OriginFinding(TARGET.id, "a", (span(a),), "original_record", "Initial interpretation."),),
            resolutions=(resolution("origin:" + TARGET.id),))
        report = run([[a], [a]], lambda value, context: first if context["usage"]["rounds"] == 1 else Analysis(),
                     ScriptedVerifier())
        self.assertTrue(report["assessment_valid"])
        self.assertEqual(asdict(Analysis()), report["analyses"]["a"])
        self.assertEqual([], report["origins"])
        self.assertEqual([], report["fragments"])
        self.assertEqual([], report["resolutions"])
        self.assertEqual("partial", report["provenance_status"])
        self.assertIn("origin:" + TARGET.id, {gap["id"] for gap in report["gaps"]})
        self.assertEqual(asdict(first), report["analysis_history"][0]["analysis"])

    def test_invalid_full_replacement_rolls_back_all_accepted_semantic_state(self):
        a, b = material("a"), material("b")
        first = {"a": Analysis(fragments=(fragment(a),), gaps=(Gap("a:task", "Find original"),)),
                 "b": Analysis(fragments=(fragment(b),))}
        invalid = Analysis(fragments=(fragment(a, "b:notice"),), gaps=(Gap("never-accepted", "Candidate task"),))
        checkpoints = []
        report = run([[a, b], [a]], lambda value, context: first[value.version_id]
            if context["usage"]["rounds"] == 1 else invalid, ScriptedVerifier(), checkpoints)
        self.assert_invalid(report, "decomposer", "conflicting fragment")
        self.assertEqual("unresolved", report["fact_status"])
        self.assertEqual({key: asdict(value) for key, value in first.items()}, report["analyses"])
        self.assertEqual(2, len(report["analysis_history"]))
        self.assertEqual(1, len(checkpoints))
        prior = checkpoints[0].state
        self.assertEqual([asdict(item) for item in prior["fragments"].values()], report["fragments"])
        self.assertEqual([asdict(item) for item in prior["gaps"].values()], report["gaps"])
        self.assertEqual([asdict(item) for item in prior["gap_registry"].values()], report["gap_registry"])
        self.assertEqual(prior["verifications"], report["verification_history"])

    def test_invalid_verifier_batch_does_not_register_its_new_gap_or_history(self):
        a = material("a")
        new = Gap("never-accepted", "Candidate task", "verification")
        verifier = ScriptedVerifier(lambda context: VerificationResult(gaps=(new,), resolutions=(resolution("unknown"),)))
        report = run([[a]], lambda value, context: Analysis(), verifier)
        self.assert_invalid(report, "verifier", "unknown gap")
        self.assertEqual([], report["verification_history"])
        self.assertEqual([], report["resolutions"])
        self.assertNotIn(new.id, {gap["id"] for gap in report["gap_registry"]})
        rejected = [event for event in report["operations"] if event["action"] == "verification_rejected"]
        self.assertEqual(new.id, rejected[0]["verification"]["gaps"][0]["id"])

    def test_conflicting_verifier_gap_definitions_reject_both_orders(self):
        a = material("a")
        first = Gap("verify:status", "Check status", "verification")
        second = replace(first, question="Check a different status")
        for pair in ((first, second), (second, first)):
            report = run([[a]], lambda value, context: Analysis(),
                         ScriptedVerifier(lambda context: VerificationResult(gaps=pair)))
            self.assert_invalid(report, "verifier", "conflicting gap")
            self.assertEqual([], report["verification_history"])
            self.assertNotIn(first.id, {gap["id"] for gap in report["gap_registry"]})

    def test_historical_resolution_remains_valid_when_gap_owner_withdraws_descriptor(self):
        a, b = material("a"), material("b")
        task = Gap("a:upstream", "Locate upstream")
        def analyze(value, context):
            if value.version_id == "b":
                return Analysis(resolutions=(resolution(task.id, b),))
            return Analysis(gaps=(task,)) if context["usage"]["rounds"] == 1 else Analysis()
        decomposer = ScriptedDecomposer(analyze)
        report = run_provenance(TARGET, ReplayTraceProvider([[a, b], [a, b]]), decomposer,
            config=TraceConfig(max_rounds=2, experimental_force_rounds=True))
        self.assertTrue(report["assessment_valid"])
        self.assertEqual([asdict(resolution(task.id, b))], report["resolutions"])
        self.assertIn(asdict(task), report["gap_registry"])
        self.assertIn(asdict(task), decomposer.contexts[-1]["gap_registry"])
        self.assertNotIn(task.id, {gap["id"] for gap in decomposer.contexts[-1]["gaps"]})

    def test_multiple_owners_can_supply_resolution_evidence_for_one_registered_task(self):
        a, b = material("a"), material("b")
        report = run([[a, b]], lambda value, context: Analysis(resolutions=(resolution("origin:" + TARGET.id, value),)))
        self.assertTrue(report["assessment_valid"])
        self.assertEqual({"a", "b"}, {item["version_id"] for item in report["resolutions"][0]["basis"]})

    def test_runner_fallback_search_task_is_registered_before_psi_returns(self):
        a = material("a")
        def analyze(value, context):
            closures = (resolution("origin:" + TARGET.id),)
            if context["usage"]["rounds"] > 1:
                self.assertIn("inspect-lineage", {gap["id"] for gap in context["gap_registry"]})
                closures += (resolution("inspect-lineage"),)
            return Analysis(resolutions=closures)
        report = run([[a], [a]], analyze)
        self.assertTrue(report["assessment_valid"])
        self.assertIn("inspect-lineage", {item["gap_id"] for item in report["resolutions"]})

    def test_checkpoint_preserves_closed_gap_registration_and_idempotent_resume(self):
        a = material("a")
        task = Gap("verify:status", "Check status", "verification")
        verifier = ScriptedVerifier(lambda context: VerificationResult(
            gaps=(task,) if context["usage"]["rounds"] == 1 else (), resolutions=(resolution(task.id),)))
        checkpoints = []
        prefix = run([[a]], lambda value, context: Analysis(), verifier, checkpoints)
        checkpoint = checkpoints[0]
        self.assertEqual("experimental-round-checkpoint-v2", checkpoint.schema_version)
        self.assertEqual(task, checkpoint.state["gap_registry"][task.id])
        self.assertEqual({"verifier"}, checkpoint.state["gap_owners"][task.id])
        identity = canonical_checkpoint_json(checkpoint)
        noop = run_provenance(TARGET, ReplayTraceProvider([]), verifier=ScriptedVerifier(), checkpoint=checkpoint)
        self.assertEqual(prefix, noop)
        resumed_verifier = ScriptedVerifier(lambda context: VerificationResult(resolutions=(resolution(task.id),)))
        continued_points = []
        continued = run_provenance(TARGET, ReplayTraceProvider([[], [a], [a]]),
            ScriptedDecomposer(lambda value, context: Analysis()), resumed_verifier,
            checkpoint=checkpoint, config=replace(checkpoint.config, max_rounds=3),
            checkpoint_callback=continued_points.append)
        self.assertTrue(continued["assessment_valid"])
        self.assertEqual(prefix["resolutions"], continued["resolutions"])
        self.assertEqual(3, len(continued["verification_history"]))
        self.assertIn(asdict(task), resumed_verifier.contexts[0]["gap_registry"])
        self.assertEqual(identity, canonical_checkpoint_json(checkpoint))
        self.assertEqual(checkpoint.sha256, checkpoint_sha256(checkpoint))
        self.assertEqual(checkpoint.state["gap_registry"], continued_points[-1].state["gap_registry"])

    def test_v1_checkpoint_is_rejected_before_plugins_instead_of_inventing_history(self):
        checkpoints = []
        run([[material("a")]], lambda value, context: Analysis(), ScriptedVerifier(), checkpoints)
        old = replace(checkpoints[0], schema_version="experimental-round-checkpoint-v1")
        old = replace(old, sha256=checkpoint_sha256(old))
        verifier = ScriptedVerifier()
        with self.assertRaisesRegex(ValueError, "regenerate the verified prefix"):
            run_provenance(TARGET, ReplayTraceProvider([]), verifier=verifier, checkpoint=old)
        self.assertEqual([], verifier.contexts)

    def test_verifier_reopening_supersedes_an_older_psi_resolution(self):
        a, b = material("a"), material("b")
        task = Gap("verify:status", "Check bridge status", "verification")
        def analyze(value, context):
            if context["usage"]["rounds"] == 2:
                return Analysis(resolutions=(resolution(task.id),))
            return Analysis()
        verifier = ScriptedVerifier(lambda context: VerificationResult(
            gaps=(task,) if context["usage"]["rounds"] in {1, 2} else ()))
        report = run([[a], [a], [b]], analyze, verifier)
        self.assertTrue(report["assessment_valid"])
        self.assertIn(task.id, {gap["id"] for gap in report["gaps"]})
        self.assertNotIn(task.id, {item["gap_id"] for item in report["resolutions"]})
        self.assertIn(asdict(resolution(task.id)), report["analyses"]["a"]["resolutions"])


if __name__ == "__main__":
    unittest.main()
