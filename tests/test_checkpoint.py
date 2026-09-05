"""Offline continuation/fork contracts; these do not measure news accuracy."""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import unittest

from newsverify.provenance import (
    Analysis, ConservativeDecomposer, MaterialVersion, OriginFinding, Resolution,
    Span, Target, TraceCheckpoint, TraceConfig, VerificationResult,
    canonical_checkpoint_json, checkpoint_sha256, run_provenance,
)


TARGET = Target("checkpoint-claim", "The synthetic record contains a result.",
                "2026-09-04T20:00:00Z")


def material(identifier):
    return MaterialVersion(
        identifier, "https://example.org/" + identifier,
        "Synthetic record " + identifier + " contains a result.",
        "2026-09-05T01:00:00Z", "2026-08-01T12:00:00Z",
        "2026-08-01T12:00:00Z", "Synthetic exact-version capture.")


def span(value):
    return Span(value.version_id, 0, len(value.content), value.content)


class RecordingProvider:
    def __init__(self, rounds):
        self.rounds = rounds
        self.calls = []

    def search(self, target, tasks, round_number, limit):
        self.calls.append((round_number, deepcopy(tasks), limit))
        return self.rounds.get(round_number, ())[:limit]


class RecordingDecomposer:
    def __init__(self, complete=False, fail=False):
        self.calls = []
        self.complete = complete
        self.fail = fail

    def decompose(self, target, value, context):
        self.calls.append((value.version_id, deepcopy(context)))
        if self.fail:
            context["materials"].clear()
            raise RuntimeError("synthetic branch failure")
        analysis = ConservativeDecomposer().decompose(target, value, context)
        if self.complete:
            analysis = replace(analysis,
                origins=(OriginFinding(target.id, value.version_id, (span(value),),
                                       "original_record", "Synthetic fixture origin."),),
                resolutions=(Resolution("origin:" + target.id, (span(value),),
                                        "Synthetic fixture resolution."),))
        return analysis


class RecordingVerifier:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def verify(self, target, context):
        self.calls.append(deepcopy(context))
        if self.fail:
            raise RuntimeError("synthetic verifier failure")
        value = MaterialVersion(**context["materials"][0])
        return VerificationResult("supported", (span(value),),
                                  "Synthetic fixture judgement.")


class SavedAnalysisDecomposer:
    """A frozen-analysis arm reuses typed prefix analyses without a model call."""

    def __init__(self, checkpoint):
        self.saved = deepcopy(checkpoint.state["current_analyses"])
        self.fresh = RecordingDecomposer()
        self.reused = []

    def decompose(self, target, value, context):
        if value.version_id in self.saved:
            self.reused.append(value.version_id)
            return deepcopy(self.saved[value.version_id])
        return self.fresh.decompose(target, value, context)


class CheckpointTests(unittest.TestCase):
    def build_checkpoint(self, max_rounds=3):
        provider = RecordingProvider({i: (material(chr(96 + i)),)
                                      for i in range(1, max_rounds + 1)})
        decomposer, verifier, checkpoints = RecordingDecomposer(), RecordingVerifier(), []
        report = run_provenance(TARGET, provider, decomposer, verifier,
            config=TraceConfig(max_rounds=max_rounds, experimental_force_rounds=True),
            checkpoint_callback=checkpoints.append)
        self.assertEqual(max_rounds, len(checkpoints))
        return checkpoints, report, decomposer, verifier

    def test_typed_checkpoint_hashes_canonical_payload_and_precedes_stop(self):
        checkpoints, report, _, _ = self.build_checkpoint()
        checkpoint = checkpoints[0]
        self.assertIsInstance(checkpoint, TraceCheckpoint)
        self.assertEqual(TARGET, checkpoint.target)
        self.assertIsInstance(checkpoint.config, TraceConfig)
        self.assertIsInstance(checkpoint.state["current_analyses"]["a"], Analysis)
        self.assertEqual(1, checkpoint.state["usage"]["rounds"])
        self.assertEqual(1, checkpoint.state["usage"]["verification_calls"])
        payload = canonical_checkpoint_json(checkpoint)
        json.loads(payload)
        self.assertEqual(payload, canonical_checkpoint_json(deepcopy(checkpoint)))
        self.assertEqual(hashlib.sha256(payload.encode("utf-8")).hexdigest(), checkpoint.sha256)
        self.assertEqual(checkpoint.sha256, checkpoint_sha256(checkpoint))
        actions = [entry["action"] for entry in checkpoint.state["operations"]]
        self.assertIn("verification_completed", actions)
        self.assertIn("progress_checked", actions)
        self.assertNotIn("stopped", actions)
        self.assertEqual("stopped", report["operations"][-1]["action"])

    def test_resume_inherits_config_and_does_not_replay_prefix_plugins(self):
        checkpoints, original, _, original_verifier = self.build_checkpoint()
        checkpoint = checkpoints[0]
        unchanged = canonical_checkpoint_json(checkpoint)
        provider = RecordingProvider({2: (material("b"),), 3: (material("c"),)})
        decomposer, verifier = RecordingDecomposer(), RecordingVerifier()
        resumed = run_provenance(TARGET, provider, decomposer, verifier, checkpoint=checkpoint)
        self.assertEqual([2, 3], [call[0] for call in provider.calls])
        self.assertEqual(["b", "c"], [call[0] for call in decomposer.calls])
        self.assertEqual([2, 3], [call["usage"]["rounds"] for call in verifier.calls])
        self.assertEqual(original_verifier.calls[1:], verifier.calls)
        self.assertEqual(asdict(checkpoint.config), resumed["config"])
        for key in ("materials", "analyses", "analysis_history", "verification_history",
                    "observations", "usage", "assessments", "gaps", "resolutions"):
            self.assertEqual(original[key], resumed[key], key)
        prefix = checkpoint.state["operations"]
        self.assertEqual(prefix, resumed["operations"][:len(prefix)])
        self.assertEqual(1, sum(x["action"] == "stopped" for x in resumed["operations"]))
        self.assertEqual(unchanged, canonical_checkpoint_json(checkpoint))

    def test_same_state_forks_reuse_saved_analysis_and_keep_independent_histories(self):
        checkpoints, _, _, _ = self.build_checkpoint(max_rounds=1)
        checkpoint = checkpoints[0]
        before = canonical_checkpoint_json(checkpoint)
        config = replace(checkpoint.config, max_rounds=3)
        rounds = {2: (material("a"),), 3: (material("a"),)}
        frozen = SavedAnalysisDecomposer(checkpoint)
        fresh = RecordingDecomposer()
        frozen_provider, fresh_provider = RecordingProvider(rounds), RecordingProvider(rounds)
        frozen_verifier, fresh_verifier = RecordingVerifier(), RecordingVerifier()
        frozen_report = run_provenance(TARGET, frozen_provider, frozen, frozen_verifier,
                                       config=config, checkpoint=checkpoint)
        fresh_report = run_provenance(TARGET, fresh_provider, fresh, fresh_verifier,
                                      config=config, checkpoint=checkpoint)
        self.assertEqual(["a", "a"], frozen.reused)
        self.assertEqual([], frozen.fresh.calls)
        self.assertEqual(["a", "a"], [call[0] for call in fresh.calls])
        self.assertEqual([2, 3], [call[0] for call in frozen_provider.calls])
        self.assertEqual([2, 3], [call[0] for call in fresh_provider.calls])
        self.assertEqual([2, 3], [call["usage"]["rounds"] for call in frozen_verifier.calls])
        self.assertEqual(fresh_verifier.calls, frozen_verifier.calls)
        self.assertEqual([1, 2, 3], [event["round"] for event in frozen_report["operations"]
                                    if event["action"] == "graph_updated"])
        self.assertEqual(fresh_report["analysis_history"], frozen_report["analysis_history"])
        self.assertEqual(fresh_report["verification_history"], frozen_report["verification_history"])
        frozen_report["analysis_history"][0]["analysis"]["notes"] = "changed branch output"
        frozen_report["operations"][0]["action"] = "changed branch output"
        self.assertNotEqual(frozen_report["analysis_history"], fresh_report["analysis_history"])
        self.assertEqual(before, canonical_checkpoint_json(checkpoint))

    def test_failed_sibling_leaves_checkpoint_and_successful_sibling_intact(self):
        checkpoints, _, _, _ = self.build_checkpoint(max_rounds=1)
        checkpoint = checkpoints[0]
        before = canonical_checkpoint_json(checkpoint)
        config = replace(checkpoint.config, max_rounds=2)
        failed = run_provenance(TARGET, RecordingProvider({2: (material("b"),)}),
            RecordingDecomposer(fail=True), RecordingVerifier(), config=config, checkpoint=checkpoint)
        successful = run_provenance(TARGET, RecordingProvider({2: (material("b"),)}),
            RecordingDecomposer(), RecordingVerifier(), config=config, checkpoint=checkpoint)
        self.assertEqual("decomposer_error", failed["stop_reason"])
        self.assertFalse(failed["assessment_valid"])
        self.assertEqual("unresolved", failed["fact_status"])
        self.assertTrue(successful["assessment_valid"])
        self.assertEqual("supported", successful["fact_status"])
        self.assertEqual(1, len(failed["verification_history"]))
        self.assertEqual(2, len(successful["verification_history"]))
        self.assertEqual(before, canonical_checkpoint_json(checkpoint))

    def test_callback_mutation_cannot_corrupt_live_runner_or_later_snapshots(self):
        clean = []
        def callback(checkpoint):
            clean.append(deepcopy(checkpoint))
            checkpoint.state["current_analyses"].clear()
            checkpoint.state["operations"][0]["action"] = "callback mutation"
            checkpoint.state["usage"]["documents"] = 999

        report = run_provenance(TARGET,
            RecordingProvider({1: (material("a"),), 2: (material("b"),)}),
            RecordingDecomposer(), RecordingVerifier(),
            config=TraceConfig(max_rounds=2, experimental_force_rounds=True),
            checkpoint_callback=callback)
        self.assertEqual(2, report["usage"]["documents"])
        self.assertEqual({"a", "b"}, set(report["analyses"]))
        self.assertEqual({"a", "b"}, set(clean[1].state["current_analyses"]))
        self.assertEqual("search", report["operations"][0]["action"])
        self.assertEqual("search", clean[1].state["operations"][0]["action"])
        self.assertEqual(clean[0].sha256, checkpoint_sha256(clean[0]))

    def assert_rejected_before_plugins(self, checkpoint, target=TARGET, config=None,
                                       omit_verifier=False):
        provider, decomposer, verifier = RecordingProvider({}), RecordingDecomposer(), RecordingVerifier()
        with self.assertRaises(ValueError):
            run_provenance(target, provider, decomposer, None if omit_verifier else verifier,
                           config=config, checkpoint=checkpoint)
        self.assertEqual([], provider.calls)
        self.assertEqual([], decomposer.calls)
        self.assertEqual([], verifier.calls)

    def test_target_mismatch_hash_mutation_and_malformed_checkpoint_rejected(self):
        checkpoints, _, _, _ = self.build_checkpoint()
        checkpoint = checkpoints[0]
        for target in (replace(TARGET, id="different"), replace(TARGET, text="Changed claim."),
                       replace(TARGET, as_of="2026-09-03T20:00:00Z"),
                       replace(TARGET, assessment_mode="evidence")):
            with self.subTest(target=target):
                self.assert_rejected_before_plugins(checkpoint, target=target)
        damaged = deepcopy(checkpoint)
        damaged.state["usage"]["documents"] += 1
        malformed = replace(checkpoint, state={})
        malformed = replace(malformed, sha256=checkpoint_sha256(malformed))
        for candidate in ({}, "checkpoint", replace(checkpoint, sha256="0" * 64), damaged, malformed):
            with self.subTest(candidate_type=type(candidate).__name__):
                self.assert_rejected_before_plugins(candidate)

    def test_resume_rejects_changed_hard_caps_round_regression_and_missing_verifier(self):
        checkpoints, _, _, _ = self.build_checkpoint()
        checkpoint = checkpoints[1]
        for config in (replace(checkpoint.config, max_documents=31),
                       replace(checkpoint.config, max_decomposition_calls=31),
                       replace(checkpoint.config, max_rounds=1)):
            with self.subTest(config=config):
                self.assert_rejected_before_plugins(checkpoint, config=config)
        self.assert_rejected_before_plugins(checkpoint, omit_verifier=True)

    def test_resume_can_disable_force_and_retains_prior_progress_bookkeeping(self):
        checkpoints, _, _, _ = self.build_checkpoint(max_rounds=1)
        checkpoint = checkpoints[0]
        provider = RecordingProvider({2: (material("a"),), 3: (material("b"),)})
        report = run_provenance(TARGET, provider, RecordingDecomposer(), RecordingVerifier(),
            checkpoint=checkpoint, config=replace(checkpoint.config, max_rounds=3,
                                                  experimental_force_rounds=False))
        self.assertEqual("no_new_eligible_materials", report["stop_reason"])
        self.assertEqual([2], [call[0] for call in provider.calls])
        self.assertEqual(2, report["usage"]["rounds"])
        self.assertFalse(report["config"]["experimental_force_rounds"])
        self.assertTrue(checkpoint.config.experimental_force_rounds)

    def test_callback_runs_on_complete_and_default_completion_is_retained(self):
        target = replace(TARGET, source_version_id="a")
        provider, checkpoints = RecordingProvider({1: (material("a"),), 2: (material("b"),)}), []
        report = run_provenance(target, provider, RecordingDecomposer(complete=True),
            RecordingVerifier(), checkpoint_callback=checkpoints.append)
        self.assertEqual("complete", report["stop_reason"])
        self.assertEqual([1], [call[0] for call in provider.calls])
        self.assertEqual(1, len(checkpoints))
        self.assertEqual(1, checkpoints[0].state["usage"]["verification_calls"])
        self.assertFalse(report["config"]["experimental_force_rounds"])
        self.assertNotIn("stopped", [x["action"] for x in checkpoints[0].state["operations"]])

    def test_completed_prefix_can_resume_forced_with_origin_and_resolution_state(self):
        target, checkpoints = replace(TARGET, source_version_id="a"), []
        prefix = run_provenance(target, RecordingProvider({1: (material("a"),)}),
            RecordingDecomposer(complete=True), RecordingVerifier(),
            config=TraceConfig(max_rounds=1), checkpoint_callback=checkpoints.append)
        self.assertEqual("complete", prefix["stop_reason"])
        checkpoint = checkpoints[0]
        self.assertEqual({(target.id, "a")}, set(checkpoint.state["origins"]))
        self.assertEqual({"origin:" + target.id}, set(checkpoint.state["resolved"]))
        before = canonical_checkpoint_json(checkpoint)
        provider = RecordingProvider({2: (material("a"),), 3: (material("a"),)})
        decomposer, verifier = SavedAnalysisDecomposer(checkpoint), RecordingVerifier()
        continued = run_provenance(target, provider, decomposer, verifier,
            config=replace(checkpoint.config, max_rounds=3, experimental_force_rounds=True),
            checkpoint=checkpoint)
        self.assertEqual([2, 3], [call[0] for call in provider.calls])
        self.assertEqual([2, 3], [call["usage"]["rounds"] for call in verifier.calls])
        self.assertEqual([], decomposer.fresh.calls)
        self.assertEqual(["a", "a"], decomposer.reused)
        self.assertEqual(prefix["origins"], continued["origins"])
        self.assertEqual(prefix["resolutions"], continued["resolutions"])
        self.assertEqual(prefix["analysis_history"], continued["analysis_history"][:1])
        self.assertEqual(prefix["verification_history"], continued["verification_history"][:1])
        self.assertEqual(prefix["operations"][:-1],
                         continued["operations"][:len(prefix["operations"]) - 1])
        self.assertEqual("original_material_located", continued["provenance_status"])
        self.assertEqual("supported", continued["fact_status"])
        self.assertEqual("round_budget", continued["stop_reason"])
        self.assertEqual(before, canonical_checkpoint_json(checkpoint))

    def test_noop_resume_of_completed_prefix_retains_complete_stop_reason(self):
        target, checkpoints = replace(TARGET, source_version_id="a"), []
        prefix = run_provenance(target, RecordingProvider({1: (material("a"),)}),
            RecordingDecomposer(complete=True), RecordingVerifier(),
            config=TraceConfig(max_rounds=1), checkpoint_callback=checkpoints.append)
        provider, decomposer, verifier = RecordingProvider({}), RecordingDecomposer(), RecordingVerifier()
        resumed = run_provenance(target, provider, decomposer, verifier, checkpoint=checkpoints[0])
        self.assertEqual([], provider.calls)
        self.assertEqual([], decomposer.calls)
        self.assertEqual([], verifier.calls)
        self.assertEqual("complete", resumed["stop_reason"])
        self.assertEqual(prefix, resumed)

    def test_force_bypasses_complete_and_no_progress_but_empty_round_does_not_verify(self):
        target = replace(TARGET, source_version_id="a")
        provider = RecordingProvider({1: (material("a"),), 2: (material("a"),)})
        verifier, checkpoints = RecordingVerifier(), []
        report = run_provenance(target, provider, RecordingDecomposer(complete=True), verifier,
            config=TraceConfig(max_rounds=5, experimental_force_rounds=True),
            checkpoint_callback=checkpoints.append)
        self.assertEqual([1, 2, 3], [call[0] for call in provider.calls])
        self.assertEqual([1, 2], [call["usage"]["rounds"] for call in verifier.calls])
        self.assertEqual(2, len(checkpoints))
        self.assertEqual("empty_results", report["stop_reason"])
        self.assertEqual(2, report["usage"]["verification_calls"])

    def test_default_no_progress_stop_is_retained(self):
        provider = RecordingProvider({i: (material("a"),) for i in range(1, 6)})
        report = run_provenance(TARGET, provider, RecordingDecomposer(), RecordingVerifier())
        self.assertEqual("no_new_eligible_materials", report["stop_reason"])
        self.assertEqual([1, 2], [call[0] for call in provider.calls])

    def test_force_does_not_bypass_document_or_decomposition_budgets(self):
        for overrides, reason in (({"max_documents": 2}, "document_budget"),
                                  ({"max_decomposition_calls": 2}, "decomposition_budget")):
            with self.subTest(reason=reason):
                provider = RecordingProvider({i: (material("a"),) for i in range(1, 6)})
                report = run_provenance(TARGET, provider, RecordingDecomposer(), RecordingVerifier(),
                    config=TraceConfig(max_rounds=5, experimental_force_rounds=True, **overrides))
                self.assertEqual(reason, report["stop_reason"])
                self.assertEqual([1, 2], [call[0] for call in provider.calls])
                self.assertEqual(2, report["usage"]["decomposition_calls"])

    def test_failed_verification_does_not_emit_checkpoint_even_when_forced(self):
        checkpoints, verifier = [], RecordingVerifier(fail=True)
        provider = RecordingProvider({1: (material("a"),), 2: (material("b"),)})
        report = run_provenance(TARGET, provider, RecordingDecomposer(), verifier,
            config=TraceConfig(experimental_force_rounds=True), checkpoint_callback=checkpoints.append)
        self.assertEqual("verifier_error", report["stop_reason"])
        self.assertEqual([], checkpoints)
        self.assertEqual([1], [call[0] for call in provider.calls])


if __name__ == "__main__":
    unittest.main()
