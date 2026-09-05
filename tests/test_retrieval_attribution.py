"""Offline v4 retrieval-attribution and strict probe-followup contracts."""
from dataclasses import asdict, replace
import unittest

from newsverify.provenance import (
    Analysis,
    Gap,
    MaterialVersion,
    ProbeAssessment,
    ProbeStop,
    ReplayTraceProvider,
    Resolution,
    RetrievalHit,
    Span,
    Target,
    TraceConfig,
    VerificationResult,
    run_provenance,
)


TARGET = Target("claim", "A synthetic claim.", "2026-09-04T20:00:00Z")


def material(identifier="a"):
    return MaterialVersion(
        identifier,
        "https://example.org/" + identifier,
        "Synthetic source passage.",
        "2026-09-05T01:00:00Z",
        "2026-08-01T12:00:00Z",
        "2026-08-01T12:00:00Z",
        "Synthetic exact-version archive capture.",
    )


def full(value):
    return Span(value.version_id, 0, len(value.content), value.content)


class Decomposer:
    def __init__(self, function):
        self.function = function
        self.contexts = []

    def decompose(self, target, value, context):
        self.contexts.append(context)
        return self.function(target, value, context)


class Verifier:
    def __init__(self, function):
        self.function = function
        self.calls = 0

    def verify(self, target, context):
        self.calls += 1
        return self.function(target, context, self.calls)


class RetrievalAttributionTests(unittest.TestCase):
    def test_legacy_material_return_remains_accepted_and_context_shape_is_unchanged(self):
        decomposer = Decomposer(lambda *_: Analysis())
        report = run_provenance(
            TARGET, ReplayTraceProvider(((material(),),)), decomposer,
            config=TraceConfig(max_rounds=1))
        self.assertFalse(report["errors"])
        self.assertEqual("legacy-compatible", report["retrieval_attribution_mode"])
        self.assertNotIn("current_return", decomposer.contexts[0])
        self.assertNotIn("trigger_task_ids", report["analysis_history"][0])
        self.assertFalse(any(item["action"].startswith("retrieval_attribution_")
                             for item in report["operations"]))

    def test_strict_mode_rejects_bare_material_before_snapshot_or_decomposition(self):
        decomposer = Decomposer(lambda *_: Analysis())
        report = run_provenance(
            TARGET, ReplayTraceProvider(((material(),),)), decomposer,
            config=TraceConfig(max_rounds=1), strict_retrieval_attribution=True)
        self.assertEqual("provider_error", report["stop_reason"])
        self.assertEqual([], report["materials"])
        self.assertEqual(0, report["usage"]["documents"])
        self.assertEqual(0, report["usage"]["decomposition_calls"])
        self.assertEqual([], decomposer.contexts)
        rejected = next(item for item in report["operations"]
                        if item["action"] == "retrieval_attribution_rejected")
        self.assertEqual("MaterialVersion", rejected["return_type"])

    def test_hit_tasks_and_probe_ids_reach_context_observation_history_and_operations(self):
        first, second = material("a"), material("b")

        class Provider:
            def search(self, target, tasks, round_number, limit):
                if round_number == 1:
                    yield RetrievalHit(first, ("origin:claim",))
                elif round_number == 2:
                    self.assert_tasks = tuple(item.id for item in tasks)
                    yield RetrievalHit(second, ("probe-task",))

        provider = Provider()

        def decompose(target, value, context):
            if value.version_id == "a":
                return Analysis(gaps=(Gap(
                    "probe-task", "Retrieve evidence for this probe.",
                    stage="provenance", dimension="provenance",
                    target_id=target.id, probe_id="probe-7"),))
            return Analysis()

        decomposer = Decomposer(decompose)
        report = run_provenance(
            TARGET, provider, decomposer,
            config=TraceConfig(max_rounds=2), strict_retrieval_attribution=True)
        self.assertFalse(report["errors"])
        self.assertIn("probe-task", provider.assert_tasks)
        first_context, second_context = decomposer.contexts
        self.assertEqual(["origin:claim"], first_context["trigger_task_ids"])
        self.assertEqual([], first_context["trigger_probe_ids"])
        current_return = second_context["current_return"]
        self.assertEqual({
            "version_id": "b", "trigger_task_ids": ["probe-task"],
            "trigger_probe_ids": ["probe-7"]},
            {key: current_return[key] for key in (
                "version_id", "trigger_task_ids", "trigger_probe_ids")})
        self.assertEqual([asdict(Gap(
            "probe-task", "Retrieve evidence for this probe.",
            stage="provenance", dimension="provenance",
            target_id=TARGET.id, probe_id="probe-7"))],
            current_return["issued_tasks"])
        self.assertEqual(["probe-task"], second_context["trigger_task_ids"])
        self.assertEqual(["probe-7"], second_context["trigger_probe_ids"])
        self.assertEqual(["probe-task"], report["observations"][1]["trigger_task_ids"])
        self.assertEqual(["probe-7"], report["analysis_history"][1]["trigger_probe_ids"])
        validated = [item for item in report["operations"]
                     if item["action"] == "retrieval_attribution_validated"]
        self.assertEqual(2, len(validated))
        self.assertEqual(["probe-7"], validated[1]["trigger_probe_ids"])

    def test_empty_duplicate_unknown_and_malformed_attribution_fail_closed(self):
        cases = (
            (RetrievalHit(material(), ()), "nonempty tuple"),
            (RetrievalHit(material(), ("origin:claim", "origin:claim")),
             "duplicate task attribution"),
            (RetrievalHit(material(), ("not-issued",)), "not issued in this round"),
            (RetrievalHit(material(), ["origin:claim"]), "nonempty tuple"),
            (RetrievalHit("not-a-material", ("origin:claim",)), "must be a MaterialVersion"),
        )
        for returned, message in cases:
            with self.subTest(message=message):
                class Provider:
                    def search(self, target, tasks, round_number, limit):
                        yield returned
                report = run_provenance(
                    TARGET, Provider(), config=TraceConfig(max_rounds=1),
                    strict_retrieval_attribution=True)
                self.assertEqual("provider_error", report["stop_reason"])
                self.assertIn(message, report["errors"][-1]["message"])
                self.assertEqual([], report["analysis_history"])
                self.assertTrue(any(item["action"] == "retrieval_attribution_rejected"
                                    for item in report["operations"]))

    def test_duplicate_snapshot_keeps_each_valid_return_attribution(self):
        value = material()

        class Provider:
            def search(self, target, tasks, round_number, limit):
                yield RetrievalHit(value, (tasks[0].id,))

        report = run_provenance(
            TARGET, Provider(), config=TraceConfig(
                max_rounds=2, experimental_force_rounds=True),
            strict_retrieval_attribution=True)
        self.assertFalse(report["errors"])
        self.assertEqual(2, len(report["analysis_history"]))
        self.assertEqual(["origin:claim"],
                         report["analysis_history"][0]["trigger_task_ids"])
        self.assertEqual(["origin:claim"],
                         report["analysis_history"][1]["trigger_task_ids"])
        self.assertTrue(report["analysis_history"][1]["duplicate"])

    def test_verification_task_attribution_carries_its_probe_to_next_decomposition(self):
        first, second = material("a"), material("b")
        target = replace(TARGET, evidence_scope=("a", "b"))

        class Provider:
            def search(self, target, tasks, round_number, limit):
                selected = "origin:claim" if round_number == 1 else "probe-follow"
                self.last_tasks = tuple(item.id for item in tasks)
                yield RetrievalHit(first if round_number == 1 else second, (selected,))

        provider = Provider()
        decomposer = Decomposer(lambda *_: Analysis())
        verifier_returns = []

        def verify(target, context, call):
            verifier_returns.append(context["current_round_returns"])
            value = first if call == 1 else second
            status = "unresolved" if call == 1 else "supported"
            assessment = ProbeAssessment(
                "probe-9", "claim-part", "evidence", status, (full(value),),
                "Synthetic attributed follow-up result.")
            if call == 1:
                return VerificationResult(
                    evidence_probe_results=(assessment,), gaps=(Gap(
                        "probe-follow", "Retrieve a source for probe 9.",
                        stage="verification", dimension="evidence", target_id=target.id,
                        basis=(full(value),), decision_impact="Could decide probe 9.",
                        probe_id="probe-9"),), strict_probe_followups=True)
            return VerificationResult(
                "supported", (full(value),), "The second source supports probe 9.",
                evidence_verdict="supported", evidence_probe_results=(assessment,),
                strict_probe_followups=True)

        report = run_provenance(
            target, provider, decomposer, Verifier(verify),
            TraceConfig(max_rounds=2, experimental_force_rounds=True),
            strict_retrieval_attribution=True)
        self.assertFalse(report["errors"])
        self.assertIn("probe-follow", provider.last_tasks)
        self.assertEqual(["probe-follow"], decomposer.contexts[1]["trigger_task_ids"])
        self.assertEqual(["probe-9"], decomposer.contexts[1]["trigger_probe_ids"])
        self.assertEqual(["b"], [item["version_id"]
                                 for item in verifier_returns[1]])
        self.assertEqual(["probe-follow"],
                         verifier_returns[1][0]["trigger_task_ids"])
        self.assertEqual(["probe-9"],
                         verifier_returns[1][0]["trigger_probe_ids"])
        self.assertEqual("probe-follow",
                         verifier_returns[1][0]["issued_tasks"][0]["id"])
        self.assertEqual("supported", report["decision_status"])

    def test_checkpoint_cannot_silently_downgrade_strict_retrieval(self):
        value = material()
        checkpoints = []

        class Provider:
            def search(self, target, tasks, round_number, limit):
                yield RetrievalHit(value, (tasks[0].id,))

        verifier = Verifier(lambda *_: VerificationResult())
        run_provenance(
            TARGET, Provider(), verifier=verifier,
            config=TraceConfig(max_rounds=1), checkpoint_callback=checkpoints.append,
            strict_retrieval_attribution=True)
        self.assertEqual(1, len(checkpoints))
        with self.assertRaisesRegex(ValueError, "preserve strict retrieval attribution"):
            run_provenance(
                TARGET, ReplayTraceProvider(()), verifier=verifier,
                config=TraceConfig(max_rounds=2), checkpoint=checkpoints[0])


class StrictProbeFollowupTests(unittest.TestCase):
    def run_check(self, check):
        return run_provenance(
            TARGET, ReplayTraceProvider(((material(),),)),
            verifier=Verifier(lambda *_: check), config=TraceConfig(max_rounds=1))

    @staticmethod
    def assessment(stage="evidence", status="unresolved", probe_id="probe"):
        value = material()
        return ProbeAssessment(
            probe_id, "claim-part", stage, status, (full(value),),
            "Synthetic grounded probe assessment.")

    @staticmethod
    def gap(stage="evidence", probe_id="probe", identifier="follow"):
        value = material()
        return Gap(
            identifier, "Retrieve one missing source.", stage="verification",
            dimension=stage, blocking=True, target_id=TARGET.id,
            basis=(full(value),), decision_impact="Could decide the probe.",
            action="search", probe_id=probe_id)

    def test_unresolved_probe_accepts_exactly_one_gap(self):
        check = VerificationResult(
            evidence_probe_results=(self.assessment(),), gaps=(self.gap(),),
            strict_probe_followups=True)
        report = self.run_check(check)
        self.assertFalse(report["errors"])
        retained = report["verification_history"][0]
        self.assertTrue(retained["strict_probe_followups"])
        self.assertEqual((), retained["probe_stops"])
        audit = next(item for item in report["operations"]
                     if item["action"] == "probe_followups_validated")
        self.assertEqual("follow", audit["active"][0]["gap_id"])

    def test_unresolved_probe_accepts_one_stop_and_no_gap(self):
        stop = ProbeStop("probe", "evidence", "corpus_exhausted",
                         "Every supplied corpus snapshot was checked.")
        check = VerificationResult(
            evidence_probe_results=(self.assessment(),),
            strict_probe_followups=True, probe_stops=(stop,))
        report = self.run_check(check)
        self.assertFalse(report["errors"])
        retained = report["verification_history"][0]["probe_stops"]
        self.assertEqual("corpus_exhausted", retained[0]["reason"])
        self.assertNotIn("follow", {item["id"] for item in report["gaps"]})

    def test_nondecisive_unresolved_probe_keeps_a_nonblocking_task(self):
        value = material()
        check = VerificationResult(
            "supported", (full(value),),
            "Another branch establishes the aggregate while this probe remains open.",
            evidence_verdict="supported",
            evidence_probe_results=(self.assessment(),),
            gaps=(replace(self.gap(), blocking=False),),
            strict_probe_followups=True)
        report = self.run_check(check)
        self.assertFalse(report["errors"])
        retained = next(item for item in report["gaps"] if item["id"] == "follow")
        self.assertFalse(retained["blocking"])
        audit = next(item for item in report["operations"]
                     if item["action"] == "probe_followups_validated")
        self.assertFalse(audit["active"][0]["blocking"])

        wrong = replace(check, gaps=(self.gap(),))
        rejected = self.run_check(wrong)
        self.assertEqual("verifier_error", rejected["stop_reason"])
        self.assertIn("blocking must match", rejected["errors"][-1]["message"])

    def test_same_probe_in_two_layers_gets_independent_followups(self):
        stop = ProbeStop("probe", "evidence", "no_source_lead",
                         "No scoped evidence lead remains.")
        check = VerificationResult(
            evidence_probe_results=(self.assessment("evidence"),),
            world_probe_results=(self.assessment("world"),),
            gaps=(self.gap("world", identifier="world-follow"),),
            strict_probe_followups=True, probe_stops=(stop,))
        report = self.run_check(check)
        self.assertFalse(report["errors"])
        audit = next(item for item in report["operations"]
                     if item["action"] == "probe_followups_validated")
        self.assertEqual("world", audit["active"][0]["stage"])
        self.assertEqual("evidence", audit["stops"][0]["stage"])

    def test_conclusive_probe_accepts_no_gap_or_stop(self):
        value = material()
        assessment = self.assessment(status="supported")
        check = VerificationResult(
            "supported", (full(value),), "The passage supports the probe.",
            evidence_verdict="supported", evidence_probe_results=(assessment,),
            strict_probe_followups=True)
        report = self.run_check(check)
        self.assertFalse(report["errors"])
        self.assertEqual((), report["verification_history"][0]["probe_stops"])

    def test_strict_xor_invalid_shapes_fail_before_history_commit(self):
        value = material()
        unresolved = self.assessment()
        supported = self.assessment(status="supported")
        valid_stop = ProbeStop("probe", "evidence", "no_source_lead",
                               "No identifiable source lead remains.")
        cases = (
            (VerificationResult(evidence_probe_results=(unresolved,),
                                strict_probe_followups=True),
             "exactly one active gap or one stop"),
            (VerificationResult(evidence_probe_results=(unresolved,),
                                gaps=(self.gap(),), strict_probe_followups=True,
                                probe_stops=(valid_stop,)),
             "both a gap and a stop"),
            (VerificationResult(
                "supported", (full(value),), "Supported.",
                evidence_verdict="supported", evidence_probe_results=(supported,),
                gaps=(self.gap(),), strict_probe_followups=True),
             "conclusive probes cannot retain"),
            (VerificationResult(
                "supported", (full(value),), "Supported.",
                evidence_verdict="supported", evidence_probe_results=(supported,),
                gaps=(self.gap(),),
                resolutions=(Resolution(
                    "follow", (full(value),), "The same response closes it."),),
                strict_probe_followups=True),
             "conclusive probes cannot retain"),
            (VerificationResult(
                "supported", (full(value),), "Supported.",
                evidence_verdict="supported", evidence_probe_results=(supported,),
                strict_probe_followups=True, probe_stops=(valid_stop,)),
             "conclusive probes cannot retain"),
            (VerificationResult(evidence_probe_results=(unresolved,),
                                strict_probe_followups=True,
                                probe_stops=(replace(valid_stop, probe_id="unknown"),)),
             "unassessed layer/probe"),
            (VerificationResult(evidence_probe_results=(unresolved,),
                                strict_probe_followups=True,
                                probe_stops=(replace(valid_stop, reason="gave_up"),)),
             "invalid probe stop reason"),
            (VerificationResult(evidence_probe_results=(unresolved,),
                                gaps=(replace(self.gap(), dimension="auto"),),
                                strict_probe_followups=True),
             "explicit layer and probe_id"),
            (VerificationResult(evidence_probe_results=(unresolved,),
                                gaps=(replace(self.gap(), blocking=False),),
                                strict_probe_followups=True),
             "blocking must match"),
            (VerificationResult(evidence_probe_results=(unresolved,),
                                strict_probe_followups=True,
                                probe_stops=(valid_stop, valid_stop)),
             "duplicate probe stop"),
            (VerificationResult(evidence_probe_results=(unresolved,),
                                probe_stops=(valid_stop,)),
             "require strict_probe_followups"),
        )
        for check, message in cases:
            with self.subTest(message=message):
                report = self.run_check(check)
                self.assertEqual("verifier_error", report["stop_reason"])
                self.assertEqual([], report["verification_history"])
                self.assertIn(message, report["errors"][-1]["message"])
                self.assertTrue(any(item["action"] == "verification_rejected"
                                    for item in report["operations"]))

    def test_later_stop_closes_prior_active_gap_without_fake_resolution(self):
        first, second = material("a"), material("b")
        target = replace(TARGET, evidence_scope=("a", "b"))

        def verify(target, context, call):
            basis = (full(first if call == 1 else second),)
            assessment = ProbeAssessment(
                "probe", "claim-part", "evidence", "unresolved", basis,
                "The probe remains unresolved.")
            if call == 1:
                gap = Gap(
                    "follow", "Find another source.", stage="verification",
                    dimension="evidence", target_id=target.id, basis=basis,
                    decision_impact="Could decide the probe.", probe_id="probe")
                return VerificationResult(
                    evidence_probe_results=(assessment,), gaps=(gap,),
                    strict_probe_followups=True)
            return VerificationResult(
                evidence_probe_results=(assessment,), strict_probe_followups=True,
                probe_stops=(ProbeStop(
                    "probe", "evidence", "budget_exhausted",
                    "The frozen retrieval budget has been exhausted."),))

        report = run_provenance(
            target, ReplayTraceProvider(((first,), (second,))),
            verifier=Verifier(verify),
            config=TraceConfig(max_rounds=2, experimental_force_rounds=True))
        self.assertFalse(report["errors"])
        self.assertNotIn("follow", {item["id"] for item in report["gaps"]})
        self.assertNotIn("follow", {item["gap_id"] for item in report["resolutions"]})
        audit = [item for item in report["operations"]
                 if item["action"] == "probe_followups_validated"][-1]
        self.assertEqual(["follow"], audit["stopped_prior_gap_ids"])

    def test_new_gap_id_atomically_supersedes_the_same_probe_slot(self):
        first, second = material("a"), material("b")
        target = replace(TARGET, evidence_scope=("a", "b"))
        old = replace(self.gap(identifier="old-follow"), locator="old lead")
        new = replace(self.gap(identifier="new-follow"), locator="new lead")

        def verify(target, context, call):
            value = first if call == 1 else second
            assessment = self.assessment()
            assessment = replace(assessment, basis=(full(value),))
            gap = replace(old if call == 1 else new, basis=(full(value),))
            return VerificationResult(
                evidence_probe_results=(assessment,), gaps=(gap,),
                strict_probe_followups=True)

        report = run_provenance(
            target, ReplayTraceProvider(((first,), (second,))),
            verifier=Verifier(verify),
            config=TraceConfig(max_rounds=2, experimental_force_rounds=True))
        self.assertFalse(report["errors"])
        active = {item["id"] for item in report["gaps"]
                  if item["stage"] == "verification"}
        self.assertEqual({"new-follow"}, active)
        registry = {item["id"] for item in report["gap_registry"]}
        self.assertTrue({"old-follow", "new-follow"} <= registry)
        self.assertNotIn("old-follow",
                         {item["gap_id"] for item in report["resolutions"]})
        audit = [item for item in report["operations"]
                 if item["action"] == "probe_followups_validated"][-1]
        self.assertEqual([{
            "stage": "evidence", "probe_id": "probe",
            "prior_gap_id": "old-follow", "replacement_gap_id": "new-follow",
        }], audit["superseded_prior_gaps"])
        self.assertEqual(["old-follow", "new-follow"], [
            record["gaps"][0]["id"] for record in report["verification_history"]])

    def test_same_probe_id_in_another_layer_does_not_supersede(self):
        first, second = material("a"), material("b")
        target = replace(TARGET, evidence_scope=("a", "b"))

        def verify(target, context, call):
            value = first if call == 1 else second
            evidence = replace(self.assessment("evidence"), basis=(full(value),))
            if call == 1:
                return VerificationResult(
                    evidence_probe_results=(evidence,),
                    gaps=(replace(self.gap("evidence", identifier="evidence-old"),
                                  basis=(full(value),)),),
                    strict_probe_followups=True)
            world = replace(self.assessment("world"), basis=(full(value),))
            return VerificationResult(
                evidence_probe_results=(evidence,), world_probe_results=(world,),
                gaps=(replace(self.gap("world", identifier="world-new"),
                              basis=(full(value),)),),
                strict_probe_followups=True)

        report = run_provenance(
            target, ReplayTraceProvider(((first,), (second,))),
            verifier=Verifier(verify),
            config=TraceConfig(max_rounds=2, experimental_force_rounds=True))
        self.assertFalse(report["errors"])
        active = {item["id"] for item in report["gaps"]
                  if item["stage"] == "verification"}
        self.assertEqual({"evidence-old", "world-new"}, active)
        audit = [item for item in report["operations"]
                 if item["action"] == "probe_followups_validated"][-1]
        self.assertEqual([], audit["superseded_prior_gaps"])

    def test_failed_replacement_rolls_back_the_active_gap_and_registry(self):
        first, second = material("a"), material("b")
        target = replace(TARGET, evidence_scope=("a", "b"))

        def verify(target, context, call):
            value = first if call == 1 else second
            assessment = replace(self.assessment(), basis=(full(value),))
            gap = replace(self.gap(identifier="old" if call == 1 else "new"),
                          basis=(full(value),), blocking=call == 1)
            return VerificationResult(
                evidence_probe_results=(assessment,), gaps=(gap,),
                strict_probe_followups=True)

        report = run_provenance(
            target, ReplayTraceProvider(((first,), (second,))),
            verifier=Verifier(verify),
            config=TraceConfig(max_rounds=2, experimental_force_rounds=True))
        self.assertEqual("verifier_error", report["stop_reason"])
        self.assertIn("blocking must match", report["errors"][-1]["message"])
        self.assertEqual({"old"}, {item["id"] for item in report["gaps"]
                                  if item["stage"] == "verification"})
        self.assertNotIn("new", {item["id"] for item in report["gap_registry"]})
        self.assertEqual(1, len(report["verification_history"]))

    def test_superseded_id_cannot_reactivate_in_run_or_after_checkpoint_resume(self):
        values = [material(identifier) for identifier in ("a", "b", "c")]
        target = replace(TARGET, evidence_scope=tuple(item.version_id for item in values))
        gaps = {
            "old": replace(self.gap(identifier="old"), locator="old lead"),
            "new": replace(self.gap(identifier="new"), locator="new lead"),
        }

        def result(value, identifier):
            assessment = replace(self.assessment(), basis=(full(value),))
            gap = replace(gaps[identifier], basis=(full(value),))
            return VerificationResult(
                evidence_probe_results=(assessment,), gaps=(gap,),
                strict_probe_followups=True)

        verifier = Verifier(lambda target, context, call:
                            result(values[call - 1], "new" if call == 2 else "old"))
        report = run_provenance(
            target, ReplayTraceProvider(tuple((value,) for value in values)),
            verifier=verifier,
            config=TraceConfig(max_rounds=3, experimental_force_rounds=True))
        self.assertEqual("verifier_error", report["stop_reason"])
        self.assertIn("cannot be reactivated", report["errors"][-1]["message"])
        self.assertEqual(["old", "new"], [record["gaps"][0]["id"]
                                           for record in report["verification_history"]])
        self.assertEqual({"new"}, {item["id"] for item in report["gaps"]
                                  if item["stage"] == "verification"})

        checkpoints = []
        prefix_verifier = Verifier(lambda target, context, call:
                                   result(values[call - 1], "old" if call == 1 else "new"))
        run_provenance(
            target, ReplayTraceProvider(((values[0],), (values[1],))),
            verifier=prefix_verifier,
            config=TraceConfig(max_rounds=2, experimental_force_rounds=True),
            checkpoint_callback=checkpoints.append)
        checkpoint = checkpoints[-1]
        resumed = run_provenance(
            target, ReplayTraceProvider(((), (), (values[2],))),
            verifier=Verifier(lambda *_: result(values[2], "old")),
            config=replace(checkpoint.config, max_rounds=3), checkpoint=checkpoint)
        self.assertEqual("verifier_error", resumed["stop_reason"])
        self.assertIn("cannot be reactivated", resumed["errors"][-1]["message"])
        self.assertEqual({"new"}, {item["id"] for item in resumed["gaps"]
                                  if item["stage"] == "verification"})

    def test_default_false_keeps_legacy_unresolved_probe_without_followup(self):
        check = VerificationResult(evidence_probe_results=(self.assessment(),))
        report = self.run_check(check)
        self.assertFalse(report["errors"])
        retained = report["verification_history"][0]
        self.assertNotIn("strict_probe_followups", retained)
        self.assertNotIn("probe_stops", retained)


if __name__ == "__main__":
    unittest.main()
