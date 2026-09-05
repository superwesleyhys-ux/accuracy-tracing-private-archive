from copy import deepcopy
import unittest

from experiments.proof_score import ARMS, exact_mcnemar, power_planning, score


def fixture(decisions=("true", "false", "disputed", "unverifiable"), events=None):
    events = events or [f"event-{i}" for i in range(len(decisions))]
    gold = {"cases": [
        {"id": f"case-{i}", "event_id": events[i], "assessment_mode": "evidence",
         "decision": decision, "origins": [f"root-{i}"],
         "edges": [[f"leaf-{i}", f"root-{i}", "cites"]],
         "origin_evaluable": True, "edges_evaluable": True,
         "annotation_status": "provisional_ai_review_required"}
        for i, decision in enumerate(decisions)]}
    rows = [{"id": case["id"], "arm": arm, "assessment_mode": "evidence",
             "status": "completed", "prediction": {
                 "decision": case["decision"], "origins": case["origins"][:],
                 "edges": deepcopy(case["edges"]), "provenance_evaluable": True},
             "usage": {"model_calls": 1, "input_tokens": 20, "output_tokens": 5, "seconds": 0.5}}
            for case in gold["cases"] for arm in ARMS]
    return gold, rows


def row(rows, case, arm):
    return next(item for item in rows if item["id"] == f"case-{case}" and item["arm"] == arm)


class ProofScoreTests(unittest.TestCase):
    def test_fixes_breaks_and_exact_test(self):
        gold, rows = fixture()
        row(rows, 0, "single")["prediction"]["decision"] = "false"
        row(rows, 1, "single")["prediction"]["decision"] = "true"
        row(rows, 2, "loop_psi")["prediction"]["decision"] = "true"
        report = score(gold, rows, bootstrap_samples=200)
        comparison = report["contrasts"]["loop_psi_vs_single"]
        self.assertEqual(comparison["case_task_success_pairs"]["fixes"], 2)
        self.assertEqual(comparison["case_task_success_pairs"]["breaks"], 1)
        self.assertEqual(comparison["overall_task_success_delta"]["delta"], 0.25)
        self.assertAlmostEqual(comparison["exact_mcnemar"]["two_sided_p"], 1)
        self.assertAlmostEqual(exact_mcnemar(6, 0), 0.03125)
        self.assertAlmostEqual(exact_mcnemar(0, 6), 0.03125)
        self.assertEqual(exact_mcnemar(0, 0), 1)
        self.assertEqual(comparison["role"], "preregistered_primary")
        self.assertEqual(report["contrasts"]["loop_psi_vs_independent"]["role"],
                         "preregistered_exploratory_secondary")

    def test_errors_not_semantic_false_or_unknown_and_fixed_denominator(self):
        gold, rows = fixture()
        # Even a stale correct prediction on an error row must never count.
        row(rows, 0, "single")["status"] = "error"
        row(rows, 0, "single")["error_type"] = "timeout"
        report = score(gold, rows, bootstrap_samples=20)
        result = report["arms"]["single"]
        self.assertEqual(result["scheduled_cases"], 4)
        self.assertEqual(result["completed"], 3)
        self.assertEqual(result["execution_errors"], 1)
        self.assertEqual(result["semantic_errors_among_completed"], 0)
        self.assertEqual(result["overall_task_success"], 0.75)
        self.assertEqual(result["label_accuracy_among_completed"], 1)
        self.assertEqual(result["unverifiable_predictions"], 1)
        self.assertEqual(result["unverifiable_rate_among_completed"], 1 / 3)
        self.assertEqual(result["nonabstaining_coverage_of_scheduled"], 0.5)
        self.assertEqual(result["error_predictions_ignored"], 1)
        contrast = report["contrasts"]["loop_psi_vs_single"]
        self.assertEqual(contrast["completion_recoveries"], 1)
        self.assertEqual(contrast["paired_completed_semantic_pairs"]["fixes"], 0)
        self.assertEqual(contrast["paired_completed_label_accuracy_delta"]["case_pairs"], 3)

    def test_all_errors_are_not_all_unverifiable(self):
        gold, rows = fixture()
        for item in rows:
            if item["arm"] == "original":
                item.update(status="error", prediction=None)
        result = score(gold, rows, bootstrap_samples=20)["arms"]["original"]
        self.assertEqual(result["overall_task_success"], 0)
        self.assertIsNone(result["label_accuracy_among_completed"])
        self.assertEqual(result["unverifiable_predictions"], 0)
        self.assertIsNone(result["unverifiable_rate_among_completed"])
        self.assertEqual(result["nonabstaining_coverage_of_scheduled"], 0)

    def test_repeated_events_do_not_inflate_exact_test(self):
        gold, rows = fixture(("true",) * 6, events=["a", "a", "a", "b", "b", "b"])
        for i in range(6):
            row(rows, i, "single")["prediction"]["decision"] = "false"
        comparison = score(gold, rows, bootstrap_samples=50)["contrasts"]["loop_psi_vs_single"]
        self.assertEqual(comparison["case_task_success_pairs"]["fixes"], 6)
        self.assertEqual(comparison["event_binary_pairs"]["fixes"], 2)
        self.assertEqual(comparison["independent_events"], 2)
        self.assertAlmostEqual(comparison["exact_mcnemar"]["two_sided_p"], 0.5)
        self.assertEqual(comparison["overall_task_success_delta"]["independent_events"], 2)
        self.assertIsNone(comparison["overall_task_success_delta"]["ci95"])

    def test_cluster_bootstrap_retains_within_event_correlation(self):
        gold, rows = fixture(("true",) * 4, events=["a", "a", "b", "b"])
        for i in (0, 1):
            row(rows, i, "single")["prediction"]["decision"] = "false"
        comparison = score(gold, rows, bootstrap_samples=1000)["contrasts"]["loop_psi_vs_single"]
        interval = comparison["overall_task_success_delta"]
        self.assertEqual(interval["delta"], 0.5)
        self.assertEqual(interval["ci95"], [0, 1])
        self.assertEqual(interval["independent_events"], 2)

    def test_no_discordance_has_no_fake_precision(self):
        gold, rows = fixture()
        report = score(gold, rows, bootstrap_samples=20)
        comparison = report["contrasts"]["loop_psi_vs_single"]
        self.assertTrue(comparison["exact_mcnemar"]["zero_discordance"])
        self.assertEqual(comparison["exact_mcnemar"]["two_sided_p"], 1)
        self.assertTrue(comparison["overall_task_success_delta"]["degenerate"])
        self.assertIsNone(comparison["overall_task_success_delta"]["ci95"])
        self.assertTrue(any("not evidence of equivalence" in note for note in comparison["cautions"]))
        self.assertEqual(report["conclusion"], "pilot_only_no_proof_of_realworld_accuracy_improvement")
        self.assertIn("not_proof", report["power_planning"]["status"])

    def test_one_event_with_many_probes_has_no_ci(self):
        gold, rows = fixture(events=["one"] * 4)
        row(rows, 0, "single")["prediction"]["decision"] = "false"
        interval = score(gold, rows, bootstrap_samples=20)["contrasts"]["loop_psi_vs_single"]["overall_task_success_delta"]
        self.assertEqual(interval["reason"], "fewer_than_two_independent_events")
        self.assertIsNone(interval["ci95"])

    def test_partial_graph_evaluation_and_alternative_origins(self):
        gold, rows = fixture()
        gold["cases"][0]["origins"].append("alternative-root")
        gold["cases"][1]["origin_evaluable"] = False
        gold["cases"][2]["edges_evaluable"] = False
        row(rows, 3, "original")["prediction"]["provenance_evaluable"] = False
        row(rows, 0, "original")["prediction"]["origins"].append("wrong-root")
        row(rows, 1, "original")["prediction"]["edges"] = []
        result = score(gold, rows, bootstrap_samples=20)["arms"]["original"]
        self.assertEqual(result["overall_task_success"], 1)
        self.assertEqual(result["origins"]["evaluable_cases"], 2)
        self.assertAlmostEqual(result["origins"]["precision"], 2 / 3)
        self.assertEqual(result["origins"]["recall"], 1)
        self.assertEqual(result["edges"]["evaluable_cases"], 2)
        self.assertEqual(result["edges"]["precision"], 1)
        self.assertEqual(result["edges"]["recall"], 0.5)
        self.assertEqual(result["origins"]["exclusions"]["prediction_not_evaluable"], 1)
        self.assertEqual(result["edges"]["exclusions"]["gold_not_evaluable_or_unknown"], 1)

    def test_unknown_incomplete_gold_excluded_explicitly(self):
        gold, rows = fixture()
        gold["cases"][0]["decision"] = "unknown"
        gold["cases"][0]["origins"] = None
        gold["cases"][1]["edges"] = None
        result = score(gold, rows, bootstrap_samples=20)["arms"]["single"]
        self.assertEqual(result["label_gold_excluded"], 1)
        self.assertEqual(result["overall_task_success_denominator"], 3)
        self.assertEqual(result["overall_task_success"], 1)
        self.assertEqual(result["origins"]["exclusions"]["gold_incomplete"], 1)
        self.assertEqual(result["edges"]["exclusions"]["gold_incomplete"], 1)

    def test_dimension_overrides_allow_original_roots_without_fake_edges(self):
        gold, rows = fixture()
        for item in rows:
            if item["arm"] == "original":
                item["prediction"].update(provenance_evaluable=False,
                                          origin_evaluable=True, edges_evaluable=False,
                                          edges=[])
        result = score(gold, rows, bootstrap_samples=20)["arms"]["original"]
        self.assertEqual(result["overall_task_success"], 1)
        self.assertEqual(result["origins"]["evaluable_cases"], 4)
        self.assertEqual(result["origins"]["recall"], 1)
        self.assertEqual(result["origins"]["metric"], "corpus-relative documentary-root accuracy")
        self.assertEqual(result["edges"]["evaluable_cases"], 0)
        self.assertIsNone(result["edges"]["precision"])
        self.assertIsNone(result["edges"]["recall"])
        self.assertEqual(result["edges"]["exclusions"], {"prediction_dimension_not_evaluable": 4})

    def test_strict_matrix_validation(self):
        gold, rows = fixture()
        for changed, message in ((rows[:-1], "missing result rows"),
                                 (rows + [deepcopy(rows[0])], "duplicate result row")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                score(gold, changed, bootstrap_samples=20)
        for key, value, message in (("arm", "best_after_selection", "unknown arm"),
                                    ("id", "missing", "unknown result case"),
                                    ("assessment_mode", "snapshot", "unknown assessment_mode"),
                                    ("assessment_mode", "world", "assessment_mode mismatch or missing")):
            changed = deepcopy(rows)
            changed[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, message):
                score(gold, changed, bootstrap_samples=20)
        changed_gold = deepcopy(gold)
        changed_gold["cases"].append(deepcopy(gold["cases"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate gold id"):
            score(changed_gold, rows, bootstrap_samples=20)

    def test_heterogeneous_case_modes_match_per_case_and_preserve_event_clusters(self):
        gold, rows = fixture(events=["a", "a", "b", "b"])
        gold["cases"][0]["assessment_mode"] = "world"
        for item in rows:
            if item["id"] == "case-0":
                item["assessment_mode"] = "world"
        report = score(gold, rows, bootstrap_samples=20)
        self.assertEqual(report["assessment_mode"], "mixed")
        self.assertEqual(report["assessment_mode_case_counts"], {"evidence": 3, "world": 1})
        self.assertEqual(report["independent_events"], 2)
        self.assertEqual(report["arms"]["single"]["overall_task_success_denominator"], 4)
        contrast = report["contrasts"]["loop_psi_vs_single"]
        self.assertEqual(contrast["exact_mcnemar"]["independent_event_pairs"], 2)
        self.assertEqual(contrast["overall_task_success_delta"]["independent_events"], 2)
        row(rows, 0, "single")["assessment_mode"] = "evidence"
        with self.assertRaisesRegex(ValueError, "assessment_mode mismatch or missing"):
            score(gold, rows, bootstrap_samples=20)

    def test_partial_mode_omission_rejected_but_all_omitted_remains_supported(self):
        gold, rows = fixture()
        del gold["cases"][0]["assessment_mode"]
        for item in rows:
            if item["id"] == "case-0":
                del item["assessment_mode"]
        with self.assertRaisesRegex(ValueError, "mixed or missing assessment_mode in gold"):
            score(gold, rows, bootstrap_samples=20)
        for case in gold["cases"]:
            case.pop("assessment_mode", None)
        for item in rows:
            item.pop("assessment_mode", None)
        report = score(gold, rows, bootstrap_samples=20)
        self.assertIsNone(report["assessment_mode"])
        self.assertEqual(report["assessment_mode_case_counts"], {"unspecified": 4})

    def test_missing_mode_and_invalid_prediction_rejected(self):
        gold, rows = fixture()
        del rows[0]["assessment_mode"]
        with self.assertRaisesRegex(ValueError, "assessment_mode mismatch or missing"):
            score(gold, rows, bootstrap_samples=20)
        gold, rows = fixture()
        rows[0]["prediction"]["decision"] = "error"
        with self.assertRaisesRegex(ValueError, "unknown or missing prediction decision"):
            score(gold, rows, bootstrap_samples=20)

    def test_usage_includes_errors_and_rejects_invalid_numbers(self):
        gold, rows = fixture()
        row(rows, 0, "original").update(status="error", prediction=None)
        result = score(gold, rows, bootstrap_samples=20)["arms"]["original"]
        self.assertEqual(result["usage_total"]["model_calls"], 4)
        self.assertEqual(result["usage_total"]["input_tokens"], 80)
        for value in (float("nan"), -1, True):
            changed = deepcopy(rows)
            changed[0]["usage"]["seconds"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                score(gold, changed, bootstrap_samples=20)

    def test_power_planning_is_assumption_based_and_monotonic(self):
        planning = power_planning()
        self.assertEqual(planning["minimum_worthwhile_delta"], 0.03)
        self.assertEqual(planning["two_sided_alpha"], 0.05)
        self.assertEqual(planning["target_power"], 0.8)
        ns = [item["approximate_independent_events"] for item in planning["scenarios"]]
        self.assertEqual(ns, sorted(ns))
        self.assertGreater(ns[0], 200)
        self.assertGreater(ns[-1], 3000)
        with self.assertRaises(ValueError):
            power_planning(discordance_scenarios=(0.01,))

    def test_seed_reproducibility(self):
        gold, rows = fixture()
        row(rows, 0, "single")["prediction"]["decision"] = "false"
        a = score(gold, rows, bootstrap_samples=200, seed=13)
        self.assertEqual(a, score(gold, rows, bootstrap_samples=200, seed=13))


if __name__ == "__main__":
    unittest.main()
