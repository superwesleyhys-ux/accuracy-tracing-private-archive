from copy import deepcopy
import unittest

from experiments.proof_reconcile_costs import (
    assert_scores_only_usage_changed, reconcile_costs,
)


def fixture():
    usage = {"model_calls": 2, "input_tokens": 40, "output_tokens": 12, "seconds": 3.5}
    actual = {key: usage[key] for key in ("model_calls", "input_tokens", "output_tokens")}
    source = {"id": "case-a", "arm": "single", "status": "error", "prediction": None,
              "assessment_mode": "evidence", "error_type": "ValueError",
              "usage": usage, "actual_new_api_usage": actual}
    rows = [source]
    for arm in ("loop_psi", "loop_frozen", "independent"):
        rows.append({"id": "case-a", "arm": arm, "status": "error", "prediction": None,
                     "assessment_mode": "evidence", "error_type": "InvalidSharedPrefix",
                     "usage": {key: 0 for key in usage},
                     "actual_new_api_usage": {key: 0 for key in actual}})
    return rows


class ProofCostTests(unittest.TestCase):
    def test_inherits_only_logical_prefix_cost_and_preserves_outcomes_and_actual_usage(self):
        rows = fixture()
        original = deepcopy(rows)
        corrected, audit = reconcile_costs(rows)
        self.assertEqual(rows, original)
        self.assertEqual(audit["changed_row_count"], 3)
        self.assertEqual(audit["logical_usage_before"]["model_calls"], 2)
        self.assertEqual(audit["logical_usage_after"]["model_calls"], 8)
        self.assertEqual(audit["actual_new_api_usage_before"], audit["actual_new_api_usage_after"])
        self.assertEqual(audit["actual_new_api_usage_after"]["model_calls"], 2)
        self.assertEqual(corrected[0], rows[0])
        for before, after in zip(rows[1:], corrected[1:]):
            self.assertEqual(after["usage"], rows[0]["usage"])
            for key in before.keys() - {"usage"}:
                self.assertEqual(before[key], after[key])
        raw_scores = {"arms": {"single": {"correct": 0, "usage_total": {"model_calls": 2},
                                           "usage_mean_per_scheduled_case": {"model_calls": 2}}}}
        changed_scores = deepcopy(raw_scores)
        changed_scores["arms"]["single"]["usage_total"]["model_calls"] = 8
        assert_scores_only_usage_changed(raw_scores, changed_scores)
        changed_scores["arms"]["single"]["correct"] = 1
        with self.assertRaisesRegex(ValueError, "non-usage metrics"):
            assert_scores_only_usage_changed(raw_scores, changed_scores)

    def test_idempotent_and_already_inherited_cost_is_not_added_again(self):
        rows = fixture()
        rows[1]["usage"] = deepcopy(rows[0]["usage"])
        corrected, first_audit = reconcile_costs(rows)
        repeated, repeated_audit = reconcile_costs(corrected)
        self.assertEqual(corrected, repeated)
        self.assertEqual(first_audit["logical_usage_after"]["model_calls"], 8)
        self.assertEqual(repeated_audit["changed_row_count"], 0)
        self.assertEqual(len(repeated_audit["already_adjusted_rows"]), 3)
        self.assertEqual(repeated_audit["logical_usage_before"], repeated_audit["logical_usage_after"])

    def test_rejects_ambiguous_or_inconsistent_accounting(self):
        invalid = []
        changed = fixture(); changed[1]["usage"]["model_calls"] = 1; invalid.append(changed)
        changed = fixture(); changed[1]["actual_new_api_usage"]["model_calls"] = 1; invalid.append(changed)
        changed = fixture(); changed[0]["status"] = "completed"; invalid.append(changed)
        changed = fixture(); changed[1]["status"] = "completed"; invalid.append(changed)
        changed = fixture(); changed[1]["id"] = "wrong-case"; invalid.append(changed)
        changed = fixture(); changed.append(deepcopy(changed[0])); invalid.append(changed)
        changed = fixture(); changed[0]["prefix_usage"] = deepcopy(changed[0]["usage"]); invalid.append(changed)
        changed, _ = reconcile_costs(fixture())
        changed[1]["usage"]["model_calls"] = 0; invalid.append(changed)
        for index, rows in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(ValueError):
                reconcile_costs(rows)


if __name__ == "__main__":
    unittest.main()
