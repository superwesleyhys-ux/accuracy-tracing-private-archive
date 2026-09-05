"""Offline accounting correction for failed shared prefixes; never reruns inference."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

if __package__:
    from .proof_score import score
else:
    from proof_score import score


USAGE_FIELDS = ("model_calls", "input_tokens", "output_tokens", "seconds")
ACTUAL_FIELDS = USAGE_FIELDS[:3]
SHARED_ARMS = frozenset(("loop_psi", "loop_frozen", "independent"))
ADJUSTMENT_KIND = "inherit_failed_single_prefix_logical_usage_v1"


def _usage(value, fields, context):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"{context}: expected exactly {fields}")
    for key in fields:
        number = value[key]
        if (isinstance(number, bool) or not isinstance(number, (int, float))
                or not math.isfinite(number) or number < 0
                or (key != "seconds" and number != int(number))):
            raise ValueError(f"{context}: invalid {key}")
    return value


def _totals(rows, field, keys):
    return {key: sum(row[field][key] for row in rows) for key in keys}


def reconcile_costs(results):
    """Return a new matrix and audit; assign inherited cost once instead of adding it.

    Only InvalidSharedPrefix error rows in shared-prefix arms may change. Their
    logical usage must be zero or already exactly equal to the matching failed
    single run. Actual new API usage must remain zero on those rows.
    """
    if not isinstance(results, list):
        raise ValueError("results must be an array")
    index = {}
    for row in results:
        if not isinstance(row, dict):
            raise ValueError("result rows must be objects")
        key = (row.get("id"), row.get("arm"))
        if not all(isinstance(value, str) and value for value in key):
            raise ValueError("every row requires nonempty id and arm")
        if key in index:
            raise ValueError(f"duplicate result row: {key}")
        _usage(row.get("usage"), USAGE_FIELDS, f"{key} usage")
        _usage(row.get("actual_new_api_usage"), ACTUAL_FIELDS, f"{key} actual_new_api_usage")
        index[key] = row
    corrected = deepcopy(results)
    changed, unchanged = [], []
    for row in corrected:
        key = (row["id"], row["arm"])
        if row.get("error_type") != "InvalidSharedPrefix":
            if "accounting_adjustment" in row:
                raise ValueError(f"{key}: accounting marker on an ineligible row")
            continue
        if row["arm"] not in SHARED_ARMS or row.get("status") != "error":
            raise ValueError(f"{key}: InvalidSharedPrefix must be a shared-arm error")
        if any(row["actual_new_api_usage"].values()):
            raise ValueError(f"{key}: failed shared prefix has nonzero actual new usage")
        source = index.get((row["id"], "single"))
        if source is None or source.get("status") != "error":
            raise ValueError(f"{key}: matching failed single row required")
        if "prefix_usage" in source:
            raise ValueError(f"{key}: donor has a separate prefix_usage; full failed-single charge is ambiguous")
        if row.get("assessment_mode") != source.get("assessment_mode"):
            raise ValueError(f"{key}: single and shared-arm assessment modes disagree")
        expected = source["usage"]
        marker = {"kind": ADJUSTMENT_KIND, "source_id": row["id"],
                  "source_arm": "single", "inherited_usage": deepcopy(expected)}
        prior_marker = row.get("accounting_adjustment")
        if prior_marker is not None:
            if prior_marker != marker or row["usage"] != expected:
                raise ValueError(f"{key}: inconsistent prior accounting adjustment")
            unchanged.append({"id": row["id"], "arm": row["arm"]})
            continue
        if any(row["usage"].values()) and row["usage"] != expected:
            raise ValueError(f"{key}: prior logical usage must be zero or exactly inherited")
        changed.append({"id": row["id"], "arm": row["arm"],
                        "source_id": row["id"], "source_arm": "single",
                        "usage_before": deepcopy(row["usage"]),
                        "usage_after": deepcopy(expected)})
        row["usage"] = deepcopy(expected)
        row["accounting_adjustment"] = marker
    # A future edit cannot silently extend the correction to outcomes or API spend.
    for original, adjusted in zip(results, corrected):
        a, b = deepcopy(original), deepcopy(adjusted)
        for item in (a, b):
            item.pop("usage", None)
            item.pop("accounting_adjustment", None)
        if a != b:
            raise AssertionError("accounting correction changed a non-accounting row field")
    actual_before = _totals(results, "actual_new_api_usage", ACTUAL_FIELDS)
    actual_after = _totals(corrected, "actual_new_api_usage", ACTUAL_FIELDS)
    if actual_before != actual_after:
        raise AssertionError("actual API usage changed")
    audit = {
        "schema_version": "proof-cost-reconciliation-v1", "adjustment_kind": ADJUSTMENT_KIND,
        "changed_rows": changed, "changed_row_count": len(changed),
        "already_adjusted_rows": unchanged,
        "logical_usage_before": _totals(results, "usage", USAGE_FIELDS),
        "logical_usage_after": _totals(corrected, "usage", USAGE_FIELDS),
        "actual_new_api_usage_before": actual_before, "actual_new_api_usage_after": actual_after,
        "outcomes_unchanged": True,
        "note": "Each invalid shared-prefix candidate inherits its matching failed single run's logical cost. This allocates existing prefix work and does not represent additional API calls.",
        "seconds_definition": "logical charged time summed across arms, not elapsed experiment runtime",
    }
    return corrected, audit


def assert_scores_only_usage_changed(raw_scores, corrected_scores):
    """Reject any change outside the two per-arm usage summaries."""
    a, b = deepcopy(raw_scores), deepcopy(corrected_scores)
    for report in (a, b):
        for arm in report["arms"].values():
            arm.pop("usage_total", None)
            arm.pop("usage_mean_per_scheduled_case", None)
    if a != b:
        raise ValueError("rescoring changed non-usage metrics")


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("results", "output", "audit", "gold", "raw-scores", "scores-output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--expect-logical-calls", type=int)
    parser.add_argument("--expect-actual-calls", type=int)
    args = parser.parse_args(argv)
    try:
        inputs = (args.results, args.gold, args.raw_scores)
        outputs = (args.output, args.audit, args.scores_output)
        paths = [path.resolve() for path in inputs + outputs]
        if len(set(paths)) != len(paths):
            raise ValueError("input and output paths must all be distinct; original files cannot be overwritten")
        input_bytes = {path: path.read_bytes() for path in inputs}
        corrected, audit = reconcile_costs(json.loads(input_bytes[args.results]))
        for expected, actual, label in (
                (args.expect_logical_calls, audit["logical_usage_after"]["model_calls"], "logical"),
                (args.expect_actual_calls, audit["actual_new_api_usage_after"]["model_calls"], "actual")):
            if expected is not None and actual != expected:
                raise ValueError(f"expected {expected} {label} calls, got {actual}")
        raw_scores = json.loads(input_bytes[args.raw_scores])
        config = raw_scores["contrasts"]["loop_psi_vs_single"]["overall_task_success_delta"]
        corrected_scores = score(json.loads(input_bytes[args.gold]), corrected,
                                 bootstrap_samples=config["bootstrap_samples"], seed=config["seed"])
        assert_scores_only_usage_changed(raw_scores, corrected_scores)
        corrected_bytes, scored_bytes = _json_bytes(corrected), _json_bytes(corrected_scores)
        audit["score_comparison"] = "all_non_usage_fields_identical"
        audit["files"] = {
            "raw_results": {"path": str(args.results), "sha256": _sha(input_bytes[args.results])},
            "corrected_results": {"path": str(args.output), "sha256": _sha(corrected_bytes)},
            "gold": {"path": str(args.gold), "sha256": _sha(input_bytes[args.gold])},
            "raw_scores": {"path": str(args.raw_scores), "sha256": _sha(input_bytes[args.raw_scores])},
            "corrected_scores": {"path": str(args.scores_output), "sha256": _sha(scored_bytes)},
            "reconciliation_code": {"path": str(Path(__file__)), "sha256": _sha(Path(__file__).read_bytes())},
            "scoring_code": {"path": str(Path(__file__).with_name("proof_score.py")),
                             "sha256": _sha(Path(__file__).with_name("proof_score.py").read_bytes())},
        }
        if any(path.read_bytes() != data for path, data in input_bytes.items()):
            raise ValueError("an input changed during reconciliation")
        args.output.write_bytes(corrected_bytes)
        args.scores_output.write_bytes(scored_bytes)
        args.audit.write_bytes(_json_bytes(audit))
        if any(path.read_bytes() != data for path, data in input_bytes.items()):
            raise AssertionError("an original input changed")
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps({"changed_rows": audit["changed_row_count"],
                      "logical_calls_before": audit["logical_usage_before"]["model_calls"],
                      "logical_calls_after": audit["logical_usage_after"]["model_calls"],
                      "actual_calls": audit["actual_new_api_usage_after"]["model_calls"],
                      "score_comparison": audit["score_comparison"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
