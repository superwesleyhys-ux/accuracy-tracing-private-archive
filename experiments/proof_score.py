"""Offline, fixed-denominator scoring for the preregistered five-arm pilot.

This module never imports an inference adapter or calls a model.  Gold is supplied
only to this separate scoring process.  Missing runs must be recorded as explicit
error rows; a missing matrix cell is a malformed experiment, not an abstention.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
from statistics import NormalDist


ARMS = ("original", "single", "loop_psi", "loop_frozen", "independent")
DECISIONS = frozenset(("true", "false", "disputed", "unverifiable"))
MODES = frozenset(("evidence", "world"))
USAGE_FIELDS = ("model_calls", "input_tokens", "output_tokens", "seconds")


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _ident(value, context):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a nonempty string")
    return value


def _origins(value, context):
    if not isinstance(value, list):
        raise ValueError(f"{context} must be an array")
    values = [_ident(item, context) for item in value]
    if len(set(values)) != len(values):
        raise ValueError(f"{context} contains duplicate origins")
    return set(values)


def _edges(value, context):
    if not isinstance(value, list):
        raise ValueError(f"{context} must be an array")
    result = set()
    for edge in value:
        if not isinstance(edge, (list, tuple)) or len(edge) != 3:
            raise ValueError(f"{context}: edges must be [from_version, to_version, kind]")
        a, b, kind = edge
        _ident(a, context)
        _ident(b, context)
        if kind not in ("cites", "quotes"):
            raise ValueError(f"{context}: unknown edge kind {kind!r}")
        item = (a, b, kind)
        if item in result:
            raise ValueError(f"{context} contains duplicate edges")
        result.add(item)
    return result


def _mode(item, default=None):
    # target_mode is accepted only as an older spelling of assessment_mode.
    if "target_mode" in item and "assessment_mode" in item:
        if item["target_mode"] != item["assessment_mode"]:
            raise ValueError("conflicting target_mode and assessment_mode")
    value = item.get("assessment_mode", item.get("target_mode", default))
    if value is not None and value not in MODES:
        raise ValueError(f"unknown assessment_mode {value!r}; expected evidence or world")
    return value


def _validate(gold, results, arms):
    if not isinstance(gold, dict) or not isinstance(gold.get("cases"), list):
        raise ValueError("gold must contain a cases array")
    if not gold["cases"]:
        raise ValueError("gold cases cannot be empty")
    if not isinstance(results, list):
        raise ValueError("results must be a JSON array")
    if tuple(arms) != ARMS:
        raise ValueError(f"the preregistered arms must be {ARMS}")
    default_mode = _mode(gold)
    cases, modes = {}, set()
    for case in gold["cases"]:
        if not isinstance(case, dict):
            raise ValueError("gold cases must be objects")
        case_id = _ident(case.get("id"), "gold id")
        if case_id in cases:
            raise ValueError(f"duplicate gold id: {case_id}")
        _ident(case.get("event_id"), f"gold {case_id} event_id")
        if "annotation_status" in case:
            _ident(case["annotation_status"], f"gold {case_id} annotation_status")
        decision = case.get("decision")
        if decision is not None and decision not in DECISIONS and decision != "unknown":
            raise ValueError(f"gold {case_id}: unknown decision {decision!r}")
        for flag in ("origin_evaluable", "edges_evaluable", "label_evaluable"):
            if flag in case and not isinstance(case[flag], bool):
                raise ValueError(f"gold {case_id}: {flag} must be boolean")
        for key, parser in (("origins", _origins), ("edges", _edges)):
            if case.get(key) is not None:
                parser(case[key], f"gold {case_id} {key}")
        modes.add(_mode(case, default_mode))
        cases[case_id] = case
    matrix = {}
    for row in results:
        if not isinstance(row, dict):
            raise ValueError("results rows must be objects")
        case_id, arm = row.get("id"), row.get("arm")
        if case_id not in cases:
            raise ValueError(f"unknown result case id: {case_id!r}")
        if arm not in arms:
            raise ValueError(f"unknown arm: {arm!r}")
        if "event_id" in row and row["event_id"] != cases[case_id]["event_id"]:
            raise ValueError(f"{case_id}: result event_id disagrees with gold")
        key = (case_id, arm)
        if key in matrix:
            raise ValueError(f"duplicate result row: {key}")
        modes.add(_mode(row))
        if row.get("status") not in ("completed", "error"):
            raise ValueError(f"{key}: status must be completed or error")
        if row["status"] == "completed":
            prediction = row.get("prediction")
            if not isinstance(prediction, dict):
                raise ValueError(f"{key}: completed row requires prediction object")
            if prediction.get("decision") not in DECISIONS:
                raise ValueError(f"{key}: unknown or missing prediction decision")
            if not isinstance(prediction.get("provenance_evaluable"), bool):
                raise ValueError(f"{key}: provenance_evaluable must be boolean")
            for flag in ("origin_evaluable", "edges_evaluable"):
                if flag in prediction and not isinstance(prediction[flag], bool):
                    raise ValueError(f"{key}: prediction {flag} must be boolean")
            _origins(prediction.get("origins"), f"{key} predicted origins")
            _edges(prediction.get("edges"), f"{key} predicted edges")
        usage = row.get("usage")
        if not isinstance(usage, dict):
            raise ValueError(f"{key}: usage is required even for errors")
        for field in USAGE_FIELDS:
            value = usage.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key}: usage.{field} must be a nonnegative number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key}: usage.{field} must be finite and nonnegative")
            if field != "seconds" and int(value) != value:
                raise ValueError(f"{key}: usage.{field} must be an integer")
        matrix[key] = row
    if len(modes) > 1:
        raise ValueError("mixed or missing assessment_mode; gold and every row must agree")
    missing = [(case_id, arm) for case_id in cases for arm in arms
               if (case_id, arm) not in matrix]
    if missing:
        raise ValueError(f"missing result rows (record failures as status=error): {missing}")
    return cases, matrix, next(iter(modes))


def exact_mcnemar(fixes, breaks):
    """Two-sided exact binomial test conditional on discordant binary pairs."""
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0
           for n in (fixes, breaks)):
        raise ValueError("fixes and breaks must be nonnegative integers")
    discordant = fixes + breaks
    if not discordant:
        return 1.0
    k = min(fixes, breaks)
    log_terms = [math.lgamma(discordant + 1) - math.lgamma(i + 1)
                 - math.lgamma(discordant - i + 1) - discordant * math.log(2)
                 for i in range(k + 1)]
    largest = max(log_terms)
    return min(1.0, 2 * math.exp(largest) * sum(math.exp(x - largest) for x in log_terms))


def _quantile(values, probability):
    position = (len(values) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def event_cluster_bootstrap(pairs, samples=10000, seed=2026):
    """Pairs are (event_id, baseline_binary, candidate_binary).

    Resample events with replacement, retaining all cases in each selected event.
    The estimand is the case-weighted accuracy delta. Constant intervals are
    withheld: a small empirical distribution cannot demonstrate zero uncertainty.
    """
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    grouped = defaultdict(list)
    for event_id, baseline, candidate in pairs:
        grouped[event_id].append(int(candidate) - int(baseline))
    clusters = [(sum(values), len(values)) for values in grouped.values()]
    delta = _ratio(sum(total for total, _ in clusters), len(pairs))
    report = {"delta": delta, "ci95": None, "independent_events": len(clusters),
              "case_pairs": len(pairs), "bootstrap_samples": samples, "seed": seed,
              "degenerate": False, "reason": None,
              "method": "percentile bootstrap resampling whole event clusters"}
    if len(clusters) < 2:
        report.update(degenerate=True, reason="fewer_than_two_independent_events")
        return report
    if not any(base != candidate for _, base, candidate in pairs):
        report.update(degenerate=True, reason="zero_discordant_case_pairs")
        return report
    if len({total / count for total, count in clusters}) == 1:
        report.update(degenerate=True, reason="constant_event_cluster_deltas")
        return report
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        selected = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        draws.append(sum(total for total, _ in selected) / sum(n for _, n in selected))
    draws.sort()
    interval = [_quantile(draws, 0.025), _quantile(draws, 0.975)]
    if interval[0] == interval[1]:
        report.update(degenerate=True, reason="constant_bootstrap_percentile_interval")
    else:
        report["ci95"] = interval
    return report


def power_planning(delta=0.03, alpha=0.05, power=0.8,
                   discordance_scenarios=(0.03, 0.05, 0.10, 0.20, 0.40)):
    """Normal approximation for a future paired binary, independent-event study."""
    if not (0 < delta < 1 and 0 < alpha < 1 and 0.5 < power < 1):
        raise ValueError("invalid power-planning inputs")
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_power = NormalDist().inv_cdf(power)
    scenarios = []
    for q in discordance_scenarios:
        if not delta <= q <= 1:
            raise ValueError("assumed discordance must lie between delta and 1")
        n = math.ceil((z_alpha * math.sqrt(q)
                       + z_power * math.sqrt(q - delta * delta)) ** 2 / (delta * delta))
        scenarios.append({"assumed_discordance": q,
                          "assumed_fix_probability": (q + delta) / 2,
                          "assumed_break_probability": (q - delta) / 2,
                          "approximate_independent_events": n})
    return {"minimum_worthwhile_delta": delta, "two_sided_alpha": alpha,
            "target_power": power, "scenarios": scenarios,
            "formula": "ceil((z_(1-alpha/2)*sqrt(q) + z_power*sqrt(q-delta^2))^2/delta^2)",
            "status": "planning_approximations_not_proof_or_a_pilot_fitted_sample_size",
            "notes": [
                "q is an assumed probability of discordant paired event outcomes, not estimated from this pilot.",
                "The estimates omit continuity correction, attrition, and multiplicity adjustments; validate with exact calculations or simulation before a confirmatory study.",
                "Units are independent events. Repeated probes do not create new independent events; a clustered case-level design needs separate planning.",
                "Zero observed discordance gives no empirical sample-size estimate and does not establish equivalence.",
            ]}


def _label_evaluable(case):
    return case.get("decision") in DECISIONS and case.get("label_evaluable", True)


def _graph_score(cases, matrix, arm, dimension):
    is_origin = dimension == "origins"
    flag = "origin_evaluable" if is_origin else "edges_evaluable"
    parser = _origins if is_origin else _edges
    excluded = Counter()
    evaluated, predicted_count, reference_count, matched, hit_cases = 0, 0, 0, 0, 0
    for case_id, case in cases.items():
        reference = case.get(dimension)
        if not case.get(flag, False):
            excluded["gold_not_evaluable_or_unknown"] += 1
            continue
        if reference is None or (is_origin and not reference):
            excluded["gold_incomplete"] += 1
            continue
        row = matrix[(case_id, arm)]
        if row["status"] != "completed":
            excluded["execution_error"] += 1
            continue
        prediction = row["prediction"]
        if not prediction.get(flag, prediction["provenance_evaluable"]):
            reason = ("prediction_dimension_not_evaluable" if flag in prediction
                      else "prediction_not_evaluable")
            excluded[reason] += 1
            continue
        expected = parser(reference, "gold graph")
        actual = parser(prediction[dimension], "predicted graph")
        intersection = len(expected & actual)
        evaluated += 1
        predicted_count += len(actual)
        reference_count += len(expected)
        matched += intersection
        hit_cases += bool(intersection)
    result = {"evaluable_cases": evaluated, "excluded_cases": sum(excluded.values()),
              "exclusions": dict(sorted(excluded.items())),
              "matched_predictions": matched, "predicted_items": predicted_count,
              "precision": _ratio(matched, predicted_count),
              "scope": "corpus-relative, restricted to explicitly evaluable material"}
    if is_origin:
        result.update(metric="corpus-relative documentary-root accuracy",
                      recall=_ratio(hit_cases, evaluated), cases_with_acceptable_origin=hit_cases,
                      recall_denominator=evaluated,
                      recall_definition="fraction of evaluable cases with at least one acceptable origin; gold origins are alternatives")
    else:
        result.update(recall=_ratio(matched, reference_count), gold_edges=reference_count,
                      recall_denominator=reference_count,
                      recall_definition="micro recall over known exhaustive excerpt-level edge sets")
    return result


def _arm_score(cases, matrix, arm):
    scheduled = len(cases)
    counts = Counter()
    errors = Counter()
    confusion = Counter()
    usage = {key: 0 for key in USAGE_FIELDS}
    for case_id, case in cases.items():
        row = matrix[(case_id, arm)]
        for key in usage:
            usage[key] += row["usage"][key]
        if row["status"] == "error":
            counts["errors"] += 1
            errors[str(row.get("error_type", "unspecified"))] += 1
            counts["error_predictions_ignored"] += row.get("prediction") is not None
            continue
        counts["completed"] += 1
        prediction = row["prediction"]["decision"]
        counts["unverifiable"] += prediction == "unverifiable"
        counts["nonabstaining"] += prediction != "unverifiable"
        if _label_evaluable(case):
            counts["completed_label_evaluable"] += 1
            correct = prediction == case["decision"]
            counts["correct"] += correct
            counts["semantic_errors"] += not correct
            confusion[(case["decision"], prediction)] += 1
    label_n = sum(_label_evaluable(case) for case in cases.values())
    return {
        "scheduled_cases": scheduled, "completed": counts["completed"],
        "completion_rate": _ratio(counts["completed"], scheduled),
        "execution_errors": counts["errors"], "error_types": dict(sorted(errors.items())),
        "error_predictions_ignored": counts["error_predictions_ignored"],
        "label_evaluable_cases": label_n, "label_gold_excluded": scheduled - label_n,
        "correct": counts["correct"], "semantic_errors_among_completed": counts["semantic_errors"],
        "overall_task_success": _ratio(counts["correct"], label_n),
        "overall_task_success_denominator": label_n,
        "overall_task_success_definition": "correct completed labels / all label-evaluable scheduled cases; execution errors are task failures",
        "label_accuracy_among_completed": _ratio(counts["correct"], counts["completed_label_evaluable"]),
        "label_accuracy_among_completed_denominator": counts["completed_label_evaluable"],
        "unverifiable_predictions": counts["unverifiable"],
        "unverifiable_rate_among_completed": _ratio(counts["unverifiable"], counts["completed"]),
        "nonabstaining_predictions": counts["nonabstaining"],
        "nonabstaining_coverage_of_scheduled": _ratio(counts["nonabstaining"], scheduled),
        "nonabstaining_coverage_among_completed": _ratio(counts["nonabstaining"], counts["completed"]),
        "confusion_among_completed": [{"gold": a, "prediction": b, "count": n}
                                      for (a, b), n in sorted(confusion.items())],
        "origins": _graph_score(cases, matrix, arm, "origins"),
        "edges": _graph_score(cases, matrix, arm, "edges"),
        "usage_total": usage,
        "usage_mean_per_scheduled_case": {key: value / scheduled for key, value in usage.items()},
    }


def _pair_counts(pairs):
    return {"fixes": sum(not base and candidate for _, base, candidate in pairs),
            "breaks": sum(base and not candidate for _, base, candidate in pairs),
            "both_successful": sum(base and candidate for _, base, candidate in pairs),
            "both_unsuccessful": sum(not base and not candidate for _, base, candidate in pairs)}


def _contrast(cases, matrix, baseline, candidate, samples, seed, primary):
    pairs, semantic_pairs = [], []
    completion_recoveries = completion_regressions = 0
    events = defaultdict(list)
    for case_id, case in cases.items():
        if not _label_evaluable(case):
            continue
        a, b = matrix[(case_id, baseline)], matrix[(case_id, candidate)]
        ac, bc = a["status"] == "completed", b["status"] == "completed"
        av = ac and a["prediction"]["decision"] == case["decision"]
        bv = bc and b["prediction"]["decision"] == case["decision"]
        event_id = case["event_id"]
        pairs.append((event_id, av, bv))
        events[event_id].append((av, bv))
        if ac and bc:
            semantic_pairs.append((event_id, av, bv))
        completion_recoveries += not ac and bc
        completion_regressions += ac and not bc
    event_pairs = [(event_id, all(a for a, _ in values), all(b for _, b in values))
                   for event_id, values in events.items()]
    case_counts, event_counts = _pair_counts(pairs), _pair_counts(event_pairs)
    discordant = event_counts["fixes"] + event_counts["breaks"]
    caution = []
    if not discordant:
        caution.append("Zero discordant event outcomes: p=1 is not evidence of equivalence; no pilot-fitted sample-size estimate is available.")
    if len(events) < 30:
        caution.append("Few independent events: bootstrap coverage and power are unreliable; this pilot cannot establish a small real-world improvement.")
    if len(pairs) != len(events):
        caution.append("Exact McNemar uses one all-cases-correct binary per event. The bootstrap delta is case-weighted; these are different estimands.")
    return {
        "baseline": baseline, "candidate": candidate,
        "role": "preregistered_primary" if primary else "preregistered_exploratory_secondary",
        "case_pairs": len(pairs), "independent_events": len(events),
        "case_task_success_pairs": case_counts,
        "completion_recoveries": completion_recoveries,
        "completion_regressions": completion_regressions,
        "overall_task_success_delta": event_cluster_bootstrap(pairs, samples, seed),
        "paired_completed_semantic_pairs": _pair_counts(semantic_pairs),
        "paired_completed_label_accuracy_delta": event_cluster_bootstrap(semantic_pairs, samples, seed),
        "paired_completed_label_note": "Includes only cases completed by both arms; selection can differ from fixed-denominator task success.",
        "event_binary_pairs": event_counts,
        "event_binary_definition": "an event succeeds only when every label-evaluable case completes with its correct label",
        "event_task_success_delta": _ratio(event_counts["fixes"] - event_counts["breaks"], len(events)),
        "exact_mcnemar": {"two_sided_p": exact_mcnemar(event_counts["fixes"], event_counts["breaks"]),
                          "discordant_events": discordant, "independent_event_pairs": len(events),
                          "null": "fix and break probabilities are equal", "alpha": 0.05,
                          "zero_discordance": discordant == 0,
                          "multiplicity": "primary contrast fixed in advance; secondary p-values descriptive and unadjusted"},
        "cautions": caution,
    }


def score(gold, results, bootstrap_samples=10000, seed=2026, arms=ARMS):
    """Validate the full result matrix and return a JSON-serializable report."""
    if not isinstance(bootstrap_samples, int) or isinstance(bootstrap_samples, bool) or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    cases, matrix, mode = _validate(gold, results, arms)
    statuses = Counter(case.get("annotation_status", "unknown") for case in cases.values())
    human_gold = all(case.get("annotation_status") in ("human_reviewed", "human_adjudicated")
                     for case in cases.values())
    comparisons = {}
    for index, baseline in enumerate(("single", "loop_frozen", "independent")):
        comparisons[f"loop_psi_vs_{baseline}"] = _contrast(
            cases, matrix, baseline, "loop_psi", bootstrap_samples, seed, index == 0)
    return {
        "schema_version": "proof-score-v1", "assessment_mode": mode,
        "scheduled_cases_per_arm": len(cases),
        "independent_events": len({case["event_id"] for case in cases.values()}),
        "annotation_status_counts": dict(sorted(statuses.items())),
        "gold_status": "human_reviewed" if human_gold else "provisional_or_incomplete_requires_human_review",
        "arms": {arm: _arm_score(cases, matrix, arm) for arm in arms},
        "contrasts": comparisons, "power_planning": power_planning(),
        "conclusion": "pilot_only_no_proof_of_realworld_accuracy_improvement",
        "interpretation": [
            "Every arm has the same scheduled denominator; explicit execution errors never become false labels or unverifiable predictions.",
            "Unknown or explicitly non-evaluable gold dimensions are excluded and counted; provenance omissions do not alter label scoring.",
            "An unverifiable prediction is a completed abstention and can be correct when the gold label is unverifiable.",
            "Origin recall accepts any listed documentary origin. This evaluates the supplied corpus, not the globally earliest publisher.",
            "Primary comparison is loop_psi versus single; comparisons against loop_frozen and independent are exploratory. Do not select the best comparison and claim confirmatory significance.",
            "Confidence intervals and p-values assume independent sampled events and valid gold. A small convenience sample with provisional AI gold cannot establish population accuracy.",
            "Planning estimates use assumed discordance scenarios, not observed pilot effect sizes, and are not proof or a guaranteed sample size.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    try:
        report = score(json.loads(args.gold.read_text()), json.loads(args.results.read_text()),
                       bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
