# Accuracy Tracing — NewsVerify Harness

## Current development snapshot — 2026-09-05

The current development code is target-extension v4. The synchronized snapshot
passes 433 offline regression tests after the latest scoring-audit sync; these are
not model accuracy results. The earlier snapshot also passed 22 synthetic policy cases. A matching v4 freeze and fresh live-model comparison are
still pending. See [latest synchronization status](reports/V4_FOLLOWUP_SYNC_20260905.json),
[earlier snapshot](reports/WORKTREE_SYNC_20260905.md), and
the [v4 protocol](docs/TARGET_EXTENSION_V4_PROTOCOL.md). Older pilot results below
are retained historical records, not the current v4 score.

An experimental [shared-checkpoint ablation pilot](docs/PROOF_EXPERIMENT.md) now
separates three-round verification, psi reanalysis and independent resampling.
Its real-source-derived reference labels are provisional pending human review;
it does not establish real-news accuracy or general superiority.

The [completed eight-event pilot](reports/PROOF_PILOT_RESULTS.md) exposed execution
failures: original task success was 7/8; single and both loop arms were 1/8;
independent resampling was 0/8. Seven Accuracy prefixes failed before verification.
Only one case completed three rounds, with no label improvement. Raw failures,
an independent audit and explicitly corrected logical costs are retained. Later
diagnostic and prompt fixes pass offline tests but have not been evaluated live.

Version 0.3.0: a bounded news provenance loop with **decomposition on every material return**, separate evidence/world assessments, task-directed snapshot retrieval, and a shared single-round/multi-round decision policy.

**Status: model adapter and targeted regression experiments implemented.** The optional OpenAI adapter is in `experiments/`; the original demos remain hand-annotated. A searchable snapshot provider executes fetch/search/reanalysis tasks and reports missing coverage. It does not browse the open web. There is no independently reviewed real-news accuracy claim. See [v0.3 repair contract](docs/REPAIR_V0.3.md) and [observed results](reports/REPAIR_RESULTS_V0.3.md).

Repository Discussions are enabled, and the repository includes a prepared **Accuracy decline** reporting form for reproducible metric regressions or weaker trace outcomes. Reports should identify the affected metric or behavior, include the run configuration, and avoid treating synthetic fixtures as real-world performance evidence.

## Run

Python 3.11+; standard library only. From this project directory:

```bash
python -m newsverify trace-demo --output reports/trace-demo-v0.2.json
python -m newsverify score examples/evaluation_gold.json examples/evaluation_predictions.json --output reports/all-metrics-v0.2.json
python -m newsverify compare examples/evaluation_gold.json examples/comparison_baseline.json examples/comparison_candidate.json --bootstrap-samples 100 --seed 0 --output reports/comparison-v0.2.json
python -m unittest discover -s tests -v
```

The trace demo follows four material versions over three retrieval rounds, reopens affected old analyses, and routes a verification-requested correction through decomposition. It preserves the original target and separates lineage from semantic contradiction.

The metric example is a deliberately imperfect set of four **handwritten predictions**, used to verify arithmetic against independent expected values. The comparison example uses identical handwritten runs to check paired differences. Neither example is a model performance result.

## Trace engine

`run_provenance(target, provider, decomposer=None, verifier=None, config=None)` uses typed plug-ins documented in [TRACE_ADAPTER.md](docs/TRACE_ADAPTER.md).

- Immutable material versions, exact source spans and observation records.
- Every valid return is saved and decomposed before graph or verifier admission.
- Same-URL revisions keep distinct identities; an ID/content collision fails unresolved.
- Candidate relations distinguish direct, declared, inferred, unresolved and excluded evidence.
- Original-source completion requires an explicit finding and a direct lineage path from the target's source version. Support/contradiction edges do not substitute for that path.
- `revisit_versions` triggers affected earlier analyses; current results are rebuilt while history remains available.
- Verification gaps follow the same retrieval/decomposition route.
- Round, material and decomposition budgets, plus explicit no-progress and error results.
- Historical admission requires an exact version availability declaration and basis. There is no arbitrary age cutoff for old original records.

Historically excluded versions now receive isolated archival decomposition and are never passed to the stateful semantic plugin. This does not remove later knowledge already present in model pretraining. The optional model transport enforces call/output/time caps and records actual server token usage.

## Fixed-contract model comparison

Install optional model dependencies with `python -m pip install '.[model]'` and configure `OPENAI_API_KEY` in the execution environment. Keep credentials out of repository files. Then:

```bash
python experiments/loop_compare.py run --inputs experiments/inputs-v03.json --output reports/my-run --model gpt-6-astra --max-rounds 5
python experiments/loop_compare.py score --gold experiments/gold-v03.json --run reports/my-run
```

The first-round checkpoint and continued loop share the actual first-round model output and the same deterministic final decision function. A flagged case also runs once with all snapshots supplied, to distinguish additional source access from repeated reasoning. These five author-written synthetic contract cases are not a hidden or independently reviewed benchmark, and their scores are not comparable with the earlier ambiguous two-case trial.

`Target.assessment_mode` is `evidence` or `world`. `evidence_scope` fixes which snapshot texts are assessed for entailment. The engine preserves raw evidence findings while applying only relevant blocking gaps to the chosen assessment. `present_decision()` performs the same final mapping in every variant; it cannot resample an answer into a higher score.

## Evaluation

Gold labels and predictions are separate files. Cases are fixed before inference. Missing/extra/duplicate target IDs, mismatched cutoffs, unjudged evidence IDs and invalid probabilities fail validation.

| Metric | Measures |
|---|---|
| VP | Precision of claims admitted to the trusted feed |
| FR | Fraction of false claims withheld from the feed |
| TR | Fraction of true claims admitted |
| SR | Correct original-root sets with valid provenance paths on traceable targets |
| EN | Precision of evidence asserted to support its target |
| CA | One minus half the four-class Brier score; probability quality, not pure calibration |
| HFAR | High-risk false claims incorrectly admitted |

Also reports source precision, false promotion of unknown origins, edge precision/recall, evidence recall, duplicate-pair F1, confusion matrix, coverage and ECE. A zero denominator is `null`. Missing probability vectors make CA and the aggregate unavailable. Scores never establish truth by themselves.

The experimental `NVScore` preserves the weights discussed in the design. It is not Terminal-Bench or an official benchmark, and its weights are not empirically validated. `compare` checks **declared** equal model/corpus/budget settings and supplied usage, then produces paired event-cluster bootstrap intervals. It cannot attest that external model usage logs are authentic.

The trace engine and scorer have separate schemas. A production exporter and independently reviewed data are still required; no implicit conversion turns plug-in judgments into gold labels.

## Documents

- [Chinese design conclusion and all metric definitions](docs/ACCURACY_TRACING_SPEC.md)
- [Codex execution plan and remaining implementation sequence](docs/CODEX_EXECUTION_PLAN.md)
- [Trace adapter API](docs/TRACE_ADAPTER.md)
- [Evaluation schema](docs/EVALUATION_SCHEMA.md)
- [Observed validation report](reports/VALIDATION_V0.2.md)
- [v0.3 repair and migration contract](docs/REPAIR_V0.3.md)
- [v0.3 regression and actual API results](reports/REPAIR_RESULTS_V0.3.md)

## Legacy compatibility

`python -m newsverify demo`, `verify`, and `benchmark` retain the v0.1 annotated-evidence policy runner in `core.py`. Its publisher/origin grouping and 72-hour default window are legacy policy choices, not the v0.2 provenance algorithm. Its 22 synthetic scenarios remain regression tests, not real-news accuracy estimates. The earlier [adapter contract](docs/ADAPTER_CONTRACT.md) applies to that runner only.

Released under the [MIT License](LICENSE). Source repository: [superwesleyhys-ux/accuracy-tracing](https://github.com/superwesleyhys-ux/accuracy-tracing).
