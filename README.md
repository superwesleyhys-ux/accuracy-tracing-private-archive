# Accuracy Tracing — NewsVerify Harness

Version 0.3.0: a bounded news provenance loop with **decomposition on every material return**, split single-responsibility semantic prompts, separate evidence/world assessments, task-directed snapshot retrieval, and a shared single-round/multi-round decision policy.

**Status: staged model adapter and targeted regression experiments implemented.** The optional adapter in `experiments/` builds a deterministic target plan and runs seven bounded model stages: atoms, lineage, decomposition critic, evidence, evidence critic, world, and world critic. Python then performs atomic decomposition and judgement assembly. Eligible returns use that path; historically ineligible returns use isolated conservative archival decomposition. A searchable snapshot provider executes fetch/search/reanalysis tasks and reports exact task attribution and coverage. It does not browse the open web. There is no independently reviewed real-news accuracy claim. See the [staged validation loop](docs/STAGED_VALIDATION_LOOP.md), [v0.3 repair contract](docs/REPAIR_V0.3.md), and [observed earlier results](reports/REPAIR_RESULTS_V0.3.md).

Repository Discussions are enabled, and the repository includes a prepared **Accuracy decline** reporting form for reproducible metric regressions or weaker trace outcomes. Reports should identify the affected metric or behavior, include the run configuration, and avoid treating synthetic fixtures as real-world performance evidence.

## Run

Python 3.11+; standard library only. From this project directory:

```bash
python -m newsverify trace-demo --output reports/trace-demo-v0.3.json
python -m newsverify score examples/evaluation_gold.json examples/evaluation_predictions.json --output reports/all-metrics-v0.3.json
python -m newsverify compare examples/evaluation_gold.json examples/comparison_baseline.json examples/comparison_candidate.json --bootstrap-samples 100 --seed 0 --output reports/comparison-v0.3.json
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
- Task-aware returns carry exact issued-gap attribution; staged verification persists each task's target-probe identity and validates provider feedback against actual consumed versions.
- Round, material and decomposition budgets, plus explicit no-progress and error results.
- Historical admission requires an exact version availability declaration and basis. There is no arbitrary age cutoff for old original records.

Historically excluded versions now receive isolated archival decomposition and are never passed to the stateful semantic plugin. This does not remove later knowledge already present in model pretraining. The optional model transport enforces call/output/time caps and records actual server token usage.

## Fixed-contract model comparison

Install optional model dependencies with `python -m pip install '.[model]'` and configure `OPENAI_API_KEY` in the execution environment. Keep credentials out of repository files. Then:

```bash
python experiments/loop_compare.py run --inputs experiments/inputs-v03.json --output reports/my-run --model MODEL_NAME --semantic-mode staged --max-repairs 1 --max-rounds 5
python experiments/loop_compare.py score --gold experiments/gold-v03.json --run reports/my-run
```

The staged mode is the default for new runs. Each eligible material follows `atoms → lineage → decomposition_critic`; the complete graph state then follows the isolated `evidence → evidence_critic` and `world → world_critic` paths before deterministic judgement assembly. The decomposition transaction shares one repair allowance, and each judgement layer separately shares one allowance between its deterministic gate and critic. Every immutable target probe and required dimension must appear exactly once with source-grounded basis before Python aggregates the result. Prompt/schema/input/output hashes, transaction links and stage audit records are retained. `--semantic-mode monolithic --max-repairs 0` preserves the earlier adapter for compatibility and later ablation. The two modes use different budgets and response caches, so their current results are not an equal-cost causal measurement of prompt splitting.

For a real-source pilot rather than the synthetic contract examples, [`docs/HISTORICAL_2023_BENCHMARK.md`](docs/HISTORICAL_2023_BENCHMARK.md) defines eight official 2023 propositions whose outcomes were settled by the end of 2024. Its paired runner blocks live retrieval, keeps checksummed gold outside inference, rejects post-cutoff source versions, requires versioned raw-artifact identities and a clean committed benchmark before model calls, and gives the original and staged adapters identical outer resource and trace ceilings. It is intentionally described as historical evidence isolation: current-model parameter knowledge cannot be rolled back, and measured input tokens are not part of the shared caps.

A built `newsverify-harness` wheel contains the standard-library core CLI and trace engine. The optional staged experiment runtime remains source-checkout-only under `experiments/`; it is smoke-tested from a checkout but is not advertised as an installed console entry point.

The first-round checkpoint and continued loop share the actual first-round model output and the same deterministic final decision function. A flagged case also runs once with all snapshots supplied, to distinguish additional source access from repeated reasoning. These five author-written synthetic contract cases are not a hidden or independently reviewed benchmark, and splitting prompts does not itself prove an accuracy gain.

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
- [Staged prompt and validation loop](docs/STAGED_VALIDATION_LOOP.md)
- [Evaluation schema](docs/EVALUATION_SCHEMA.md)
- [Current offline validation report](reports/VALIDATION_V0.3.md)
- [Historical v0.2 validation report](reports/VALIDATION_V0.2.md)
- [v0.3 repair and migration contract](docs/REPAIR_V0.3.md)
- [v0.3 regression notes and historical pre-staged API results](reports/REPAIR_RESULTS_V0.3.md)

## Legacy compatibility

`python -m newsverify demo`, `verify`, and `benchmark` retain the v0.1 annotated-evidence policy runner in `core.py`. Its publisher/origin grouping and 72-hour default window are legacy policy choices, not the v0.2 provenance algorithm. Its 22 synthetic scenarios remain regression tests, not real-news accuracy estimates. The earlier [adapter contract](docs/ADAPTER_CONTRACT.md) applies to that runner only.

Released under the [MIT License](LICENSE). Source repository: [superwesleyhys-ux/accuracy-tracing](https://github.com/superwesleyhys-ux/accuracy-tracing).
