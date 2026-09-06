# Accuracy evidence-building experiment

The first ablation pilot uses eight source-derived probes and five arms. Its
[preregistered protocol](../experiments/proof_pilot/PROTOCOL.md) distinguishes
verification, psi reanalysis, independent resampling, and overall pipeline effects.
All reference labels remain provisional pending human review.

`TraceConfig.experimental_force_rounds` defaults to `False`. Default adaptive
stopping remains unchanged. Forced mode still stops on empty retrieval, errors
and hard budgets; it never fabricates a verification response.

`run_provenance(..., checkpoint_callback=callback)` returns a deep-isolated
`TraceCheckpoint` after completed verification. `checkpoint=cp` resumes at the
next round without replaying prefix calls. Target and document/decomposition
budgets must match. Round limits and experimental settings may change per branch.
`canonical_checkpoint_json(cp)` and `checkpoint_sha256(cp)` provide audit and
integrity hashes; JSON is not an authenticated external checkpoint loader.
Model clients, providers and credentials are not in checkpoints.

Install model dependencies and those required by the original repository; provide
credentials through the environment or `--prompt-key` in an echo-disabled terminal.

```bash
python experiments/proof_run.py --inputs experiments/proof_pilot/inputs.json --original /path/to/original --output reports/new-proof-pilot
python experiments/proof_score.py --gold experiments/proof_pilot/gold.json --results reports/new-proof-pilot/results.json --output reports/new-proof-pilot/scores.json
```

Every run has a new directory. Shared prefixes are marked and charged to each
logical arm; actual new API use is separate. The common report-only extractor
cannot override native Accuracy decisions. Original source flags are exported,
but its missing source-edge representation is explicitly unsupported.

The pilot is not a held-out test. Human reviewers must separately verify gold
citations, dates, roots and uncertainty. Formal testing needs new event-disjoint
cases and preregistered comparisons, metrics, minimum useful effect, budget and
sample size. The scorer's assumption-based planning output helps scope that work;
a perfect small pilot does not establish superiority.
# Completed pilot and subsequent corrections

Read [PROOF_PILOT_RESULTS.md](../reports/PROOF_PILOT_RESULTS.md) before running a
larger experiment. Pilot 01 failed the practical execution gate: only one of
eight Accuracy prefixes completed. The frozen inference code is retained under
`reports/proof-pilot-01/executed-code/`; the preregistration commit and immutable
file hashes identify the exact configuration used. Current code includes later
scoring, cost-accounting, diagnostic and complete-replacement prompt corrections.
Those changes have offline validation only; do not attribute them to pilot 01's
live results. For any new run, create a new manifest and output directory rather
than replacing the old protocol, gold, inputs or outputs. Treat these eight cases
as development data and use new human-reviewed events for confirmatory claims.
