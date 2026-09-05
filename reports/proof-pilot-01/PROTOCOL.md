# Frozen-source ablation pilot — preregistered before inference

This pilot starts the evidence-building process. It is not a claim of proven
general accuracy improvement. It replaces the five old synthetic examples with
eight new source-derived probes from eight distinct real public-source events.
The claims include deliberate scope/quantity/time probes and are not necessarily
headlines that anyone actually published. Gold is AI-authored and reviewed by a
second execution-team AI before inference; human adjudication remains pending.

## Scope and data separation

`inputs.json` contains only target contracts, material excerpts, and seed settings.
`gold.json` and `sources.json` are never opened by the inference runner. The frozen
manifest hashes inputs, gold, sources, protocol and source review before any model
calls. Independent events, not the number of rewritten claims or repeated runs,
are the statistical clusters. All eight cases remain in every arm's denominator.
No gold/prompt/implementation change after outputs is permitted within this pilot.

The excerpts are current captures. Historical publication dates are metadata,
not proof that this exact webpage version was available in the past. Targets
use a current as-of later than collection. Government announcements support
documentary claims; they do not automatically prove every actual-world claim.
Gold permits unknown results and alternative acceptable documentary roots.
Source edges are scored only when the references in the supplied excerpts can
be evaluated exhaustively. The task does not find the unknowable first author
of an AI-authored probe.

## Arms and primary contrast

| Arm | Procedure |
|---|---|
| original | Original `NewsTracingAgent.run`, depth 1, with fixed-corpus interface |
| single | Accuracy round 1, then stop |
| loop_psi | Exact same round-1 checkpoint, then two real verification rounds with psi reanalysis |
| loop_frozen | Same checkpoint and two verification rounds, but reuse frozen round-1 analyses |
| independent | Round-1 sample plus two fresh one-round samples, then one selector of an existing answer |

All branches receive the same complete eligible excerpt corpus. There is no live
retrieval advantage. `loop_psi` and `loop_frozen` must record exactly three actual
verifications, not three search calls. Experimental forced continuation overrides
normal complete/no-progress stopping but preserves errors and hard budgets. This
tests a forced three-round intervention, not the default adaptive production policy.

First-round state is copied before continuation, SHA-256 hashed, and never mutated
by another branch. It includes analyses, graph, gaps, resolutions, previous
verification and counters. Later verification/decomposition sees previous outputs
as hypotheses, not new evidence. The no-psi arm has no new semantic decomposition
calls after round 1, although typed frozen analyses pass through graph validation.

The independent arm's additional samples receive no previous sample. Its selector
sees the three trial assessments and the same source material, and must choose
one existing answer. The common label extractor sees only target and final answer,
requires a verbatim quote, and cannot override Accuracy's valid native decision.
Original documentary roots are exported from native `is_original` flags; original
source-edge export is explicitly unsupported and excluded from edge comparisons.

Primary contrast: `loop_psi` minus `single` on four-class claim correctness.
Secondary exploratory contrasts: `loop_psi` minus `loop_frozen`, `independent`, and
the original baseline. No strongest-looking secondary comparison will replace
the primary one after results are known. No multiple-testing superiority claim.

## Resources and sampling

One new model run per event/arm using `gpt-6-astra`, with four event jobs concurrent.
Maximum per logical arm: 24 calls, 36,000 output tokens, 2,500 output tokens per call,
600 seconds, 24 material returns and 36 decompositions. Input tokens are measured
but not capped. Calls are serialized within a logical arm, SDK retries disabled.

The shared first-round prefix is sent once and charged to every arm that uses it;
all such records are marked `shared_first_round`. Actual new response IDs and
token usage are reported separately from logical-arm totals. The single arm's
final extractor is not injected into the continued branches. Branch elapsed time
charges the prefix duration plus that branch's work, excluding unrelated branch
execution. Equal caps are not equal actual computation: report measured use and
do not claim that a faster result alone proves a causal accuracy benefit.

Case order is shuffled with seed 20260905; branch order is deterministically
shuffled per case. These seeds do not control model sampling. No best-of reruns,
no selective removal of failed tasks, no supplied correctness feedback.

## Metrics and decisions

Report completed tasks, task success over all assigned cases, accuracy among
completed tasks (separately named), abstention/unknown frequency, documentary-root
precision and coverage, and citation-edge precision/recall where evaluable. A
native error fallback cannot be scored as a correct unknown answer. Scoring
units come from the gold cases, never from a method's own number of fragments.

For paired label comparisons report wrong-to-right and right-to-wrong counts,
net difference and event-cluster confidence intervals. Exact McNemar calculations
use independent event-level outcomes; repetitions do not inflate event count.
Degenerate all-equal/no-discordance bootstrap intervals must not be presented as
proof of precise equivalence. Eight pilot events are not sufficient for a general
superiority claim; provisional gold also prevents a confirmatory conclusion.

Formal evaluation will freeze one primary metric and a minimum worthwhile gain
(planning example: three percentage points). Determine sample size by prospective
power analysis using pilot information plus conservative disagreement scenarios,
not an arbitrary fixed count or repeated testing until significance. A confirmatory
claim requires human-reviewed event-disjoint held-out cases, predeclared stopping,
and a confidence interval above the chosen threshold on that task distribution.

This pilot does not implement a live-web retrieval superiority comparison. That
follow-up must separately compare retrieval strategies and give a one-round
control all material collected by the loop to distinguish retrieval and reasoning.

## Reproduction

```bash
python experiments/proof_run.py --inputs experiments/proof_pilot/inputs.json --original /path/to/pinned-original --output reports/proof-pilot-new-run
python experiments/proof_score.py --gold experiments/proof_pilot/gold.json --results reports/proof-pilot-new-run/results.json --output reports/proof-pilot-new-run/scores.json
```

Credentials are provided only through the environment or echo-disabled prompt.
Do not commit them. Preserve every run under a new directory. Source hashes and
the executed-code archive accompany the run. The human-review requirement remains
visible until humans actually perform and record that review.
