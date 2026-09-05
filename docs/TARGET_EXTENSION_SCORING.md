# Target-extension offline comparison

`experiments/target_extension_compare_score.py` compares the original agent, the
staged ψ pipeline, and the target-extension pipeline from retained artifacts. It
does not import an inference client, read an API credential, or make a network
request.

The extension run defines the case denominator. The scorer requires those exact
target and material contracts to exist in both baseline runs, then recalculates
all three arms on that same subset. A failed execution remains a failed scheduled
task; it is never relabeled `unverifiable`.

```bash
python experiments/target_extension_compare_score.py \
  --gold experiments/proof_pilot/gold.json \
  --original-run reports/staged-baseline-01 \
  --staged-run reports/staged-dev-04 \
  --extension-run reports/target-extension-dev-01 \
  --output reports/target-extension-comparison-01.json
```

The output separates four questions:

| Section | What it measures |
|---|---|
| `arms` and `label_comparisons` | Completion, fixed-denominator task success, conditional label agreement, pairwise fixes and breaks |
| `round1_to_final` | First-round to final label fixes and breaks among completed cases with a real first checkpoint |
| `origins` / `edges` inside each arm | Corpus-relative documentary-root and direct-edge accuracy on explicitly evaluable cases |
| `target_plan_probe_coverage` | Structural checks only: required-probe presence, deterministic projections and whether each projected stage executed |
| `target_probe_result_audit` | Retained v2/v3 answers: exactly one evidence/world result per projected probe in every accepted judgement cycle, plus grounded-basis and status counts |
| `loop_audit` | Target-plan, material-critic, and judgement-critic calls, repair requests, and bounded repair attempts; legacy two-counter and current three-counter planner histories remain separately auditable |
| `usage_total` and deltas | Calls, input/output/reasoning/visible-output tokens, and summed service elapsed seconds |

The scorer accepts retained `target_extended_psi_development_v1`, v2, and v3
artifacts. A v1 plan may lack `match_policy`, and v1 reports did not retain
per-probe answers; their result coverage is therefore `null`/unavailable, not
zero. A v2 config must declare `target_plan_schema=decision-probe-v2`; v3 must
declare `decision-probe-v3`. Neither can silently downgrade to a legacy plan.

For v2 and v3, the scorer independently rebuilds the plan checksum from the complete
canonical `critic` projection. Every projection must carry the same `logic`,
`notes`, target signature and checksum, and its claims/probes must exactly equal
the deterministic stage view. It also rechecks rather than trusts runtime
validation:

- Claim and dimension offsets must quote the immutable target exactly, and every
  dimension must remain inside its claim.
- Identity/designation spans may not overlap. `exact_designation` is permitted
  only with an explicit `name`, `rename`, `call`, `title`, `designate`, `label`
  or `term` inflection in that claim's own predicate anchor; attribution
  dimensions cannot cross the reporting boundary into attributed content.
- Every claim containing `exact_designation` requires one
  `designation_relation` probe bound to all of the claim's dimensions. It uses
  `semantic_constraint` and checks the directed naming transition. Every
  conclusive result must have one grounded sentence/line containing both an
  explicit naming predicate and all exact target labels; cue and labels split
  across sentences are rejected.
- Conditional, comparison and causal plans require their corresponding relation
  probe, bound to every dimension of their single retained relational claim.
- Probe routes, gates and match policies are recomputed from probe kind. The
  scorer rejects missing, extra or altered required bindings.
- In v3, every `entity_identity` question is independently regenerated from its
  target anchor and `same_referent` policy. A model-authored question cannot add
  an actor, owner or institutional-role requirement; those belong to
  `semantic_core` or `designation_relation`.
- In v3, a claim may retain multiple non-overlapping time dimensions. Each time
  dimension requires its own singleton `time_boundary` probe. A merged time
  question, a missing time ID, or a partial binding is rejected.
- The scorer rechecks the planner's 4-claim, 9-dimension-per-claim and 44-probe
  bounds, Boolean role restrictions, and recomputes claim, dimension and probe
  IDs from the immutable target signature and anchors.
- Each accepted final judgement cycle needs exactly one evidence result and one
  world result for every probe projected to those stages. Every quoted span must
  match retained `material.content`; evidence spans must remain inside the frozen
  evidence scope.
- A conclusive result needs a nonempty source basis, while `conflicting` needs at
  least two distinct passages. `referent_relation` must agree with both status
  and `match_policy`: descriptive/alias/anaphoric matches can support
  `same_referent`, but only an exact match can support `exact_designation`.
- In v3, final origins are independently reconstructed from retained raw
  candidates and the direct lineage graph. Only candidates reachable from the
  target source and terminal among reachable candidates may be exported. An
  intermediate candidate, an omitted terminal candidate, a disconnected
  candidate, or a non-terminating cycle fails the audit; independent parallel
  terminal roots remain valid.

The designation sentence rule is lexical and structural, not a semantic proof:
an in-sentence actor, direction, qualifier or factual interpretation can still
be wrong and must be caught by judgement review and independently labelled
evaluation. Structural coverage is deliberately not folded into result coverage or label
accuracy. Structural coverage of `1.0` says that the checklist was complete and
its stages ran. Result coverage of `1.0` says that every required result was
retained and structurally grounded. Neither proves that the result interpretation
or final label is correct. Provider-reported reasoning tokens are likewise a
resource diagnostic, not a quality score.

The current eight cases are previously seen development examples with provisional
AI-reviewed references. They can expose regressions and accounting errors, but
cannot prove a real-world accuracy improvement. That claim requires a separately
frozen held-out set with independent adjudication.
