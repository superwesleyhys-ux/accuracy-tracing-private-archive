# v0.3 repair and migration contract

This document supersedes conflicting v0.2 adapter/implementation notes. The legacy policy runner and seven-metric arithmetic evaluator remain available. Historical v0.2 reports are retained as historical records.

## Fixed question and layered assessments

`Target` adds `assessment_mode="world"` and `evidence_scope=()`. Evidence mode asks what the named document versions support or contradict; world mode asks whether the actual event is established. Source originality and truth remain separate questions. An empty evidence scope uses all eligible material. Dict inputs may use a JSON array for the scope; the engine normalizes it to an immutable tuple.

`VerificationResult` preserves the five old positional fields and adds `evidence_verdict`, `world_verdict`, `world_basis`, and `world_rationale`. Legacy plugins may omit the new fields; their old verdict is used for both raw assessments. The supplied model adapter always returns both explicitly and keeps the legacy `verdict` equal to its evidence verdict. A positive world judgment requires its own exact evidence basis and explanation. This is structural validation, not a programmatic proof that a model's semantic judgment is true.

The report adds `assessments`, `decision_status`, and `assessment_valid`. `assessments[dimension]` stores `raw_verdict`, the gated `decision`, and `blocking_gap_ids`. `fact_status` continues to describe the world decision. Evidence findings are preserved when world verification remains unknown. `decision_status` follows the fixed target assessment mode. Errors invalidate final decision presentation while retaining prior audited findings.

Use `newsverify.decisions.present_decision(report)` for every compared variant. `round_decisions(report)` exposes actual completed-round checkpoints using the same label mapping. Neither function calls a model or invents confidence probabilities. Changing assessment mode means changing the task and requires separate gold labels; it is not a way to repair an old score after seeing predictions.

## Gaps have scope and executable actions

`Gap` adds dimension (`provenance`, `evidence`, `world`, or legacy `auto`), a boolean `blocking`, optional `target_id`, exact `basis`, `decision_impact`, action (`fetch`, `search`, `reanalyse`), and optional `locator`.

Only blocking gaps for the relevant dimension affect its decision. Explicit blocking evidence/world gaps need an actual source basis and a stated possible effect on the decision. Gaps for another target are rejected. The semantic adapter is responsible for judging relevance; merely citing text cannot prove that a gap is material.

`SnapshotSearchProvider` executes fetches against exact URLs/version IDs, query search against the supplied index, and explicit reanalysis of previously returned versions. It retains retrieval history and reports unavailable coverage rather than endlessly returning the seed documents. It is an offline index and makes no claim to cover the open web. When no result can satisfy the pending tasks, the engine records provider exhaustion without another unchanged verification call.

The provider protocol remains replaceable by a live collector. A collector must return genuine captured versions and explicit availability evidence, preserve corrections under new IDs, and report its scope honestly. API credentials, web authentication and independently reviewed source trust are not inferred by this index.

## Every return and affected previous versions pass through psi

Every valid arrival is saved and decomposed before graph/verification admission. Future or unavailable versions receive an isolated archival decomposition; a stateful semantic plugin does not see them. This does not solve model pretraining contamination.

When a newly admitted upstream matches an old declared citation locator, the engine automatically schedules psi for the old source even if the model omitted `revisit_versions`. It never silently marks that citation direct. Reanalysis still requires the source-side quote and available upstream. Traversal and decomposition budgets apply. Self-revisits are audited and deduplicated; they do not create a missing-news-fact gap.

Fragments must have an acyclic parent path to the target through fragments in that analysis. `qualifier_spans` locates the exact wording carrying negation, time, quantity or scope. Human-readable paraphrases remain available in `text` and `qualifiers`.

## Progress and budgets

Evidence progress hashing includes source spans, grounded qualifier spans, relation structure, origin findings and actionable gap structure. It excludes regenerated fragment/relation IDs, free paraphrases, human qualifier prose and gap explanations. A changed source span or action can justify a further round; merely changing wording cannot. This is a structural proxy and still needs semantic review for whether a new relation or qualifier is meaningful.

Round, material and decomposition caps remain hard limits. The optional transport separately enforces model call count, output-token and wall-time caps; input tokens are measured, not capped. Its SDK retries are disabled. An API or integrity failure remains an error and is retained in evaluation.

## Reproducible experiments and reuse

`experiments/inputs-v03.json` and `gold-v03.json` were frozen before inference; their hashes are in `frozen-v03.json`. They define five synthetic contract cases and one full-evidence control. All targets and all failed attempts must be retained. Inference never opens gold; scoring is a separate command.

The comparison uses the literal first-round checkpoint versus the continued run under the same mapping. It does not reexecute the original agent, and it does not claim an independent stochastic baseline. The all-materials control helps identify when an apparent loop gain comes from access to additional documents.

`--reuse-from` may reuse responses only from the same case/variant, with identical model, prompts, schema and per-call output cap. Prior API IDs, token usage and wall time are retained and charged to logical budgets. Reused responses are marked; actual new API usage is reported separately. It is response replay, not a fresh model measurement. Changed requests are sent to the official API, and errors are never cached as successful answers.

## Compatibility

The report schema is now `0.3`; consumers must account for the additional assessment fields. Legacy constructors and v0.1 CLI behavior remain supported. Plugins that previously expected to see excluded future material must adapt to isolated archival decomposition. Nested fragment parents must be included in the same analysis response. Free-form qualifier changes are no longer sufficient to prolong a loop; use grounded `qualifier_spans` when the relevant source wording changes.
