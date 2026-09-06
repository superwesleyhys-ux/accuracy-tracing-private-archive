# Fresh original versus Accuracy v0.3 — comparison protocol

This is a new execution of both implementations, not a replay of the previous
first-round checkpoint comparison. No Terminal-Bench tasks are involved.

## Pinned implementations

- Original: `superwesleyhys-ux/news-tracking-perdiction`, commit
  `789506115e7dfef1f2359c2e43bcae4234b8d3f2`; its five agent source files match the
  prior pinned snapshot. `NewsTracingAgent.run()` executes unchanged, at depth 1.
- Accuracy: `superwesleyhys-ux/accuracy-tracing`, engine commit
  `e36292957ce6a6cb5e5c187151681a4503302999`; maximum five retrieval rounds.
- New comparison adapter: `experiments/original_compare.py`. Exact executed
  sources and SHA-256 fingerprints are included with this run.

## Inputs and resources

The same five v0.3 author-written synthetic cases and gold labels remain frozen.
They are known development examples, not a held-out or independent benchmark.
Four cases ask what scoped documents say; one asks whether a world claim is
established. Both arms receive the same task contract and all eligible snapshots
from the start. Original's network-search interface is replaced by a corpus-only
model client. Accuracy's snapshot provider starts with all those snapshots.
Thus, this comparison does not measure open-web collection quality.

Both arms use `gpt-6-astra`, at most 24 API calls, 36,000 output tokens, 2,500
output tokens per call, and 600 seconds per case. Input tokens are measured,
not capped. Equal caps do not imply equal actual spend. The extractor is charged
to each arm. Requests are serialized within an arm, with two jobs concurrent.
Job order is shuffled with fixed seed 20260905. SDK retries and response replay
are disabled. There is one new run per arm and case; no best-of selection.

## Output normalization and scoring

The original agent exposes a narrative `direct_response`; Accuracy exposes a
structured, gated decision. A single identical report-only extraction prompt and
schema is applied to both. It receives the fixed task and the final answer text,
but no source corpus or gold. It must quote the answer verbatim and extract the
existing conclusion without correcting it. Missing quotes, unextractable answers,
or changes to Accuracy's explicit native decision invalidate the arm.

This common extractor avoids giving the original arm a second source-based
adjudication while forcing Accuracy to keep its answer. Nonetheless, normalization
of prose versus structured output remains a measurement limitation. Original
answers and extractor quotes are retained for review. No confidence probabilities
or source-lineage accuracy scores are manufactured.

Inference never opens the gold file. Scoring is a separate command after all jobs
finish. Every case remains in the denominator; failed jobs are reported as errors
and never silently mapped to a correct `unverifiable` answer. Any failure precludes
a blanket claim that all cases were successfully evaluated.

## Interpretation

These tasks measure narrow frozen-document judgments. They do not establish
real-news accuracy, general superiority, robustness across random samples, or
that more verification rounds are inherently better. This is a direct comparison
of two pipelines, not original-plus-Accuracy augmentation. A previous 4/5 to 5/5
checkpoint result should not be presented as the baseline result in this trial.
