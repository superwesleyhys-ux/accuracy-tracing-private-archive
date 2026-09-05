# Proof pilot 01: transport, continuation, and accounting audit

The completed pilot contains 40 arm results, 156 new request attempts, 154 unique response IDs, and nine explicitly marked successful-prefix copies. Seven single arms failed before verification with empty output after consuming the 2,500-token per-call allowance. Only p04 produced a valid shared checkpoint; both its continued arms completed three real verifications. The independent control did not complete for any case. A logical-cost omission for failed shared prefixes is documented below; no efficiency claim is supported.

This audit began with a read-only snapshot at 2026-09-05 09:52:07 UTC covering p02, p03, and p07, and was expanded after the run completed. No gold, credentials, live session, or model API was accessed. Frozen code, prompts, inputs, and run outputs were not modified by this audit. Findings concern execution and representation, not gold-label accuracy.

## Confirmed failure mechanism

All three affected arms received a response from the configured model, consumed the entire 2,500-token per-call completion allowance, and recorded no visible output text. The transport wrapper rejected the response before semantic decoding. The runner then stopped in decomposition before any verification completed, so it correctly had no checkpoint to branch from.

| Case | Failed call | Material | Requested cap | Reported completion tokens | Visible output characters | Failed-call seconds | Single-arm total seconds |
|---|---:|---|---:|---:|---:|---:|---:|
| p02 | 1 | m03 | 2,500 | 2,500 | 0 | 40.668 | 40.672 |
| p03 | 2 | m06 | 2,500 | 2,500 | 0 | 35.333 | 76.563 |
| p07 | 2 | m13 | 2,500 | 2,500 | 0 | 36.309 | 71.074 |

Evidence is in `proof-pilot-01/p02-single-calls.json`, `p03-single-calls.json`, and `p07-single-calls.json`. The failed records contain `actual_model="gpt-6-astra"`, a server response ID, `max_completion_tokens=2500`, `usage.output_tokens=2500`, `output=""`, and `error_type="RuntimeError"`.

The failed response IDs are:

- p02: `chatcmpl-EKhIbT76tx6BB57ZFsu2p8SWnHpQD`
- p03: `chatcmpl-EKhIU5w6qvTPmpytCc2hmHkZkWBPy`
- p07: `chatcmpl-EKhIVblBQiazZHZTevSMVUbrKwtgO`

The prior p03 and p07 decomposition calls succeeded, with valid JSON and 2,173 and 2,133 completion tokens respectively. At failure, accumulated output tokens were 2,500, 4,673, and 4,633—well below the 36,000-token per-arm cap. Elapsed times were also below the 600-second cap. This rules out the wrapper's observed aggregate-output and elapsed-time checks as the direct causes recorded here. The response IDs, actual model, and usage also establish that these were returned responses, rather than failures to obtain any response.

## What the current logs cannot establish

Per-call completion-cap exhaustion before visible JSON is the strongest inference from the shared signature. The logs do **not** prove the exact finish reason or whether reasoning consumed the allowance.

In the frozen `executed-code/accuracy/experiments/model_io.py`:

- Lines 119–126 record usage and visible content.
- Line 127 rejects a refusal with `RuntimeError`.
- Line 128 rejects any finish reason other than `stop` with `RuntimeError`.
- Lines 129–130 reject observed aggregate-output or elapsed-time overruns; the recorded totals exclude those conditions here.
- Line 131 parses JSON. Empty stopped content would produce `JSONDecodeError`, not the recorded `RuntimeError`.
- Lines 132–135 retain only the exception class and replace the public message with `Model call failed: RuntimeError`.

Neither `finish_reason`, refusal presence, nor reasoning-token usage is persisted. Therefore refusal and non-stop completion cannot be distinguished conclusively from these artifacts. The audit should not label the exact server condition as established fact.

## Propagation into the experiment

Each `*-single-report.json` has:

- `stop_reason="decomposer_error"`;
- an error at stage `decomposer` with message `Model call failed: RuntimeError`;
- `verification_calls=0` and no successful first-round verification.

The frozen `proof_run.py` calls the engine with a checkpoint callback at lines 237–239. Line 242 raises `ValueError('No valid first-round checkpoint')` when the callback produced no snapshot. This explains why the single-arm result exposes the less-specific `error_type="ValueError"` while the underlying report records a decomposition transport error.

Lines 251–254 then produce `InvalidSharedPrefix` results for `loop_psi`, `loop_frozen`, and `independent`. Their recorded usage is zero because those branches never ran. These are consequences of one failed common prefix per case, not three independent loop failures. No continued verification was attempted for these cases, so they cannot support a claim about the effect of extra rounds.

The checkpoint implementation's refusal to branch without a successful verification is behaving as designed. No checkpoint mutation, prefix replay, source eligibility failure, or semantic span-validation failure is evidenced by these three failures.

## Minimal general instrumentation fix for a separate change

Preserve the current transport, validation order, budget limits, one-attempt behavior, and error outcomes. Before response validation, record safe diagnostic fields:

1. `finish_reason` and `has_refusal` (boolean; do not persist refusal text as a new diagnostic field).
2. Optional `reasoning_tokens` from the SDK's completion-token details when available.
3. A stable local `error_code` distinguishing `incomplete_model_output`, `model_refusal`, `missing_usage`, `output_budget_exceeded`, `timeout_budget_exceeded`, `invalid_json`, and `sdk_error`.
4. Continue retaining only the exception class for SDK failures. Do not log arbitrary exception strings, authentication values, or request headers.
5. Record an optional `sdk_error_code` only for the exact whitelist values `insufficient_quota` and `rate_limit_exceeded`, from the SDK code attribute or structured body. Do not copy unknown remote codes or bodies.

Offline fixtures should distinguish empty content with a length finish reason and a fully consumed token allowance, refusal, empty stopped JSON, missing usage, actual aggregate/time overruns, and SDK exceptions containing a synthetic secret string. All fixtures should verify unchanged call counts and caps, and no retry.

This instrumentation patch does not fix or relabel the present pilot outcomes and does not justify a budget increase. Any future change to the model's reasoning setting, semantic output contract, or resource allowance would require a separate explicit experiment configuration. No such change or second run is proposed by this audit.

## Source integrity checked

At the observation timestamp, these current files matched their executed-code copies and manifest hashes:

| File | SHA-256 |
|---|---|
| `experiments/model_io.py` | `ca6a974d512f16068a928d87da9a19f4b05105f878ec95777cac331fa104e875` |
| `experiments/proof_run.py` | `a84dd081b96885df0c8bc994d4d9a496fa5a05e42e188c08442a7e92be9850ae` |
| `newsverify/provenance.py` | `0066f2f54290745dd527a330d342197d489dad237aab78b45e0b772eba13211b` |

This document is a separate audit artifact outside the frozen run directory.

## Final-run coverage and additional failures

| Arm | Completed | Error |
|---|---:|---:|
| Original | 7 | 1 |
| Single | 1 | 7 |
| Continued psi | 1 | 7 |
| Frozen psi | 1 | 7 |
| Independent | 0 | 8 |

The remaining single failures have the same recorded signature as the first three: `RuntimeError`, empty text, actual model `gpt-6-astra`, and 2,500 reported completion tokens against a 2,500-token requested cap.

| Additional failed single | Call | Material | Failed-call seconds | Total arm output tokens | Total arm seconds |
|---|---:|---|---:|---:|---:|
| p01 | 1 | m01 | 38.856 | 2,500 | 38.860 |
| p05 | 1 | m09 | 39.131 | 2,500 | 39.135 |
| p06 | 1 | m10 | 40.196 | 2,500 | 40.200 |
| p08 | 2 | m15 | 31.449 | 4,652 | 66.540 |

All seven failed singles stopped in decomposition with zero verification calls. Their aggregate-token and elapsed-time totals exclude the wrapper's observed arm-budget checks as the immediate cause. Each produced three zero-new-call `InvalidSharedPrefix` rows.

For p04 independent, call 5 (the second new call after three shared prefix records) failed on m08 with the same empty-output/2,500-token signature. `p04-independent-trial-1.json` records two decomposition attempts and zero verifications. This branch never reached three valid trials, selection, or final extraction.

The separate p05 original failure has `RateLimitError` records at sequences 10 and 11, no recorded response IDs or usage for those attempts, and different request digests. SDK retries were disabled; these are two distinct recorded requests, not one duplicate ledger record. The arm result is error/RuntimeError and no original report was produced. Because no SDK code is recorded, the artifacts cannot distinguish insufficient quota from a temporary rate limit.

## p04 shared-state and real-round verification

The canonical checkpoint hashes to `955a2f0425ef9cdb2d60f7a6dc9a013e05f9e398b01e8183712a0a53dc6c3f78`. The single result and all three dependent branch results report this same hash. The three copied call records in each branch exactly match the prefix's two decomposition responses and one verification response after removing only the `shared_first_round` marker. Thus nine ledger entries are explicit shared copies.

Both loop reports preserve the prefix analysis history and all prefix operations except its terminal `stopped` event. Both have verification rounds `[1, 2, 3]`, six document observations, six decomposition-adapter returns, and two unique eligible versions. Their resumed retrieval logs contain rounds `[2, 3]`.

- Continued psi makes seven new calls: four decomposition requests, two verification requests, and one extractor request. Its logical total is ten including the prefix.
- Frozen psi makes three new calls: two verification requests and one extractor request. Its four post-prefix decomposition-adapter returns reuse the stored analyses and make no semantic API requests. Its logical total is six.
- Independent makes two new decomposition calls before failure, with a logical total of five including the prefix. Its fresh trial starts at round 1 with no preceding trial's verification history.

The evidence decision remains `false` in the single and both completed loops at every verified round. One completed paired case with no decision change cannot establish an accuracy benefit from continuation or from psi.

## p04 provenance representation change

| Continued-psi round | m07 origin finding | `origin:p04` resolution | Open provenance gaps | Provenance status |
|---|---|---|---:|---|
| 1 | Present | Present | 0 | `original_material_located` |
| 2 | Removed | Removed | 1 | `partial` |
| 3 | Absent | Restored | 0 | `partial` |

Analysis revision 3 in `p04-loop_psi-report.json` returns empty origins, resolutions, gaps, and relations for m07. Its note says no current gap needs resolution. Call 4 had supplied the previous m07 analysis, including its origin and resolution, but the current gap list was empty. Full-analysis replacement at frozen `provenance.py:647` removes those findings; rebuilding reopens the initial origin gap. Revision 5 restores only the resolution. No origin path remains, so the final status stays partial even with no open gaps.

No new model-proposed gaps or relations appear, and the source materials are unchanged. Frozen psi keeps the original origin and resolution and retains `original_material_located` through all three rounds.

There is a concrete prompt-contract ambiguity: the frozen semantic prompt instructs resolving only currently named gaps and favors empty arrays over repeating earlier findings, while the engine interprets each analysis as a full replacement. The omission is consistent with interpreting the prompt as a request for incremental findings. Round 3's caution about original authorship and exact historical wording is reasonable given the selected-excerpt availability disclaimer, but the trace contains no explicit newly evidenced retraction establishing that removing the origin improved correctness. Report this as provenance representation instability, not proven semantic regression or improvement. No origin/edge correctness conclusion is made here.

A future semantic-interface change could explicitly require a complete replacement analysis retaining still-supported origins and resolutions, with source-grounded reasons for retractions. That would be a separate prompt/code change and experiment; it is not part of the instrumentation patch or this pilot.

## Failed-prefix logical-cost omission

The frozen `InvalidSharedPrefix` rows record `usage=0`. Zero new API requests is correct, but zero logical end-to-end cost omits the failed first-round computation that each dependent arm would have required.

The seven failed single prefixes together consumed 10 calls, 18,959 input tokens, and 23,958 output tokens. Charging this inherited work to each of three dependent arms adds 30 logical calls, 56,877 input tokens, and 71,874 output tokens. The frozen 165-call logical ledger therefore becomes 195 calls in an explicitly derived cost-corrected view. Actual new attempts remain 156, with 227,510 measured input tokens and 130,236 measured output tokens. Failed SDK attempts have no recorded token usage; none is invented.

The nine successful p04 prefix copies already account for 16,434 logical input tokens and 18,576 logical output tokens beyond actual-new usage. Adding the failed-prefix correction yields logical totals of 300,821 input tokens and 220,686 output tokens. These logical totals represent shared computation allocated to arms; they are not additional spending.

A future driver should retain zero `actual_new_api_usage` for skipped branches, attach explicitly labeled inherited-prefix usage, and populate their logical usage from the failed prefix. Use the saved `prefix_usage` if a checkpoint existed before the single-arm extractor failed; otherwise use the failed single's usage. This avoids incorrectly charging a single-only extractor to every continuation. Predictions and frozen results should remain unchanged. Any derived corrected view must disclose this post-run accounting adjustment. No efficiency claim should be made from this pilot.
