# Audit of fresh comparison 03

Review date: 2026-09-05. Reviewer: a separate AI agent within the execution team.
This is an execution-team AI review of the protocol and recorded run artifacts,
not independent human gold labeling or an external benchmark certification.
The reviewer made no model API calls, accessed no credentials, and changed no
benchmark inputs, gold labels, or implementation code.

All ten jobs completed successfully. The recorded exact-label score is supported:
original 5/5 and Accuracy 5/5, with zero errors. No extraction mismatch, changed
native Accuracy decision, source/input/gold mismatch, or evidence of response
replay was found in the checks below.

## Protocol and frozen artifacts

`config.json` is exactly equal as parsed JSON to the configuration of
`../direct-v03-fresh-01`. All 16 Accuracy implementation/adapter fingerprints
and all five original-agent fingerprints match the previous configuration.
The existing shared source archive at
`../direct-v03-fresh-01/executed-code` was checked against all 21 fingerprints;
every archived file matches. `repeat-contract.json` explicitly references that
archive. The current Accuracy source files also match, and the original source
files were checked before the new execution.

The archived input and gold files match fresh-01 byte for byte:

| Artifact | SHA-256 |
| --- | --- |
| `inputs-v03.json` | `4efef214c4c2d220407ee05c775945fad7377dc31328c07ccdad3842deb75cfc` |
| `gold-v03.json` | `a09220f0465dc0afa1f7504cad7370a805c8408b02631ba5dd983f84e023c850` |

The input hash agrees with `config.json`; the gold hash agrees with
`scores.json`. Each case's recorded input is identical across the two arms and
to its archived inference-input case. Four targets select evidence assessment;
one selects world assessment. Both arms have the complete eligible frozen corpus
available from the start. These examples contain only two or three snapshots,
within the configured document limit.

Source inspection confirms that `original_compare.run()` reads the inference
input, not the gold file. Gold is read by the separate scoring command. The
inference-input loader rejects unexpected top-level, case, and target fields.
The two pipelines retain their different prompts and internal procedures:
original runs unchanged at depth 1 through the corpus adapter, while Accuracy
allows up to five retrieval rounds. Both receive the same fixed task contract.
This is a comparison of those pipelines on supplied snapshots, with no web
collection evaluation.

## Output extraction and scoring

Every `*-result.json` equals its corresponding row in `results.json`. All five
case IDs occur exactly once per arm, all assessment modes match the frozen gold,
and all ten rows are marked completed. Rechecking the frozen labels reproduces
the case table and both 5/5 scores in `scores.json`.

For every job, the final recorded model request uses the same extraction system
prompt and output-schema hash. Its user payload contains only the target and
answer text. The recorded extraction output equals the saved extraction object;
every basis quote is nonempty and occurs verbatim in the answer. All saved
predictions equal their extracted labels.

For original, the extractor receives the exact raw `direct_response`. The
reviewer read each full response and checked its conclusion against the selected
assessment mode:

| Case | Original label | Meaning of the raw conclusion | Accuracy native and extracted label |
| --- | --- | --- | --- |
| `bridge-evidence` | false | The scoped notice contradicts reopening; real-world status remains unknown. | false |
| `library-evidence` | true | The scoped notice supports a 9 pm closing time; real-world execution is unverified. | true |
| `chain-evidence` | true | The scoped bulletin supports an 8 am opening time; the citation chain supplies no independent world confirmation. | true |
| `conflict-evidence` | disputed | The two scoped reports make opposing claims about ferry operation. | disputed |
| `world-unknown` | unverifiable | Textual support exists, but the selected world claim remains unestablished. | unverifiable |

The world-status caveats in the four evidence tasks do not change their selected
evidence conclusions. No extractor appears to have repaired or substituted an
original answer. The raw narrative's other assertions and source statistics are
outside this four-class score and have not been certified by this review.

For Accuracy, recomputing `present_decision()` and `round_decisions()` from every
saved report exactly reproduces its saved native decision and checkpoints. All
five reports are assessment-valid with no recorded report errors. Every
extracted label equals the explicit native decision. The extractor's input is
exactly the serialized native decision, including the separate evidence and
world assessments.

## Fresh calls and resource accounting

The call logs contain 110 request attempts and 110 successful, unique response
IDs. None overlaps the successful response IDs in fresh-01. Every recorded
actual model is `gpt-6-astra`; no call has an error record or cache marker.
Source inspection confirms that this runner constructs the live budget client
without the response-cache adapter, and the SDK has retries disabled.

Six short extraction output strings happen to equal earlier output strings.
All six carry new response IDs; all are extraction results, rather than reused
pipeline reports. Identical short labels or extraction text alone is not evidence
of replay. The code path and recorded IDs support the fresh-execution claim;
this review does not independently attest the remote provider's internals.

Within every job, sequence numbers are consecutive, the record count equals
`usage.model_calls`, and summed recorded input/output tokens equal the job usage.
Aggregating those jobs reproduces the score-file totals:

| Arm | Calls | Input tokens | Output tokens | Sum of per-job elapsed seconds |
| --- | ---: | ---: | ---: | ---: |
| Original | 84 | 74,748 | 41,093 | 781.436 |
| Accuracy | 26 | 38,578 | 18,137 | 331.121 |

Every job stays within 24 calls, 36,000 output tokens, and 600 seconds; every
call stays within its recorded output-token cap of at most 2,500. Extraction
calls are included. Input tokens are measured but uncapped. The seconds column
sums job elapsed times and is not the experiment's elapsed wall-clock time:
two jobs run concurrently. These totals establish lower recorded token/call
usage for Accuracy in this trial, without establishing a dollar-cost estimate,
equal feature coverage, or a general efficiency advantage.

## Verification rounds and interpretation

| Accuracy case | Retrieval-loop rounds recorded | Decomposition calls | Verification calls | Verified checkpoints |
| --- | ---: | ---: | ---: | ---: |
| `bridge-evidence` | 1 | 3 | 1 | 1 |
| `library-evidence` | 1 | 3 | 1 | 1 |
| `chain-evidence` | 1 | 5 | 1 | 1 |
| `conflict-evidence` | 2 | 2 | 1 | 1 |
| `world-unknown` | 2 | 3 | 1 | 1 |

Every case has exactly one verification call and one checkpoint, in round 1.
For conflict and world-unknown, round 2 queries the exhausted snapshot provider,
returns no new material, and ends without another verification. The configured
maximum of five rounds was not exercised. These results therefore provide no
observed accuracy gain from repeated verification.

All ten labels agree with the earlier successful fresh-01 run. They remain
repeated observations on the same five known, author-written synthetic examples,
not ten independent new questions or held-out validation. The ordering seed is
fixed, but model sampling is not seeded. Perfect agreement on these examples
does not establish general superiority, real-news accuracy, robustness, or
source-lineage accuracy. Shared model-based extraction remains a measurement
limitation despite the checks above.

The quota-blocked `../direct-v03-fresh-02` remains present with its error status
and no `scores.json`. It was not overwritten or silently assigned semantic
scores. This audit concerns the single new successful repetition in fresh-03.
