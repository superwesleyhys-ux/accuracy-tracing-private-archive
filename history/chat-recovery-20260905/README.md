# Cross-chat recovery — 2026-09-05

This directory preserves all seven news-tracking / accuracy-tracing delivery packages
found in the accessible saved chat files during this upload. Archives are byte-for-byte
copies; `ARCHIVE_INDEX.json` records their hashes.

## Current implementation

Use the repository source on `codex/fix-layered-provenance-v03`, based on
`a60995d174f180807b92c017e306c8e9981736b2` before this archival commit.
That snapshot contains 679 tracked files, including every path from the earlier
`codex/fix-layered-loop-v03` snapshot
`243827e47dde23db08c3ae88dd260b0f5384c066`; seven overlapping paths have newer content.
The branches have divergent commit history, so this is file coverage, not a claim
that their histories have been merged.

Important entry points:

- [Target decomposition and extension](../../experiments/extended_semantic.py)
- [Staged semantic analysis](../../experiments/staged_semantic.py)
- [Staged runner](../../experiments/staged_run.py)
- [Versioned provenance engine](../../newsverify/provenance.py)
- [Target-extension specification](../../docs/TARGET_EXTENSION.md)
- [Comparison scoring](../../experiments/target_extension_compare_score.py)

The existing [draft PR #2](https://github.com/superwesleyhys-ux/accuracy-tracing/pull/2)
remains draft. This upload does not merge it, update `main`, deploy code, or run
new paid model calls.

## Preserved deliverables

| Package | Purpose |
| --- | --- |
| [newsverify-harness-v0.1.0.zip](newsverify-harness-v0.1.0.zip) | Initial policy harness, examples, tests and specification |
| [accuracy-tracing-v0.2.0.zip](accuracy-tracing-v0.2.0.zip) | Versioned tracing, evaluation and comparison milestone |
| [accuracy-tracing-benchmark-review-20260905.zip](accuracy-tracing-benchmark-review-20260905.zip) | Benchmark-method audit and offline results |
| [original-vs-accuracy-comparison.zip](original-vs-accuracy-comparison.zip) | Controlled fault-injection comparison and source hashes |
| [direct-ab-comparison.zip](direct-ab-comparison.zip) | Initial direct A/B adapter, tests, inputs and preflight |
| [direct-ab-live-results.zip](direct-ab-live-results.zip) | Later adapter and preserved real API requests/results |
| [accuracy-regression-diagnosis.zip](accuracy-regression-diagnosis.zip) | Latest saved legacy adapter plus regression diagnosis and replay |

Older snapshots and manifests can differ from the current repository. Do not
extract them over the latest source. The legacy diagnostic layout expects sibling
`direct-ab/` and `repo-comparison/{original,accuracy}/` directories. The documented
original commit is `789506115e7dfef1f2359c2e43bcae4234b8d3f2`;
the legacy Accuracy commit is `aca102cfe13379c5924a5ffa7b60337351e4eaaa`.
A directory named `accuracy-tracing/` must not silently substitute for that
historical `accuracy/` layout.

## Result provenance and limits

The old two-case 50% versus 0% label-agreement report belongs to the legacy direct
A/B experiment only. It is not a score for the later staged or target-extension
code already present on this branch. Follow each newer report's frozen source,
dataset and inference-run identifiers; this archival upload produces no new
accuracy claim.

Validation checked ZIP integrity and paths (including the nested wheel),
parsed 43 Python members and 122 JSON members, and found no matches for the
credential patterns in the upload audit. This is a pattern scan, not a guarantee
that arbitrary private information can never be present. No credentials,
account screenshots, environments, caches or unrelated project files were selected.

Coverage is limited to saved deliverables that were recoverable and existing
GitHub branch content. Unsaved files in another chat's running environment cannot
be certified as included.
