# Latest worktree synchronization — 2026-09-05

This commit synchronizes the v4 development worktree captured at 2026-09-05 22:23:11 UTC with the existing `codex/fix-layered-provenance-v03` branch. The branch name is historical; the current target-plan code implements decision-probe-v4.

## Scope

- Source, tests, experiment protocols, historical freeze-05 through freeze-09, and the full retained test fixtures and executed-code records are synchronized.
- The source snapshot contains 1,466 files; 788 previously absent paths and 15 modified paths differ from the prior GitHub head `3987fb5c7da6d9a40d3062fbf0fd776d568ecda7`.
- Existing remote history is preserved. Build outputs, old local wheels, package metadata, bytecode, caches, and secret environment files are excluded.
- This synchronization adds this note, a source-hash manifest, and a current-status pointer in README. No algorithm edits were made by the synchronization step.

## Verification on the exact source snapshot

- Python 3.12: `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -q` — **428 tests passed**, 2.445 seconds.
- Synthetic annotated policy benchmark — **22/22 expected behaviors matched**.
- All Python files parsed successfully without executing them.
- Secret-pattern scan found no API tokens, GitHub tokens, private-key blocks, AWS access keys, or credential-bearing URLs in the synchronized files. This is a targeted scan, not an assurance that arbitrary secrets can never exist.
- The previous GitHub snapshot ran 312 tests with 12 errors and one failure, all involving absent historical fixtures or freezes. The complete synchronized snapshot passes without deleting those tests.

## Historical artifacts are not fresh results

The zero-byte `reports/staged-dev-03/p08-report.json` is deliberately retained. Its run was interrupted, as recorded in `interruption.json`; it is not a successful p08 result and is not consumed by the current regression suite.

Historical v2 model results, freeze-11's v3 preregistration, and the old 0.2.0 release manifest remain historical. They are not v4 results. At snapshot time there is no freeze-12 file; the v4 comparison protocol is an implementation/design contract awaiting a matching completed freeze and a fresh live run. Its title does not constitute proof that preregistration is complete.

**No live model calls were made for this synchronization.** Offline test pass counts and synthetic policy cases are not real-news accuracy measurements. The shared worktree may receive subsequent changes in another chat; this commit records the timestamped, tested snapshot, not future edits.

The PR remains draft and `main` is not merged or changed by this operation.
