# Real-source claim-probe pilot annotation protocol

This is a new eight-event pilot derived from 15 authentic primary agency pages. It is not a held-out benchmark. Targets are AI-authored probes, including deliberate date, quantity, or baseline mutations. None is asserted to be a real published false news article. The source materials remain authentic excerpts. No model API calls or credentials were used to build it.

Every label has `annotation_status=provisional_ai_review_required`. Initial annotation was performed by an AI agent. A separate AI agent reviews sources and entailment without model predictions. That review is a quality check, not independent human adjudication.

## Files and inference boundary

- `inputs.json` contains only `schema_version` and inference cases with `target`, `materials`, `seed_ids`, and `full_evidence_control`.
- `gold.json` contains decisions, event clusters, rationales, basis quotations and evaluability flags.
- `sources.json` contains real URLs, publication dates, retrieval date, location descriptions, exact excerpt hashes, and verified links.
- IDs `p01`–`p08` and `m01`–`m15` are neutral. Labels and categories are not included in inference inputs.

## Temporal contract

Collection completed at 2026-09-05T09:34:36Z. All supplied versions become available at that time, and all targets use that as-of time. The task therefore evaluates the current collected excerpts about date-qualified historical announcements. It does not replay what was available to a reader in 2020–2025.

The source-displayed calendar date is retained verbatim as `publication_date` in source metadata and a date header in material content. `published_at` is null because an exact timezone-qualified publication instant was not verified. A date or a timezone-free clock is not converted into a fabricated UTC timestamp. Known current-page update dates are retained separately. There is no WARC, archive snapshot, or original HTML capture, and no historical version claim.

## Decision rubric

- `true` / supported: the named evidence scope explicitly entails the full date-qualified target.
- `false` / contradicted: the scope explicitly supplies an incompatible date, quantity, or baseline. Absence of support alone is insufficient.
- `unverifiable` / unresolved: the scope leaves the target open, or the world claim is not established by the supplied records.
- Evidence-mode labels mean what the named excerpts establish. They do not certify the world claim.

The seven evidence cases use explicit document scopes. The Europa case is world mode and deliberately remains unknown: potential habitability does not establish organisms' existence or absence.

## Provenance rubric

`origins` is a list of acceptable alternatives, with one version sufficient. For this pilot, it identifies a corpus-relative documentary source of a supported assertion reached through an observed citation. It is not the oldest source anywhere on the internet or the creator of an underlying measurement. Each evaluable origin currently has one acceptable version.

No published origin is manufactured for a counterfactual or unresolved AI-authored target. Such cases have `origin_evaluable=false`. Editorial probe derivation is not represented as a factual article citation.

`edges` contains triples `[from_version, to_version, "cites"]`. The four labeled edges were observed as real links in opened source pages and clicked to verify their destinations. Two retained original hrefs redirect to the canonical URL; source metadata records both. These edges describe the supplied excerpts only. No quote edge is claimed just because two sources use similar text.

For cases with no verified path, `edges_evaluable=false`; an empty list is not a claim that no possible edges exist. GOES has a verified documentary citation but a deliberately false date probe, so its edge is evaluable and its target origin is not. There are three evaluable origins and four evaluable edge cases.

The Bennu February article's old `blogs.nasa.gov` link currently redirects to a general archive rather than the migrated October article. Both canonical sources were opened independently, but no citation edge is labeled. The NASA and NOAA temperature reports are separate institutional estimates; no direct dependency is invented. The two Europa sources share a publisher and are not counted as independent witnesses.

## Selection and controls

Eight cases correspond to eight distinct events: DART period estimate; Bennu sample-mass accounting; first finalized NIST PQC standards; 2024 annual global temperature reporting; Ingenuity's final flight; Europa Clipper launch; GOES-U renaming; and the USGS lunar map release. There are three supported, three contradicted, and two unresolved decisions.

Seven events have two authentic materials; Ingenuity uses one original agency notice because it explicitly separates the helicopter's later upright state from the still-investigated touchdown orientation. Its unresolved label is limited to that record and does not claim that all later investigation remained unresolved.

The supplied source pool is frozen. All two-material cases request the full-evidence control; the two world-case seeds already include both sources. Seed choice is a retrieval-access condition, not evidence of hidden web coverage. Event clusters must remain intact in any future split, and these eight pilot cases must never be relabeled as an untouched test set.

## Human review before stronger claims

A qualified reviewer should inspect the linked originals, confirm the exact scope and quotations, decide whether each provisional label is defensible, and check that no current excerpt is misrepresented as a historical capture. Changes must update content hashes. Report annotation uncertainty and excluded provenance dimensions. Any future validation claim must distinguish pilot agreement from representative out-of-sample accuracy.

