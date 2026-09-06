# Provisional pilot source and gold review

**Result: passed for a provisional pilot freeze, with no blocking source or label issues.** All eight decisions are defensible under the declared evidence scopes. Gold remains provisional pending human review.

This review was performed by a separate execution-team AI agent. The reviewer saw proposed labels and rationales, reopened the primary sources independently, and clicked the four retained citation links. No live model predictions, model API calls, or credentials were used. This is not independent human labeling, blinded initial annotation, or validation of real-news accuracy.

## Case findings

| Case | Reviewed decision | Evidence and scope |
| --- | --- | --- |
| p01 | true | The October 11, 2022 announcement states a 32-minute shortening with approximately ±2 minutes uncertainty. A later revision to 33 minutes does not negate what that announcement reported. [m01](https://www.nasa.gov/science-research/planetary-science/from-impact-to-innovation-a-year-of-science-and-triumph-for-historic-dart-mission/), [m02](https://www.nasa.gov/news-release/nasa-confirms-dart-mission-impact-changed-asteroids-motion-in-space/) |
| p02 | false | The February 15, 2024 record states 121.6 grams as the bulk total and distinguishes the previously measured 70.3 grams. The exact final-total assertion is explicitly contradicted. [m03](https://science.nasa.gov/blogs/osiris-rex/2024/02/15/nasa-announces-osiris-rex-bulk-sample-mass/), [m04](https://science.nasa.gov/blogs/osiris-rex/2023/10/20/nasas-osiris-rex-achieves-sample-mass-milestone/) |
| p03 | true | The official FIPS 203 abstract names three ML-KEM parameter sets: 512, 768, and 1024. This concerns parameter sets within FIPS 203, separate from the number of finalized standards. [m05](https://www.nist.gov/news-events/news/2024/08/nist-releases-first-3-finalized-post-quantum-encryption-standards), [m06](https://csrc.nist.gov/pubs/fips/203/final) |
| p04 | false | NOAA assigns 1.29°C to its 20th-century baseline and 1.46°C to 1850–1900. NASA separately reports 1.28°C against 1951–1980 and about 1.47°C against 1850–1900; that other estimate does not answer the named NOAA scope. [m07](https://www.ncei.noaa.gov/news/global-climate-202413), [m08](https://www.nasa.gov/news-release/temperatures-rising-nasa-confirms-2024-warmest-year-on-record/) |
| p05 | unverifiable | The January 25 record describes the helicopter as upright after the flight but explicitly leaves its orientation at touchdown under investigation. Later upright position entails neither upright nor non-upright orientation at the earlier moment. This is a document-scoped unknown. [m09](https://www.nasa.gov/news-release/after-three-years-on-mars-nasas-ingenuity-helicopter-mission-ends/) |
| p06 | unverifiable | Habitability and searching for conditions that could support life establish neither the existence nor absence of organisms. The mission-capability statement is not a biological detection result. [m10](https://www.nasa.gov/news-release/liftoff-nasas-europa-clipper-sails-toward-ocean-moon-of-jupiter/), [m11](https://www.nasa.gov/podcasts/curious-universe/europa-clippers-voyage-to-jupiters-ocean-moon/) |
| p07 | false | The July 8 bulletin dates renaming to July 7, 2024, upon attaining geostationary orbit. June 25 was launch day. The retrospective links to the bulletin but is not independent corroboration. [m12](https://www.nesdis.noaa.gov/news/noaas-goes-u-satellite-one-year-later), [m13](https://www.nesdis.noaa.gov/news/noaas-goes-u-reaches-geostationary-orbit-now-designated-goes-19) |
| p08 | true | The April 20, 2020 national release explicitly describes combining six Apollo-era regional maps with newer satellite-mission information. The center article links to that release; the mapped data have other contributors. [m14](https://www.usgs.gov/news/astrogeology-releases-new-map-moon), [m15](https://www.usgs.gov/news/national-news-release/usgs-releases-first-ever-comprehensive-geologic-map-moon) |

## Provenance and time

The three labeled origins are corpus-relative documentary sources for supported assertions: m02, m06, and m15. They are not claims about the earliest publication anywhere or proof of the underlying world facts. No original publication is invented for a false or unresolved AI-authored probe.

| Observed citation | Verification |
| --- | --- |
| m01 → m02 | Reviewer opened the origin, found the named link, and clicked it. The original href redirects to the recorded canonical page. |
| m05 → m06 | Reviewer opened the origin, found the named link, and clicked it. The linked destination matches the source record. |
| m12 → m13 | Reviewer opened the origin, found the named link, and clicked it. The linked destination matches the source record. |
| m14 → m15 | Reviewer opened the origin, found the named link, and clicked it. The original href redirects to the recorded canonical page. |

The Bennu February article’s old October link currently lands on a general archive. That issue was raised before freeze; the final dataset omits the edge and documents the limitation. NASA and NOAA temperature estimates are kept separate. The two Europa records and the linked institutional articles are not presented as independent witnesses.

All 15 materials use the current collection time for availability. Source-displayed publication dates are retained separately; published_at remains null when a timezone-qualified instant was not verified. The excerpts do not constitute historical exact-version captures, and current page modification dates do not establish when each selected sentence first appeared.

## File checks and limits

- Checked all 15 selected source excerpts and their displayed publication dates against opened primary pages.
- Verified all 15 excerpt hashes and all 12 gold basis quotations as exact substrings of the corresponding supplied materials.
- Confirmed eight distinct event IDs, neutral p01–p08 and m01–m15 identifiers, and no annotation labels or rationales in the inference input schema.
- Confirmed three supported, three contradicted, and two unresolved decisions; three origin dimensions and four citation-edge dimensions are evaluable.
- The actual runtime prompt boundary was not reviewed here. The driver must continue to keep separate gold and review files out of model requests.

The final Ingenuity probe is about orientation at touchdown within the January 25 record. It is not the earlier candidate about the accident’s cause, and it makes no claim that all subsequent investigation remained unresolved. Europa’s unknown result means neither biological presence nor biological absence has been established.

This small, purposively assembled corpus has AI-authored mutations and selected excerpts. It is suitable for an explicitly provisional mechanics and agreement pilot. Human review and a genuinely held-out, broader corpus are needed before stronger accuracy claims.

## Reviewed file identity

Review completed: 2026-09-05T09:40:10Z. Hashes are SHA-256 over exact file bytes. Any change to these files after review requires a matching review update before the reviewed state can be claimed.

| File | SHA-256 |
| --- | --- |
| inputs.json | 4b1e1d3bbad7d72f77822aa8cf402e923db1af9d85df3b5c732a5279ac2fa825 |
| gold.json | 4cd818aecf0693559c2b877a7067a7b765159abc4b6b832535fb6a14add13ce5 |
| sources.json | 2869b635d092d556edf35f0d57b2d150cca0728f80516614a8faaf74df8c989c |
| ANNOTATION.md | 96d3564170a85830fefb8066efa59f100873f2585a6da7b8d719a27359562777 |
