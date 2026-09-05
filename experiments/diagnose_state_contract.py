"""Synthetic offline state-contract probes; not accuracy measurements or live failures.

Run from the repository: python experiments/diagnose_state_contract.py
Only imports local newsverify.provenance. No model client, network, or credentials.
Prints JSON; optional --output writes that same JSON to the specified file.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    from newsverify import provenance as p

    target = p.Target("synthetic-target", "What do the supplied bridge notices report?",
                      "2026-09-04T20:00:00Z", "a", assessment_mode="evidence")

    def material(identifier, content):
        return p.MaterialVersion(identifier, "https://example.org/" + identifier, content,
            "2026-09-04T12:00:00Z", "2026-09-04T10:00:00Z", "2026-09-04T10:00:00Z",
            "Synthetic exact-version archive fixture")

    a = material("a", "Notice A says the bridge is closed.")
    b = material("b", "Notice B says the bridge is open.")

    def full(value):
        return p.Span(value.version_id, 0, len(value.content), value.content)

    class SameIDDecomposer:
        def decompose(self, target, value, context):
            # Each finding is independently faithful to its own source. The
            # intentionally conflicting IDs, not bad quotations, are the probe.
            return p.Analysis(fragments=(p.Fragment("same-id", value.content,
                                                   full(value), target.id),))

    def conflicting_ids(rounds):
        result = p.run_provenance(target, p.ReplayTraceProvider(rounds), SameIDDecomposer(),
            config=p.TraceConfig(max_rounds=len(rounds), experimental_force_rounds=True))
        return {"errors": result["errors"], "assessment_valid": result["assessment_valid"],
                "material_contents": {v["version_id"]: v["content"] for v in result["materials"]},
                "accepted_analyses": result["analyses"], "projected_fragments": result["fragments"],
                "observations": result["observations"]}

    before = conflicting_ids([[a, b]])
    after = conflicting_ids([[a, b], [a]])

    class OtherMaterialFragment:
        def decompose(self, target, value, context):
            if value.version_id == "a":
                return p.Analysis()
            return p.Analysis(fragments=(p.Fragment("b-owned-a-fragment", a.content,
                                                   full(a), target.id),))

    other = p.run_provenance(target, p.ReplayTraceProvider([[a, b]]), OtherMaterialFragment(),
                            config=p.TraceConfig(max_rounds=1))

    class UnknownResolutionDecomposer:
        def decompose(self, target, value, context):
            return p.Analysis(resolutions=(p.Resolution("never-registered-gap", (full(value),),
                                                       "Synthetic resolution of an absent gap ID."),))

    unknown_psi = p.run_provenance(target, p.ReplayTraceProvider([[a]]), UnknownResolutionDecomposer(),
                                 config=p.TraceConfig(max_rounds=1))

    class UnknownResolutionVerifier:
        def verify(self, target, context):
            return p.VerificationResult(resolutions=(p.Resolution("never-registered-gap", (full(a),),
                                                       "Synthetic resolution of an absent gap ID."),))

    unknown_verifier = p.run_provenance(target, p.ReplayTraceProvider([[a]]),
        verifier=UnknownResolutionVerifier(), config=p.TraceConfig(max_rounds=1))

    def resolution_result(result):
        return {"errors": result["errors"], "assessment_valid": result["assessment_valid"],
                "open_gap_ids": [g["id"] for g in result["gaps"]], "resolutions": result["resolutions"]}

    output = {
        "scope": "Synthetic offline probes, not observed live failure counts or semantic accuracy tests.",
        "module": str(Path(p.__file__).resolve()),
        "duplicate_id_overwrite": {
            "interpretation": "Two faithful, source-owned fragments with one ID silently collapse. Reobserving unchanged a changes the winner even though neither owner's Analysis changes.",
            "before_repeat": before, "after_repeat": after,
            "owner_analyses_unchanged": before["accepted_analyses"] == after["accepted_analyses"],
            "winner_changed": before["projected_fragments"] != after["projected_fragments"],
        },
        "other_material_fragment": {
            "interpretation": "Accepted: b's Analysis owns a fragment whose decisive span comes only from a. Whether this is allowed must be defined explicitly; arbitrary cross-material relation bases are not inherently invalid.",
            "errors": other["errors"], "assessment_valid": other["assessment_valid"],
            "analysis_for_b": other["analyses"]["b"],
        },
        "unknown_gap_resolution_by_decomposer": resolution_result(unknown_psi),
        "unknown_gap_resolution_by_verifier": resolution_result(unknown_verifier),
        "caveats": [
            "A legitimate relation may cite multiple materials. The probes do not propose banning cross-material evidence.",
            "If fragment IDs intentionally identify shared findings, conflicting definitions still need an explicit merge or conflict rule; last writer wins is not that rule.",
            "An unknown gap ID has no registered stage, so no claim about its actual stage can be inferred from its spelling.",
            "All seven observed failed shared prefixes stopped before verification; these probes identify separate latent defects.",
        ],
    }
    encoded = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
