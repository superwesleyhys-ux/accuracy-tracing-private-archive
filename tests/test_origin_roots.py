"""Confirmed origin roots are reachable terminal documentary candidates."""
import unittest

from newsverify.provenance import (
    Analysis, MaterialVersion, OriginFinding, Relation, ReplayTraceProvider,
    Resolution, Span, Target, TraceConfig, run_provenance,
)


STAMP = "2026-09-04T12:00:00Z"


def material(identifier):
    return MaterialVersion(identifier, "https://example.org/" + identifier,
        "Documentary material " + identifier + ".", STAMP, STAMP, STAMP,
        "Synthetic exact-version archive.")


def span(value):
    return Span(value.version_id, 0, len(value.content), value.content)


def origin(target, value, kind="original_record"):
    return OriginFinding(target.id, value.version_id, (span(value),), kind,
                         "Synthetic candidate original declaration.")


def edge(source, upstream, identifier):
    return Relation(identifier, source.version_id, upstream.version_id, "cites", "direct",
                    (span(source),), "The source-side fixture directly cites the upstream.")


def make_target(identifier="p08"):
    return Target(identifier, "The downstream report attributes a result to original records.",
                  "2026-09-05T00:00:00Z", source_version_id="source",
                  assessment_mode="evidence", evidence_scope=("source",))


class GraphDecomposer:
    def __init__(self, mapping):
        self.mapping = mapping

    def decompose(self, frozen_target, value, context):
        result = self.mapping[value.version_id]
        return result(frozen_target, value, context) if callable(result) else result


class OriginRootTests(unittest.TestCase):
    @staticmethod
    def run_case(rounds, mapping, identifier="p08", max_rounds=2):
        return run_provenance(make_target(identifier), ReplayTraceProvider(rounds),
            GraphDecomposer(mapping), config=TraceConfig(max_rounds=max_rounds))

    def test_p08_shape_keeps_only_terminal_upstream_candidate(self):
        announcement, press_release = material("m14"), material("m15")
        claim = Target("p08",
            "The USGS Unified Geologic Map of the Moon released in 2020 combined six "
            "Apollo-era regional maps with newer lunar-mission data.",
            "2026-09-05T00:00:00Z", source_version_id="m14",
            assessment_mode="evidence", evidence_scope=("m15",))
        report = run_provenance(claim, ReplayTraceProvider(((press_release, announcement),)),
            GraphDecomposer({
                "m15": Analysis(origins=(origin(claim, press_release),)),
                "m14": Analysis(relations=(edge(announcement, press_release,
                                                  "announcement-to-release"),),
                                origins=(origin(claim, announcement),)),
            }), config=TraceConfig(max_rounds=1))

        self.assertEqual(["m15"], [item["version_id"] for item in report["origins"]])
        raw_candidates = {item["version_id"] for revision in report["analysis_history"]
                          for item in revision["analysis"]["origins"]}
        self.assertEqual({"m14", "m15"}, raw_candidates)

    def test_parallel_terminal_roots_are_all_preserved(self):
        source, left, right = (material(name) for name in ("source", "left", "right"))
        claim = make_target("parallel")
        report = self.run_case(((left, right, source),), {
            "left": Analysis(origins=(origin(claim, left),)),
            "right": Analysis(origins=(origin(claim, right),)),
            "source": Analysis(relations=(edge(source, left, "source-left"),
                                           edge(source, right, "source-right"))),
        }, identifier="parallel", max_rounds=1)

        self.assertEqual({"left", "right"},
                         {item["version_id"] for item in report["origins"]})

    def test_disconnected_candidate_is_not_confirmed_and_leaves_lineage_gap(self):
        source, stray = material("source"), material("stray")
        claim = make_target("disconnected")
        report = self.run_case(((stray, source),), {
            "stray": Analysis(origins=(origin(claim, stray),)),
            "source": Analysis(),
        }, identifier="disconnected", max_rounds=1)

        self.assertEqual([], report["origins"])
        self.assertIn("lineage:disconnected", {item["id"] for item in report["gaps"]})
        raw = [item for revision in report["analysis_history"]
               for item in revision["analysis"]["origins"]]
        self.assertEqual(["stray"], [item["version_id"] for item in raw])

    def test_reachable_candidate_cycle_has_no_terminal_root_and_leaves_gap(self):
        source, left, right = (material(name) for name in ("source", "left", "right"))
        claim = make_target("cycle")

        def right_analysis(frozen_target, value, context):
            relations = ()
            resolutions = ()
            if any(item["version_id"] == "left" for item in context["materials"]):
                relations = (edge(right, left, "right-left"),)
                resolutions = (Resolution("lineage:cycle", (span(right),),
                                          "The cyclic candidate claimed closure."),)
            return Analysis(relations=relations, origins=(origin(claim, right),),
                            resolutions=resolutions)

        report = self.run_case(((right, left, source), (right,)), {
            "right": right_analysis,
            "left": Analysis(relations=(edge(left, right, "left-right"),),
                             origins=(origin(claim, left),)),
            "source": Analysis(relations=(edge(source, left, "source-left"),)),
        }, identifier="cycle", max_rounds=2)

        self.assertEqual([], report["origins"])
        self.assertIn("lineage:cycle", {item["id"] for item in report["gaps"]})
        self.assertNotIn("lineage:cycle", {item["gap_id"] for item in report["resolutions"]})
        self.assertNotEqual("original_material_located", report["provenance_status"])


if __name__ == "__main__":
    unittest.main()
