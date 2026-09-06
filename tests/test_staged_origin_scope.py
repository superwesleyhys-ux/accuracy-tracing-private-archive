"""Target-level origin scope regressions derived from the p04 NOAA/NASA case."""
from dataclasses import asdict
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
s = importlib.import_module("staged_semantic")
p = s.p


NOAA = """Assessing the Global Climate in 2024
Publication date: 2025-01-10

In 2024, global surface temperature was 2.32°F (1.29°C) above the 20th-century average.
In 2024, global temperature exceeded the pre-industrial (1850–1900) average by 2.63°F (1.46°C)."""
NASA = """Temperatures Rising: NASA Confirms 2024 Warmest Year on Record
Publication date: 2025-01-10

NASA scientists further estimate Earth in 2024 was about 2.65 degrees Fahrenheit (1.47 degrees Celsius) warmer than the mid-19th century average (1850-1900)."""
NASA_ESTIMATE = ("NASA scientists further estimate Earth in 2024 was about 2.65 degrees "
                 "Fahrenheit (1.47 degrees Celsius) warmer than the mid-19th century "
                 "average (1850-1900).")


def material(identifier, content, issuer):
    return p.MaterialVersion(identifier, "https://example.org/" + identifier, content,
        "2026-09-05T09:34:36Z", None, "2026-09-05T09:34:36Z",
        "Synthetic exact-version fixture available at collection.", issuer)


def target():
    return p.Target("p04",
        "NOAA's January 10, 2025 annual climate summary reported that the global surface "
        "temperature in 2024 was 1.29°C above the 1850–1900 average.",
        "2026-09-05T09:34:36Z", source_version_id="m07",
        assessment_mode="evidence", evidence_scope=("m07",))


def plan_views(value):
    payload = asdict(value)
    payload["evidence_scope"] = list(payload["evidence_scope"])
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
        separators=(",", ":")).encode()).hexdigest()
    return {stage: {"target_signature": signature, "plan_sha256": "f" * 64,
                    "stage": stage, "claims": [], "probes": []}
            for stage in ("atoms", "lineage", "critic", "evidence", "world")}


class ScriptClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def call(self, system, user, schema):
        prompts = (("atoms", s.ATOMS_PROMPT), ("lineage", s.LINEAGE_PROMPT),
                   ("critic", s.CRITIC_PROMPT))
        stage = next(name for name, prompt in prompts if system.startswith(prompt))
        expected, response = self.responses.pop(0)
        if stage != expected:
            raise AssertionError(f"expected {expected}, received {stage}")
        self.calls.append({"stage": stage, "payload": json.loads(user)})
        return deepcopy(response)


class StagedOriginScopeTests(unittest.TestCase):
    def test_p04_out_of_scope_child_origin_is_repaired_to_null(self):
        claim, noaa, nasa = target(), material("m07", NOAA, "NOAA"), material("m08", NASA, "NASA")
        context = {"target": asdict(claim), "materials": [asdict(noaa), asdict(nasa)],
            "analyses": {}, "fragments": [], "relations": [], "origins": [], "gaps": [],
            "gap_registry": [], "verification_history": [], "usage": {"rounds": 1}}
        bad_origin = {"kind": "original_record", "quote": NASA_ESTIMATE,
            "rationale": ("NASA produced this different estimate, not the comparison "
                          "attributed to NOAA by the target.")}
        atom = {"atoms": [{"statement": "NASA published a different 1.47°C estimate.",
            "quote": NASA_ESTIMATE,
            "qualifier_quotes": ["1.47 degrees Celsius", "1850-1900"]}], "notes": ""}
        client = ScriptClient(
            ("atoms", atom),
            ("lineage", {"citations": [], "origin": bad_origin, "notes": ""}),
            ("lineage", {"citations": [], "origin": None,
                         "notes": "The NASA estimate is not a target-level NOAA origin."}),
            ("critic", {"decision": "accept", "stage": "none", "quote": "", "issue": ""}),
        )
        decomposer = s.StagedDecomposer(client, target_plan=plan_views(claim), max_repairs=1)
        analysis = decomposer.decompose(claim, nasa, context)

        self.assertEqual((), analysis.origins)
        self.assertEqual(["atoms", "lineage", "lineage", "critic"],
                         [item["stage"] for item in client.calls])
        self.assertEqual("structure_repair_requested", decomposer.history[1]["status"])
        repair = client.calls[2]["payload"]["repair"]
        self.assertIn("disconnected", repair["issue"])
        self.assertEqual("m08", client.calls[2]["payload"]["material"]["version_id"])

    def test_core_excludes_disconnected_origin_from_final_graph(self):
        claim, noaa, nasa = target(), material("m07", NOAA, "NOAA"), material("m08", NASA, "NASA")

        class Decomposer:
            def decompose(self, frozen_target, value, context):
                if value.version_id == "m08":
                    start = value.content.index(NASA_ESTIMATE)
                    basis = p.Span("m08", start, start + len(NASA_ESTIMATE), NASA_ESTIMATE)
                    return p.Analysis(origins=(p.OriginFinding(
                        frozen_target.id, "m08", (basis,), "original_record",
                        "This is an origin only for NASA's different estimate."),))
                return p.Analysis()

        report = p.run_provenance(claim, p.ReplayTraceProvider(((noaa, nasa),)), Decomposer())
        self.assertEqual([], report["errors"])
        self.assertEqual([], report["origins"])
        self.assertTrue(report["analysis_history"][1]["analysis"]["origins"])
        self.assertTrue(any(item["id"] == "lineage:p04" for item in report["gaps"]))


if __name__ == "__main__":
    unittest.main()
