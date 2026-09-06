"""Offline contracts for the split-prompt validation loop; no model API use."""
from copy import deepcopy
from dataclasses import asdict, is_dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from newsverify import provenance as p
from newsverify.decisions import present_decision
from newsverify.retrieval import SnapshotSearchProvider
from prompt_specs import PROMPT_VERSION, prompt_manifest
from staged_semantic import StagedDecomposer, StagedSemanticError, StagedVerifier
from target_plan import build_target_plan, dimension_evidence_is_grounded


ATOMS = {"atoms": [{"statement": "The bridge remains closed until Friday.",
                     "quote": "The bridge remains closed until Friday.",
                     "qualifier_quotes": ["remains closed", "until Friday"]}],
         "notes": ""}
NO_LINEAGE = {"citations": [], "origin": None, "revisit": [], "notes": ""}
DECOMPOSITION_ACCEPT = {"decision": "accept", "stage": "none", "quote": "", "issue": ""}
LAYER_CRITIC_ACCEPT = {"decision": "accept", "issue": "", "basis": []}
DEFAULT_DIMENSIONS = ["actor_subject", "predicate_object", "scope_location", "time"]


def audit_digest(value):
    def default(item):
        if is_dataclass(item):
            return asdict(item)
        raise TypeError

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), default=default)
    return hashlib.sha256(encoded.encode()).hexdigest()


def layer(verdict="unresolved", basis=(), gaps=(), resolutions=(), rationale="No conclusion yet."):
    stop = "none" if verdict != "unresolved" or gaps else "no_source_lead"
    pool = [{"version_id": version, "quote": quote} for version, quote in basis]
    dimensions = [{"dimension": dimension, "verdict": verdict,
                   "basis_indices": list(range(len(pool))),
                   "rationale": rationale}
                  for dimension in DEFAULT_DIMENSIONS]
    return {"probe_results": [{"probe_number": 1, "basis_pool": pool,
                                "dimension_results": dimensions,
            "gaps": list(gaps), "resolutions": list(resolutions),
            "stop_reason": stop}]}


class ScriptClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def call(self, system, user, json_schema=None, json_mode=False, **metadata):
        if not self.responses:
            raise AssertionError("unexpected staged call")
        expected_stage, response = self.responses.pop(0)
        self.calls.append({"stage": metadata.get("stage"),
                           "prompt_version": metadata.get("prompt_version"),
                           "payload": json.loads(user), "schema": json_schema,
                           "system": system})
        if metadata.get("stage") != expected_stage:
            raise AssertionError(f"expected {expected_stage}, got {metadata.get('stage')}")
        return deepcopy(response)


class StagedSemanticTests(unittest.TestCase):
    def fixture(self, content="The bridge remains closed until Friday.", version="m"):
        material = p.MaterialVersion(
            version, "https://example.org/" + version, content,
            "2026-09-04T20:00:00Z", "2026-09-04T10:00:00Z",
            "2026-09-04T10:00:00Z", "Fixture snapshot")
        target = p.Target(
            "claim", "The bridge reopened before Friday.",
            "2026-09-04T20:00:00Z", source_version_id=version,
            assessment_mode="evidence", evidence_scope=(version,))
        return target, material

    def context(self, target, material, *, fragments=(), gaps=()):
        return {"target": asdict(target), "materials": [asdict(material)],
                "analyses": {}, "fragments": list(fragments), "relations": [],
                "origins": [], "gaps": list(gaps), "gap_registry": list(gaps),
                "verification_history": [], "usage": {"rounds": 1},
                "assessments": None}

    def test_split_stages_feed_canonical_fragments_into_both_verification_layers(self):
        target, material = self.fixture()
        quote = material.content
        client = ScriptClient(
            ("atoms", ATOMS), ("lineage", NO_LINEAGE),
            ("decomposition_critic", DECOMPOSITION_ACCEPT),
            ("evidence", layer("contradicted", (("m", quote),),
                               rationale="The scoped record says the bridge remains closed.")),
            ("evidence_critic", LAYER_CRITIC_ACCEPT),
            ("world", layer()), ("world_critic", LAYER_CRITIC_ACCEPT))
        report = p.run_provenance(
            target, SnapshotSearchProvider([material], ["m"]),
            StagedDecomposer(client), StagedVerifier(client),
            p.TraceConfig(max_rounds=1))
        self.assertEqual([], report["errors"])
        self.assertEqual("false", present_decision(report)["decision"])
        self.assertEqual(["atoms", "lineage", "decomposition_critic",
                          "evidence", "evidence_critic", "world", "world_critic"],
                         [call["stage"] for call in client.calls])
        self.assertTrue(all(call["prompt_version"] == PROMPT_VERSION
                            for call in client.calls))
        self.assertTrue(all("retrieved_at" not in json.dumps(call["payload"])
                            for call in client.calls))
        evidence_payload = client.calls[3]["payload"]
        expected_fragments = [{
            "parent_id": report["fragments"][0]["parent_id"],
            "span": report["fragments"][0]["span"],
            "qualifier_spans": report["fragments"][0]["qualifier_spans"],
        }]
        self.assertEqual(json.loads(json.dumps(expected_fragments)),
                         evidence_payload["fragments"])
        self.assertNotIn("qualifiers", evidence_payload["fragments"][0])
        self.assertNotIn("text", evidence_payload["fragments"][0])
        self.assertEqual(["remains closed", "until Friday"],
                         [item["quote"] for item in
                          evidence_payload["fragments"][0]["qualifier_spans"]])
        self.assertEqual(1, len(evidence_payload["target_plan"]["probes"]))

    def test_decomposition_critic_repairs_only_named_stage_then_rechecks_both(self):
        target, material = self.fixture()
        repaired = deepcopy(ATOMS)
        repaired["notes"] = "Rechecked the time boundary."
        client = ScriptClient(
            ("atoms", ATOMS), ("lineage", NO_LINEAGE),
            ("decomposition_critic", {
                "decision": "repair", "stage": "atoms", "quote": material.content,
                "issue": "Recheck the Friday time boundary."}),
            ("atoms", repaired),
            ("decomposition_critic", DECOMPOSITION_ACCEPT))
        adapter = StagedDecomposer(client, max_repairs=1)
        result = adapter.decompose(target, material, self.context(target, material))
        self.assertEqual("Rechecked the time boundary.", result.notes)
        self.assertEqual(["atoms", "lineage", "decomposition_critic", "atoms",
                          "decomposition_critic"],
                         [call["stage"] for call in client.calls])
        self.assertEqual(1, client.calls[3]["payload"]["repair"] is not None)

    def test_decomposition_assembly_hash_binds_every_decision_input(self):
        target, material = self.fixture()
        context = self.context(target, material)
        client = ScriptClient(
            ("atoms", ATOMS), ("lineage", NO_LINEAGE),
            ("decomposition_critic", DECOMPOSITION_ACCEPT))
        adapter = StagedDecomposer(client)
        adapter.decompose(target, material, context)
        record = next(item for item in adapter.history
                      if item["stage"] == "decomposition_assembly")
        source = asdict(material)
        source.pop("retrieved_at")
        plan = build_target_plan(target)
        expected = {
            "target": asdict(target),
            "source": source,
            "visible": {material.version_id: source},
            "context": context,
            "registry": {},
            "target_plan": plan,
            "drafts": {"atoms": ATOMS, "lineage": NO_LINEAGE},
        }
        self.assertEqual(audit_digest(expected), record["input_sha256"])
        self.assertNotEqual(
            audit_digest({"target_plan": plan, "drafts": expected["drafts"]}),
            record["input_sha256"])

    def test_judgement_assembly_hash_binds_reused_layer_outputs(self):
        target, material = self.fixture()

        class CapturingVerifier(StagedVerifier):
            def __init__(self, client):
                super().__init__(client)
                self.assembly_inputs = []

            def _assembly_audit(self, stage, material_id, context, inputs,
                                output=None, error_code=None):
                if stage == "judgement_assembly":
                    self.assembly_inputs.append(deepcopy(inputs))
                return super()._assembly_audit(
                    stage, material_id, context, inputs, output, error_code)

        client = ScriptClient(
            ("evidence", layer("contradicted", (("m", material.content),),
                               rationale="The scoped notice contradicts reopening.")),
            ("evidence_critic", LAYER_CRITIC_ACCEPT),
            ("world", layer()), ("world_critic", LAYER_CRITIC_ACCEPT))
        adapter = CapturingVerifier(client)
        adapter.verify(target, self.context(target, material))
        adapter.verify(target, self.context(target, material))
        reused = [item for item in adapter.history
                  if item["status"] == "reused_validated_layer"]
        self.assertEqual(2, len(reused))
        inputs = adapter.assembly_inputs[-1]
        self.assertEqual({"evidence", "world"},
                         set(inputs["accepted_or_reused_layers"]))
        record = [item for item in adapter.history
                  if item["stage"] == "judgement_assembly"][-1]
        self.assertEqual(audit_digest(inputs), record["input_sha256"])
        legacy = {"target_plan": inputs["target_plan"],
                  "layer_payload_sha256": inputs["layer_payload_sha256"]}
        self.assertNotEqual(audit_digest(legacy), record["input_sha256"])
        changed = deepcopy(inputs)
        changed["accepted_or_reused_layers"]["evidence"]["rationale"] += " changed"
        self.assertNotEqual(audit_digest(inputs), audit_digest(changed))

    def test_layer_critic_repairs_one_layer_and_python_assembles_verdict(self):
        target, material = self.fixture()
        fragment = asdict(p.Fragment(
            "f", material.content, p.Span("m", 0, len(material.content), material.content),
            target.id))
        client = ScriptClient(
            ("evidence", layer("conflicting", (("m", material.content),),
                               rationale="Initial draft.")),
            ("evidence_critic", {
                "decision": "repair",
                "issue": "The negation was interpreted backwards.",
                "basis": [{"version_id": "m", "quote": material.content}]}),
            ("evidence", layer("contradicted", (("m", material.content),),
                               rationale="Closed contradicts reopened.")),
            ("evidence_critic", LAYER_CRITIC_ACCEPT),
            ("world", layer()), ("world_critic", LAYER_CRITIC_ACCEPT))
        adapter = StagedVerifier(client, max_repairs=1)
        result = adapter.verify(target, self.context(target, material, fragments=(fragment,)))
        self.assertEqual("contradicted", result.evidence_verdict)
        self.assertEqual(result.evidence_verdict, result.verdict)
        self.assertEqual(["evidence", "evidence_critic", "evidence",
                          "evidence_critic", "world", "world_critic"],
                         [call["stage"] for call in client.calls])
        self.assertIsNotNone(client.calls[2]["payload"]["repair"])
        self.assertEqual(client.calls[0]["payload"].get("repair"), None)

    def test_invalid_stage_quote_fails_before_partial_analysis_commit(self):
        target, material = self.fixture()
        client = ScriptClient(
            ("atoms", ATOMS),
            ("lineage", {"citations": [{
                "locator": "https://missing.example/source", "quote": "not in material",
                "kind": "cites", "rationale": "Invalid fixture.",
                "decision_impact": "Would change provenance."}],
                "origin": None, "revisit": [], "notes": ""}))
        report = p.run_provenance(
            target, SnapshotSearchProvider([material], ["m"]),
            StagedDecomposer(client), config=p.TraceConfig(max_rounds=1))
        self.assertEqual("decomposer_error", report["stop_reason"])
        self.assertEqual({}, report["analyses"])
        self.assertEqual([], report["eligible_version_ids"])
        self.assertEqual(["atoms", "lineage"], [call["stage"] for call in client.calls])

    def test_failed_call_does_not_link_an_older_transport_record(self):
        target, material = self.fixture()

        class FailingClient:
            records = [{
                "sequence": 77, "request_digest": "old",
                "response_id": "old-response",
            }]

            def call(self, *args, **kwargs):
                raise RuntimeError("fixture transport failure")

        adapter = StagedDecomposer(FailingClient(), max_repairs=0)
        with self.assertRaises(StagedSemanticError):
            adapter.decompose(target, material, self.context(target, material))
        record = next(item for item in adapter.history if item["stage"] == "atoms")
        self.assertNotIn("model_call_sequence", record)
        self.assertNotIn("response_id", record)

    def test_missing_frozen_scope_becomes_exact_retrieval_task(self):
        target, material = self.fixture()
        target = p.Target(**{**asdict(target), "evidence_scope": ("m", "missing")})
        client = ScriptClient(
            ("evidence", {"probe_results": [{**layer()["probe_results"][0],
                                               "stop_reason": "scope_unavailable"}]}),
            ("evidence_critic", LAYER_CRITIC_ACCEPT),
            ("world", layer()), ("world_critic", LAYER_CRITIC_ACCEPT))
        result = StagedVerifier(client).verify(target, self.context(target, material))
        task, = [gap for gap in result.gaps if gap.locator == "missing"]
        self.assertEqual("fetch", task.action)
        self.assertEqual("evidence", task.dimension)
        self.assertFalse(task.blocking)

    def test_prompt_manifest_is_complete_and_hashes_each_distinct_contract(self):
        manifest = prompt_manifest()
        self.assertEqual(PROMPT_VERSION, manifest["version"])
        self.assertEqual({"atoms", "lineage", "decomposition_critic", "evidence",
                          "evidence_critic", "world", "world_critic"},
                         set(manifest["stages"]))
        hashes = {value["prompt_sha256"] for value in manifest["stages"].values()}
        self.assertEqual(7, len(hashes))
        self.assertEqual("deterministic-target-plan-v2",
                         manifest["target_plan_version"])

    def test_gap_return_reenters_every_material_stage_before_second_verification(self):
        first_text = "The bridge status is pending. Source: https://example.org/final"
        final_text = "The bridge remains closed until Friday."
        target, first = self.fixture(first_text, "first")
        target = p.Target(**{**asdict(target), "evidence_scope": ()})
        final = p.MaterialVersion(
            "final", "https://example.org/final", final_text,
            first.retrieved_at, first.published_at, first.available_at,
            first.availability_basis)
        calls = []

        class LoopClient:
            def call(self, system, user, json_schema=None, json_mode=False, **metadata):
                stage = metadata["stage"]
                payload = json.loads(user)
                calls.append((stage, payload))
                if stage == "atoms":
                    text = payload["material"]["content"]
                    return {"atoms": [{"statement": text, "quote": text,
                                       "qualifier_quotes": []}], "notes": ""}
                if stage == "lineage":
                    return NO_LINEAGE
                if stage == "decomposition_critic":
                    return DECOMPOSITION_ACCEPT
                if stage == "world":
                    return layer()
                if stage.endswith("_critic"):
                    return LAYER_CRITIC_ACCEPT
                registered = payload["registered_gaps"]
                if not registered:
                    return layer("unresolved", (("first", first_text),), gaps=({
                        "question": "Retrieve the final bridge record.",
                        "action": "fetch", "locator": final.url, "blocking": True,
                        "basis": [{"version_id": "first", "quote": first_text}],
                        "decision_impact": "The final record determines the bridge status."},))
                gap_id = next(gap["id"] for gap in registered
                              if gap.get("locator") == final.url)
                return layer("contradicted", (("final", final_text),), resolutions=({
                    "gap_id": gap_id,
                    "basis": [{"version_id": "final", "quote": final_text}],
                    "rationale": "The retrieved record answers the pending status."},),
                    rationale="The final record contradicts reopening.")

        report = p.run_provenance(
            target, SnapshotSearchProvider([first, final], ["first"]),
            StagedDecomposer(LoopClient()), StagedVerifier(LoopClient()),
            p.TraceConfig(max_rounds=2, max_documents=4, max_decomposition_calls=4))
        self.assertEqual([], report["errors"])
        self.assertEqual(2, report["usage"]["verification_calls"])
        self.assertEqual("contradicted", report["decision_status"])
        second_snapshot = next(item for item in report["operations"]
                               if item["round"] == 2 and item["action"] == "snapshot_saved")
        second_decompose = next(item for item in report["operations"]
                                if item["round"] == 2 and item["action"] == "decompose_completed")
        second_verify = next(item for item in report["operations"]
                             if item["round"] == 2 and item["action"] == "verification_started")
        self.assertLess(second_snapshot["sequence"], second_decompose["sequence"])
        self.assertLess(second_decompose["sequence"], second_verify["sequence"])
        # The attributed validation return deterministically reopens the old
        # basis-owning material as well as decomposing the new snapshot.
        self.assertEqual(3, sum(stage == "atoms" for stage, _ in calls))
        self.assertEqual(3, sum(stage == "lineage" for stage, _ in calls))
        feedback = [payload["verifier_feedback"] for stage, payload in calls
                    if stage == "decomposition_critic"
                    and payload["material"]["version_id"] == "final"]
        self.assertTrue(feedback)
        self.assertTrue(feedback[0]["tasks"])
        self.assertEqual("exact", feedback[0]["attribution"])

    def test_compound_target_cannot_be_supported_by_only_one_branch(self):
        target, material = self.fixture("The bridge reopened.")
        target = replace(target, text="The bridge reopened and the ferry operates.")
        plan = build_target_plan(target)
        self.assertEqual(2, len(plan["probes"]))
        quote = material.content
        results = []
        for probe in plan["probes"]:
            results.append({
                "probe_number": probe["number"],
                "basis_pool": [{"version_id": "m", "quote": quote}],
                "dimension_results": [{
                    "dimension": dimension, "verdict": "supported",
                    "basis_indices": [0], "rationale": "Claimed support."}
                    for dimension in probe["required_dimensions"]],
                "gaps": [], "resolutions": [], "stop_reason": "none",
            })
        client = ScriptClient(("evidence", {"probe_results": results}))
        with self.assertRaises(StagedSemanticError):
            StagedVerifier(client, max_repairs=0).verify(
                target, self.context(target, material))

    def test_qualifier_dimension_needs_colocated_core_grounding(self):
        content = "The bridge reopened. Friday was mentioned elsewhere."
        target, material = self.fixture(content)
        plan = build_target_plan(target)
        bridge_quote, time_quote = "The bridge reopened.", "Friday was mentioned elsewhere."
        dimensions = []
        for dimension in plan["probes"][0]["required_dimensions"]:
            dimensions.append({
                "dimension": dimension, "verdict": "supported",
                "basis_indices": [1 if dimension == "time" else 0],
                "rationale": "Claimed support.",
            })
        raw = {"probe_results": [{
            "probe_number": 1,
            "basis_pool": [{"version_id": "m", "quote": bridge_quote},
                           {"version_id": "m", "quote": time_quote}],
            "dimension_results": dimensions,
            "gaps": [], "resolutions": [], "stop_reason": "none",
        }]}
        with self.assertRaises(StagedSemanticError):
            StagedVerifier(ScriptClient(("evidence", raw)), max_repairs=0).verify(
                target, self.context(target, material))

    def test_deterministic_layer_repair_names_the_failed_dimension(self):
        content = "The bridge reopened before Friday. Friday was mentioned elsewhere."
        target, material = self.fixture(content)
        plan = build_target_plan(target)
        dimensions = []
        for dimension in plan["probes"][0]["required_dimensions"]:
            dimensions.append({
                "dimension": dimension, "verdict": "supported",
                "basis_indices": [1 if dimension == "time" else 0],
                "rationale": "Fixture support.",
            })
        invalid = {"probe_results": [{
            "probe_number": 1,
            "basis_pool": [
                {"version_id": "m", "quote": "The bridge reopened before Friday."},
                {"version_id": "m", "quote": "Friday was mentioned elsewhere."},
            ],
            "dimension_results": dimensions,
            "gaps": [], "resolutions": [], "stop_reason": "none",
        }]}
        repaired = layer(
            "supported", (("m", content),), rationale="Full quote grounds time.")
        client = ScriptClient(
            ("evidence", invalid), ("evidence", repaired),
            ("evidence_critic", LAYER_CRITIC_ACCEPT),
            ("world", layer()), ("world_critic", LAYER_CRITIC_ACCEPT))
        StagedVerifier(client, max_repairs=1).verify(
            target, self.context(target, material))
        repair = client.calls[1]["payload"]["repair"]
        self.assertEqual("dimension_grounding", repair["error_code"])
        self.assertEqual(1, repair["probe_number"])
        self.assertEqual("time", repair["dimension"])

    def test_unresolved_probe_cannot_silently_end_without_task_or_stop(self):
        target, material = self.fixture()
        raw = layer()
        raw["probe_results"][0]["stop_reason"] = "none"
        with self.assertRaises(StagedSemanticError):
            StagedVerifier(ScriptClient(("evidence", raw)), max_repairs=0).verify(
                target, self.context(target, material))

    def test_resolution_requires_current_exact_task_receipt(self):
        target, material = self.fixture()
        task = p.Gap(
            "g", "Recheck the bridge notice", "verification", "evidence",
            True, target.id,
            (p.Span("m", 0, len(material.content), material.content),),
            "Could change the evidence decision", "reanalyse", "m")
        raw = layer(resolutions=({
            "gap_id": "g",
            "basis": [{"version_id": "m", "quote": material.content}],
            "rationale": "Old basis reused."},))
        context = self.context(target, material, gaps=(asdict(task),))
        with self.assertRaises(StagedSemanticError):
            StagedVerifier(ScriptClient(("evidence", raw)), max_repairs=0).verify(
                target, context)

    def test_out_of_scope_world_material_does_not_resample_evidence_layer(self):
        target, material = self.fixture()
        other = replace(material, version_id="world", url="https://example.org/world",
                        content="A separate world report.")
        client = ScriptClient(
            ("evidence", layer("contradicted", (("m", material.content),))),
            ("evidence_critic", LAYER_CRITIC_ACCEPT),
            ("world", layer()), ("world_critic", LAYER_CRITIC_ACCEPT),
            ("world", layer()), ("world_critic", LAYER_CRITIC_ACCEPT))
        verifier = StagedVerifier(client)
        verifier.verify(target, self.context(target, material))
        second = self.context(target, material)
        second["materials"].append(asdict(other))
        second["usage"]["rounds"] = 2
        verifier.verify(target, second)
        stages = [call["stage"] for call in client.calls]
        self.assertEqual(1, stages.count("evidence"))
        self.assertEqual(2, stages.count("world"))
        self.assertTrue(any(item["status"] == "reused_validated_layer"
                            for item in verifier.history))
        transactions = {item.get("transaction_id") for item in verifier.history
                        if item["stage"] in {"evidence", "world", "judgement_assembly"}}
        self.assertEqual(2, len(transactions))
        self.assertTrue(all(transactions))
        assemblies = [item for item in verifier.history
                      if item["stage"] == "judgement_assembly"]
        self.assertEqual(2, len(assemblies))
        self.assertTrue(all(len(item["output_sha256"]) == 64
                            for item in assemblies))

    def test_cached_layer_never_replays_a_gap_after_its_resolution(self):
        target, material = self.fixture()

        class LifecycleClient:
            def __init__(self):
                self.evidence_calls = 0

            def call(self, system, user, json_schema=None, json_mode=False, **metadata):
                stage = metadata["stage"]
                payload = json.loads(user)
                if stage.endswith("_critic"):
                    return deepcopy(LAYER_CRITIC_ACCEPT)
                if stage == "world":
                    return layer()
                self.evidence_calls += 1
                registered = payload["registered_gaps"]
                if not registered and self.evidence_calls == 1:
                    return layer(gaps=({
                        "question": "Reanalyse the frozen bridge notice.",
                        "action": "reanalyse", "locator": "m",
                        "blocking": True,
                        "basis": [{"version_id": "m", "quote": material.content}],
                        "decision_impact": "The reanalysis can change entailment.",
                    },))
                if registered:
                    return layer(
                        "contradicted", (("m", material.content),),
                        resolutions=({
                            "gap_id": registered[0]["id"],
                            "basis": [{"version_id": "m",
                                       "quote": material.content}],
                            "rationale": "The attributed reanalysis closed the task.",
                        },), rationale="The notice contradicts reopening.")
                return layer(
                    "contradicted", (("m", material.content),),
                    rationale="The notice still contradicts reopening.")

        client = LifecycleClient()
        verifier = StagedVerifier(client)
        first = verifier.verify(target, self.context(target, material))
        gap, = first.gaps
        second_context = self.context(target, material, gaps=(asdict(gap),))
        second_context["current_round_returns"] = [{
            "version_id": "m", "task_ids": [gap.id],
            "tasks": [asdict(gap)], "attribution": "envelope",
        }]
        second_context["retrieval_feedback"] = [{
            "gap_id": gap.id, "action": gap.action, "locator": gap.locator,
            "status": "returned", "version_ids": ["m"],
        }]
        second = verifier.verify(target, second_context)
        self.assertEqual([gap.id], [item.gap_id for item in second.resolutions])
        third = verifier.verify(target, self.context(target, material))
        self.assertEqual((), third.gaps)
        self.assertEqual(3, client.evidence_calls)

    def test_program_closes_returned_missing_scope_task(self):
        target, material = self.fixture()
        target = replace(target, evidence_scope=("m", "missing"))
        missing = replace(
            material, version_id="missing", url="https://example.org/missing")
        first_evidence = layer(gaps=({
            "question": "Fetch the missing frozen version.",
            "action": "fetch", "locator": "missing", "blocking": False,
            "basis": [],
            "decision_impact": "The missing version can change entailment.",
        },))
        client = ScriptClient(
            ("evidence", first_evidence),
            ("evidence_critic", LAYER_CRITIC_ACCEPT),
            ("world", layer()), ("world_critic", LAYER_CRITIC_ACCEPT),
            ("evidence", layer(
                "contradicted", (("missing", missing.content),),
                rationale="The returned frozen version contradicts reopening.")),
            ("evidence_critic", LAYER_CRITIC_ACCEPT),
            ("world", layer()), ("world_critic", LAYER_CRITIC_ACCEPT))
        verifier = StagedVerifier(client)
        first = verifier.verify(target, self.context(target, material))
        gap, = [item for item in first.gaps if item.locator == "missing"]
        self.assertIsNone(gap.probe_id)
        second_context = self.context(target, material, gaps=(asdict(gap),))
        second_context["materials"].append(asdict(missing))
        second_context["current_round_returns"] = [{
            "version_id": "missing", "task_ids": [gap.id],
            "tasks": [asdict(gap)], "attribution": "envelope",
        }]
        second_context["retrieval_feedback"] = [{
            "gap_id": gap.id, "action": gap.action, "locator": gap.locator,
            "status": "returned", "version_ids": ["missing"],
        }]
        second = verifier.verify(target, second_context)
        self.assertIn(gap.id, {item.gap_id for item in second.resolutions})

    def test_probe_specific_unavailable_feedback_cannot_stop_another_probe(self):
        target, material = self.fixture("Alice was mentioned.")
        target = replace(target, text="Alice won and Bob lost.",
                         assessment_mode="world", evidence_scope=())
        plan = build_target_plan(target)
        first_probe, second_probe = plan["probes"]
        basis = p.Span("m", 0, len(material.content), material.content)
        gap = p.Gap(
            "probe-one-task", "Find Alice's result", "verification", "world",
            True, target.id, (basis,), "Could change Alice's result", "search",
            "Alice result", first_probe["id"])
        raw = {"probe_results": [{
            "probe_number": probe["number"], "basis_pool": [],
            "dimension_results": [{
                "dimension": dimension, "verdict": "unresolved",
                "basis_indices": [], "rationale": "No conclusion.",
            } for dimension in probe["required_dimensions"]],
            "gaps": [], "resolutions": [], "stop_reason": "no_source_lead",
        } for probe in (first_probe, second_probe)]}
        with self.assertRaises(ValueError):
            StagedVerifier(ScriptClient())._assemble_layer(
                target, "world", raw, {"m": asdict(material)},
                {gap.id: asdict(gap)}, [], plan, [], [{
                    "gap_id": gap.id, "action": gap.action,
                    "locator": gap.locator, "status": "unavailable",
                    "version_ids": [],
                }])

    def test_deterministic_atom_gate_gets_one_bounded_repair(self):
        target, material = self.fixture()
        invalid = {"atoms": [{"statement": "Bad", "quote": "missing",
                               "qualifier_quotes": []}], "notes": ""}
        client = ScriptClient(
            ("atoms", invalid), ("atoms", ATOMS), ("lineage", NO_LINEAGE),
            ("decomposition_critic", DECOMPOSITION_ACCEPT))
        result_adapter = StagedDecomposer(client, max_repairs=1)
        result = result_adapter.decompose(
            target, material, self.context(target, material))
        self.assertTrue(result.fragments)
        repair = client.calls[1]["payload"]["repair"]
        self.assertEqual(invalid, repair["previous_draft"])
        stage_records = [item for item in result_adapter.history
                         if item["stage"] == "atoms"]
        self.assertEqual(stage_records[0]["sequence"],
                         stage_records[1]["repair_of"])

    def test_duplicate_lineage_lead_is_merged_and_same_url_versions_stay_ambiguous(self):
        text = "Cites https://example.org/upstream."
        target, source = self.fixture(text, "source")
        first = replace(source, version_id="up-1", url="https://example.org/upstream",
                        content="Upstream version one.")
        second = replace(source, version_id="up-2", url="https://example.org/upstream",
                         content="Upstream version two.")
        atoms = {"atoms": [{"statement": text, "quote": text,
                             "qualifier_quotes": []}], "notes": ""}
        lineage = {"citations": [{
            "locator": first.url, "quote": text, "kind": kind,
            "rationale": "Visible ambiguous citation.",
            "decision_impact": "Version identity changes lineage."}
            for kind in ("quotes", "cites")],
            "origin": None, "revisit": [], "notes": ""}
        client = ScriptClient(
            ("atoms", atoms), ("lineage", lineage),
            ("decomposition_critic", DECOMPOSITION_ACCEPT))
        context = self.context(target, source)
        context["materials"].extend([asdict(first), asdict(second)])
        result = StagedDecomposer(client).decompose(target, source, context)
        self.assertEqual(2, len(result.relations))
        self.assertTrue(all(item.status == "declared" for item in result.relations))
        self.assertEqual(1, len(result.gaps))

    def test_target_plan_does_not_split_common_abbreviations(self):
        target, _ = self.fixture()
        target = replace(target, text="Dr. Smith said U.S. GDP grew 3.2%.")
        plan = build_target_plan(target)
        self.assertEqual(1, len(plan["probes"]))
        self.assertIn("U.S. GDP", plan["probes"][0]["text"])

    def test_target_plan_splits_common_irregular_predicate_conjunctions(self):
        target, _ = self.fixture()
        for text in ("Alice won and Bob lost.",
                     "Alice left and Bob stayed.",
                     "Alice said yes and Bob denied it."):
            with self.subTest(text=text):
                plan = build_target_plan(replace(target, text=text))
                self.assertEqual(2, len(plan["probes"]))

    def test_target_plan_keeps_shared_attribution_scope_intact(self):
        target, _ = self.fixture()
        text = "The agency alleged that Alice failed and Bob cheated."
        plan = build_target_plan(replace(target, text=text))
        self.assertEqual(1, len(plan["probes"]))
        self.assertEqual(text[:-1], plan["probes"][0]["text"])

    def test_ambiguous_unrecognized_compound_target_fails_closed(self):
        target, _ = self.fixture()
        for text in ("Alice fled and Bob quit.",
                     "The plane flew and the ship sank.",
                     "Alice sang and Bob danced."):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    build_target_plan(replace(target, text=text))

    def test_target_plan_detects_calendar_time_without_calling_it_quantity(self):
        target, _ = self.fixture()
        for text in ("The event occurred Friday.",
                     "The event occurred on September 5, 2026.",
                     "The event occurred at 3 PM.",
                     "事件于周五发生。"):
            with self.subTest(text=text):
                plan = build_target_plan(replace(target, text=text))
                dimensions = plan["probes"][0]["required_dimensions"]
                self.assertIn("time", dimensions)
                self.assertNotIn("quantity_unit_denominator", dimensions)

    def test_shared_subject_cannot_ground_a_different_supported_branch(self):
        content = "The bridge reopened before Friday."
        target, material = self.fixture(content)
        target = replace(
            target,
            text="The bridge reopened before Friday and the bridge closed after Monday.")
        plan = build_target_plan(target)
        self.assertEqual(2, len(plan["probes"]))
        raw = {"probe_results": [{
            "probe_number": probe["number"],
            "basis_pool": [{"version_id": "m", "quote": content}],
            "dimension_results": [{
                "dimension": dimension, "verdict": "supported",
                "basis_indices": [0], "rationale": "Claimed support.",
            } for dimension in probe["required_dimensions"]],
            "gaps": [], "resolutions": [], "stop_reason": "none",
        } for probe in plan["probes"]]}
        with self.assertRaises(ValueError):
            StagedVerifier(ScriptClient())._assemble_layer(
                target, "evidence", raw, {"m": asdict(material)}, {}, [],
                plan, [], [])

    def test_supported_core_dimensions_require_all_target_content_terms(self):
        cases = (
            ("Officials denied the report.", "Officials confirmed the report."),
            ("Alice bought Widget.", "Bob bought Widget."),
            ("Alice bought Widget.", "Alice bought Gadget."),
            ("Alice will buy Widget.", "Alice sold Widget."),
        )
        for probe, basis in cases:
            for dimension in ("actor_subject", "predicate_object", "scope_location"):
                with self.subTest(probe=probe, basis=basis, dimension=dimension):
                    self.assertFalse(dimension_evidence_is_grounded(
                        dimension, probe, basis, "supported"))
        self.assertTrue(dimension_evidence_is_grounded(
            "predicate_object", "Revenue increased.",
            "Revenue will increase.", "supported"))

    def test_supported_core_dimensions_preserve_role_and_scope_order(self):
        for probe, basis in (
                ("Alice defeated Bob.", "Bob defeated Alice."),
                ("The agency alleged that Alice failed and Bob cheated.",
                 "The agency alleged that Alice failed. Bob cheated."),
                ("苹果公司在北京收购微软。", "苹果公司在上海发布新品。")):
            for dimension in ("actor_subject", "predicate_object",
                              "scope_location"):
                with self.subTest(probe=probe, basis=basis,
                                  dimension=dimension):
                    self.assertFalse(dimension_evidence_is_grounded(
                        dimension, probe, basis, "supported"))

    def test_quantity_support_requires_matching_unit_and_denominator(self):
        self.assertFalse(dimension_evidence_is_grounded(
            "quantity_unit_denominator", "Revenue was $10.",
            "Revenue was 10 percent.", "supported"))
        self.assertFalse(dimension_evidence_is_grounded(
            "quantity_unit_denominator", "There were 10 votes per member.",
            "There were 10 votes per district.", "supported"))
        self.assertTrue(dimension_evidence_is_grounded(
            "quantity_unit_denominator", "There were 10 votes per member.",
            "There were 10 votes per member.", "supported"))

    def test_contradicted_dimension_dominates_conflicting_dimension_for_all(self):
        target, material = self.fixture()
        plan = build_target_plan(target)
        quote = material.content
        checks = []
        for index, dimension in enumerate(plan["probes"][0]["required_dimensions"]):
            verdict = "conflicting" if index == 1 else "contradicted"
            checks.append({
                "dimension": dimension, "verdict": verdict,
                "basis_indices": [0], "rationale": "Fixture mixed result.",
            })
        raw = {"probe_results": [{
            "probe_number": 1,
            "basis_pool": [{"version_id": "m", "quote": quote}],
            "dimension_results": checks, "gaps": [], "resolutions": [],
            "stop_reason": "none",
        }]}
        result = StagedVerifier(ScriptClient())._assemble_layer(
            target, "evidence", raw, {"m": asdict(material)}, {}, [],
            plan, [], [])
        self.assertEqual("contradicted", result["verdict"])

    def test_contradicted_probe_dominates_conflicting_probe_for_all(self):
        target, material = self.fixture(
            "The bridge remains closed and the ferry does not operate.")
        target = replace(
            target, text="The bridge reopened and the ferry operates.")
        plan = build_target_plan(target)
        self.assertEqual(2, len(plan["probes"]))
        raw_results = []
        for probe, verdict in zip(plan["probes"], ("conflicting", "contradicted")):
            raw_results.append({
                "probe_number": probe["number"],
                "basis_pool": [{"version_id": "m", "quote": material.content}],
                "dimension_results": [{
                    "dimension": dimension, "verdict": verdict,
                    "basis_indices": [0], "rationale": "Fixture mixed result.",
                } for dimension in probe["required_dimensions"]],
                "gaps": [], "resolutions": [], "stop_reason": "none",
            })
        result = StagedVerifier(ScriptClient())._assemble_layer(
            target, "evidence", {"probe_results": raw_results},
            {"m": asdict(material)}, {}, [], plan, [], [])
        self.assertEqual("contradicted", result["verdict"])


if __name__ == "__main__":
    unittest.main()
