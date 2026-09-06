"""Mock transport integration of the actual staged runner, schemas and engine.

These fixtures test execution and accounting, never accuracy or a live API.
"""
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
runner = importlib.import_module("staged_run")
stages = importlib.import_module("staged_semantic")
extension = importlib.import_module("extended_semantic")


def accepted_extension(payload):
    """Return a complete v4 extension using only the supplied frozen contract."""
    probes = []
    for claim in payload["claim_contract"]["claims"]:
        dimensions = {}
        for item in claim["dimensions"]:
            dimensions.setdefault(item["kind"], []).append(item["id"])
        bindings = [("predicate_core", dimensions["predicate"]),
                    ("source_lineage", [])]
        for dimension, kind in (("negation", "polarity"),
                                ("quantity_unit", "quantity_unit"),
                                ("baseline_scope", "baseline_scope")):
            bindings.extend((kind, [identifier])
                            for identifier in dimensions.get(dimension, []))
        bindings.extend(("time_boundary", [identifier])
                        for identifier in dimensions.get("time", []))
        bindings.extend(("location", [identifier])
                        for identifier in dimensions.get("location", []))
        for dimension in ("condition", "modality"):
            bindings.extend(("condition_modality", [identifier])
                            for identifier in dimensions.get(dimension, []))
        bindings.extend(("actor_role", [identifier])
                        for identifier in dimensions.get("actor_role", []))
        bindings.extend(("entity_identity", [identifier])
                        for identifier in dimensions.get("entity_identity", []))
        bindings.extend(("exact_designation", [identifier])
                        for identifier in dimensions.get("exact_designation", []))
        all_ids = [item["id"] for item in claim["dimensions"]]
        logic = payload["claim_contract"]["logic"]
        if claim["role"] == "attributed_content":
            bindings.append(("attribution_relation", []))
        elif dimensions.get("exact_designation"):
            bindings.append(("designation_relation", all_ids))
        elif logic in {"conditional", "comparison", "causal"}:
            bindings.append((logic + "_relation", all_ids))
        else:
            bindings.append(("claim_composition", all_ids))
        if payload["target"]["assessment_mode"] == "world":
            bindings.append(("source_independence", []))
        probes.extend({"claim_id": claim["id"], "kind": kind,
                       "dimension_ids": dimension_ids,
                       "question": "Raw model wording is ignored.",
                       "decision_impact": "Raw model wording is ignored."}
                      for kind, dimension_ids in bindings)

    compatibility = {
        "number": {"quantity_unit", "time", "baseline_scope"},
        "date": {"time", "baseline_scope"},
        "negation": {"negation"},
        "modality": {"modality"},
        "condition": {"condition"},
        "designation": {"predicate", "exact_designation"},
        "attribution": {"predicate"},
        "location": {"location"},
    }
    claims = {item["id"]: item for item in payload["claim_contract"]["claims"]}
    ledger = []
    for segment in payload["coverage_segments"]:
        dimensions = claims[segment["claim_id"]]["dimensions"]
        overlapping = [item for item in dimensions
                       if max(segment["anchor"]["start"], item["anchor"]["start"]) <
                       min(segment["anchor"]["end"], item["anchor"]["end"])]
        allowed = compatibility.get(segment["cue_kind"])
        compatible = [item for item in overlapping
                      if allowed is None or item["kind"] in allowed]
        if segment["high_signal"]:
            status = "covered_by_dimension" if compatible else "suspected_missing"
            bound = compatible
        elif segment["cue_kind"] == "logic_connector":
            status, bound = "logic_connector", []
        elif overlapping:
            status, bound = "covered_by_dimension", overlapping
        else:
            status, bound = "context_only", []
        ledger.append({"segment_id": segment["id"], "status": status,
                       "dimension_ids": [item["id"] for item in bound]})
    return {"decision": "accept", "repair_quote": "", "repair_issue": "",
            "probes": probes, "coverage_ledger": ledger, "notes": ""}


class StagedRunMockTests(unittest.TestCase):
    @staticmethod
    def provider_fixture():
        target = runner.p.Target(
            "target", "The bridge was open.", "2026-09-04T20:00:00Z",
            source_version_id="b", assessment_mode="evidence",
            evidence_scope=("b",),
        )
        materials = (
            runner.p.MaterialVersion(
                "a", "https://example.org/a", "Original permit ledger.",
                "2026-09-04T20:00:00Z", "2026-09-03T00:00:00Z",
                "2026-09-03T00:00:00Z", "City Registry",
            ),
            runner.p.MaterialVersion(
                "b", "https://example.org/b", "The harbor bridge remains closed.",
                "2026-09-04T20:00:00Z", "2026-09-03T00:00:00Z",
                "2026-09-03T00:00:00Z", "Harbor Bulletin",
            ),
        )
        return target, materials

    def run_mock(self, *, include_failure=False, constructor_failure=False,
                 model_mismatch=False, target_extension=False,
                 provider_mode="fixed_reanalysis", routed_scenario="new_hit"):
        calls = []
        call_lock = threading.Lock()
        evidence_rounds = Counter()
        prompt_stages = [(getattr(stages, name.upper() + "_PROMPT"), name)
                         for name in ("atoms", "lineage", "critic", "evidence", "world")]
        prompt_stages += [(stages.JUDGMENT_CRITIC_PROMPT, "judgement_critic"),
                          (extension.CLAIM_CONTRACT_PROMPT, "claim_contract"),
                          (extension.EXTENSION_PROMPT, "extension")]
        claim_quote = "The bridge remains closed until 8 September 2026."
        origin_quote = "This is the original bridge closure record."
        citation_quote = "Source: https://example.org/a."
        active_citation_quote = ("Source: https://example.org/missing."
                                 if provider_mode == "task_routed" and
                                 routed_scenario == "no_hit" else citation_quote)

        def transport(**kwargs):
            # An unknown prompt fails immediately, including any LLM extractor.
            prompt = kwargs["messages"][0]["content"]
            stage = next(name for base, name in prompt_stages if prompt.startswith(base))
            payload = json.loads(kwargs["messages"][1]["content"])
            case_id = payload["target"]["id"]
            with call_lock:
                sequence = len(calls) + 1
                calls.append({"stage": stage, "case": case_id, "payload": payload,
                              "request": deepcopy(kwargs)})
                if stage == "evidence":
                    evidence_rounds[case_id] += 1
                evidence_round = evidence_rounds[case_id]
            if case_id == "failed-case" and stage == "evidence" and evidence_round == 2:
                raise RuntimeError("MOCK_SECRET transport detail must not be persisted")
            if stage == "claim_contract":
                text = payload["target"]["text"]
                answer = {"claims": [{"statement": text, "quote": text, "role": "main",
                    "dimensions": [{"kind": "subject", "quote": "bridge"},
                        {"kind": "predicate", "quote": "was open"},
                        {"kind": "time", "quote": "4 September 2026"}]}],
                    "logic": "single", "notes": ""}
            elif stage == "extension":
                answer = accepted_extension(payload)
            elif stage == "atoms":
                answer = {"atoms": [{"statement": claim_quote, "quote": claim_quote,
                                     "qualifier_quotes": ["until 8 September 2026"]}], "notes": ""}
                if payload.get("target_plan", {}).get("schema_version") == "decision-probe-v4":
                    answer["probe_checks"] = [{"probe_id": probe["id"],
                        "status": "addressed", "finding_indexes": [0],
                        "origin_used": False, "rationale": "The atom is relevant to this probe."}
                        for probe in payload["target_plan"]["probes"]]
            elif stage == "lineage":
                owner = payload["material"]["version_id"]
                answer = {"citations": [], "origin": None, "notes": ""}
                if owner == "a" or (owner == "b" and provider_mode == "task_routed" and
                                    routed_scenario == "no_gap"):
                    answer["origin"] = {"kind": "original_record", "quote": origin_quote,
                                        "rationale": "The source explicitly identifies its producing role."}
                elif payload["repair"] or payload["previous_analysis"]:
                    citation_locator = ("https://example.org/missing"
                                        if routed_scenario == "no_hit" else
                                        "https://example.org/a")
                    answer["citations"] = [{"locator": citation_locator,
                        "quote": active_citation_quote,
                        "kind": "cites", "rationale": "The source cites the original record.",
                        "decision_impact": "The original record establishes the cited provenance path."}]
                if payload.get("target_plan", {}).get("schema_version") == "decision-probe-v4":
                    has_citation = bool(answer["citations"])
                    has_origin = answer["origin"] is not None
                    answer["probe_checks"] = [{"probe_id": probe["id"],
                        "status": "addressed" if has_citation or has_origin else "absent",
                        "finding_indexes": [0] if has_citation else [],
                        "origin_used": has_origin,
                        "rationale": ("The material contains a lineage finding."
                                      if has_citation or has_origin else
                                      "The material contains no lineage finding.")}
                        for probe in payload["target_plan"]["probes"]]
            elif stage == "critic":
                missing = (payload["material"]["version_id"] == "b" and
                           not payload["drafts"]["lineage"]["citations"] and
                           not (provider_mode == "task_routed" and
                                routed_scenario == "no_gap"))
                answer = ({"decision": "repair", "stage": "lineage",
                           "quote": active_citation_quote,
                           "issue": "The explicit source citation is missing from the lineage draft."}
                          if missing else {"decision": "accept", "stage": "none", "quote": "", "issue": ""})
            elif stage in ("evidence", "world"):
                basis = [{"version_id": "b", "quote": claim_quote}]
                if target_extension:
                    v4 = (payload["target_plan"].get("schema_version") ==
                          "decision-probe-v4")
                    default_status = ("unresolved" if stage == "world"
                                      else "contradicted" if routed_scenario == "no_gap"
                                      else "supported" if evidence_round == 1
                                      else "contradicted")
                    probes = payload["target_plan"]["probes"]
                    answer = {"basis": basis, "rationale":
                        ("Recheck the stated date condition before deciding entailment."
                         if stage == "world" or evidence_round == 1 else
                         "The source states the bridge is closed on the target date."),
                        "probe_results": [{"probe_id": probe["id"],
                            "status": ("unresolved" if stage == "evidence" and
                                       routed_scenario != "no_gap" and
                                       evidence_round == 1 and index == 0
                                       else default_status),
                            "basis_indexes": [0], "rationale": "The passage answers this check.",
                            "referent_relation": "not_applicable"}
                            for index, probe in enumerate(probes)],
                        "gaps": [], "resolutions": []}
                    if v4:
                        answer["no_leads"] = []
                    if (stage == "evidence" and evidence_round == 1 and
                            routed_scenario != "no_gap"):
                        action = "reanalyse"
                        locator = "b"
                        gap_basis = basis
                        if provider_mode == "task_routed":
                            action = "fetch"
                            locator = ("https://example.org/a" if routed_scenario == "new_hit"
                                       else "https://example.org/missing")
                            # A v4 fetch must be derived from a visible source lead.
                            # Keep the mock response contract-valid so a structural
                            # repair call cannot be mistaken for a verification cycle.
                            declared_locator = (locator if routed_scenario == "new_hit"
                                                else "https://example.org/missing")
                            gap_basis = [{"version_id": "b",
                                          "quote": "Source: " + declared_locator + "."}]
                        answer["gaps"] = [{"probe_id": probes[0]["id"],
                            "question": probes[0]["question"],
                            "action": action, "locator": locator, "basis": gap_basis,
                            "decision_impact": probes[0]["decision_impact"]}]
                    elif stage == "evidence":
                        answer["resolutions"] = [{"gap_id": gap["id"], "basis": basis,
                            "rationale": "The quoted closure date includes the target date."}
                            for gap in payload["registered_gaps"]]
                    elif v4:
                        answer["no_leads"] = [{"probe_id": probe["id"],
                            "reason": "no_source_lead",
                            "rationale": "The frozen corpus exposes no new lead for this probe."}
                            for probe in probes]
                else:
                    answer = {"verdict": "unresolved", "basis": basis,
                              "rationale": "This fixture does not authenticate real world events.",
                              "gaps": [], "resolutions": []}
                    if stage == "evidence" and evidence_round == 1:
                        answer["rationale"] = "Recheck the stated date condition before deciding entailment."
                        answer["gaps"] = [{"question": "Does the closure date include 4 September?",
                            "action": "reanalyse", "locator": "b", "blocking": True, "basis": basis,
                            "decision_impact": "The date condition determines whether the target is contradicted."}]
                    elif stage == "evidence":
                        answer["verdict"] = "contradicted"
                        answer["rationale"] = "The source states the bridge is closed on the target date."
                        answer["resolutions"] = [{"gap_id": gap["id"], "basis": basis,
                            "rationale": "The quoted closure date includes the target date."}
                            for gap in payload["registered_gaps"]]
            else:
                answer = {"decision": "accept", "stage": "none", "probe_id": "",
                          "issue": "", "basis": []}
            return SimpleNamespace(id="mock-staged-response-" + str(sequence),
                model="unexpected-mock-model" if model_mismatch else kwargs["model"],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
                choices=[SimpleNamespace(finish_reason="stop",
                    message=SimpleNamespace(content=json.dumps(answer), refusal=None))])

        real_client = runner.BudgetClient

        class MockTransportClient(real_client):
            def __init__(self, model, budget, reasoning_effort=None):
                if constructor_failure:
                    raise RuntimeError("MOCK_SECRET constructor detail must not be persisted")
                super().__init__(model, budget, transport=transport, reasoning_effort=reasoning_effort)

        material_text = {
            "a": "City register: " + origin_quote + " " + claim_quote,
            "b": ((origin_quote + " " if provider_mode == "task_routed" and
                   routed_scenario == "no_gap" else "") +
                  claim_quote + " " +
                  (active_citation_quote
                   if provider_mode == "task_routed" and routed_scenario == "no_hit"
                   else citation_quote)),
        }
        if provider_mode == "task_routed" and routed_scenario == "provenance_only":
            material_text["b"] += " Source: https://example.org/missing."
        if provider_mode == "task_routed" and routed_scenario == "no_hit":
            material_text.pop("a")
        materials = [asdict(runner.p.MaterialVersion(owner, "https://example.org/" + owner, content,
            "2026-09-05T00:00:00Z", "2026-09-03T00:00:00Z", "2026-09-03T00:00:00Z",
            "Synthetic exact-version archive.")) for owner, content in material_text.items()]
        identifiers = ["success-case"] + (["failed-case"] if include_failure else [])
        data = {"schema_version": 1, "cases": [{
            "target": asdict(runner.p.Target(identifier, "The bridge was open on 4 September 2026.",
                "2026-09-04T20:00:00Z", source_version_id="b", assessment_mode="evidence",
                evidence_scope=("b",))), "materials": materials,
            "seed_ids": list(material_text),
            "full_evidence_control": False} for identifier in identifiers]}
        with tempfile.TemporaryDirectory(prefix="staged-mock-") as folder:
            temp = Path(folder)
            inputs = temp / "inputs.json"
            inputs.write_text(json.dumps(data))
            output = temp / "run"
            args = SimpleNamespace(inputs=str(inputs), output=str(output), cases=None,
                model="mock-model", reasoning_effort="medium", per_call_tokens=4000,
                output_tokens=64000, max_calls=64, seconds=900, rounds=2,
                max_repairs=1, workers=2, target_extension=target_extension,
                provider_mode=provider_mode)
            stdout = io.StringIO()
            # The dummy environment value only passes preflight. Every client
            # is constructed above with an explicit transport; no API is used.
            with patch.object(runner, "BudgetClient", MockTransportClient), \
                    patch.dict(runner.os.environ, {"OPENAI_API_KEY": "mock-only-no-api-access"}), \
                    redirect_stdout(stdout):
                status = runner.run(args)
            files = {path.name: json.loads(path.read_text()) for path in output.glob("*.json")}
            checkpoint_hashes = {path.name: hashlib.sha256(path.read_text().rstrip("\n").encode()).hexdigest()
                                 for path in output.glob("*-checkpoint-*.json")}
            copied_hashes = {path.relative_to(output / "executed-code").as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in (output / "executed-code").rglob("*.py")}
            original_input_hash = hashlib.sha256(inputs.read_bytes()).hexdigest()
            # Ensure neither arbitrary exception text nor the dummy key is
            # copied into any retained report, request log, or progress output.
            serialized = json.dumps(files) + stdout.getvalue()
            self.assertNotIn("MOCK_SECRET", serialized)
            self.assertNotIn("mock-only-no-api-access", serialized)
        return status, files, calls, checkpoint_hashes, copied_hashes, original_input_hash

    def test_two_materials_two_rounds_repair_native_verdict_and_frozen_evidence(self):
        status, files, calls, checkpoint_hashes, copied_hashes, input_hash = self.run_mock()
        self.assertEqual(0, status)
        row, = files["results.json"]
        self.assertEqual("completed", row["status"])
        self.assertEqual("false", row["prediction"])
        self.assertEqual(row["prediction"], row["native_prediction"]["decision"])
        self.assertEqual("unresolved", row["native_prediction"]["assessments"]["world"]["decision"])
        self.assertEqual(["unverifiable", "false"], [point["decision"] for point in row["checkpoints"]])
        self.assertEqual({"rounds": 2, "documents": 4, "unique_versions": 2,
            "decomposition_calls": 4, "verification_calls": 2}, row["engine_usage"])
        expected = ["atoms", "lineage", "critic", "atoms", "lineage", "critic", "lineage", "critic",
                    "evidence", "world", "atoms", "lineage", "critic", "atoms", "lineage", "critic",
                    "evidence", "world"]
        self.assertEqual(expected, [call["stage"] for call in calls])
        self.assertEqual(18, row["new_api_attempts"])
        self.assertEqual(18, row["returned_responses"])
        self.assertEqual(18, row["usage"]["model_calls"])
        self.assertEqual(180, row["usage"]["input_tokens"])
        self.assertEqual(360, row["usage"]["output_tokens"])
        self.assertTrue(all(call["request"]["max_completion_tokens"] == 4000 for call in calls))
        self.assertTrue(all(call["request"]["reasoning_effort"] == "medium" for call in calls))
        self.assertEqual(files["config.json"]["source_sha256"], copied_hashes)
        self.assertEqual(files["config.json"]["input_sha256"], input_hash)
        self.assertFalse(files["config.json"]["gold_read_during_inference"])
        self.assertEqual(0, files["config.json"]["automatic_transport_retries"])
        for point in row["checkpoint_hashes"]:
            self.assertEqual(point["sha256"], checkpoint_hashes[f"success-case-checkpoint-{point['round']}.json"])
        report = files["success-case-report.json"]
        self.assertEqual([], report["errors"])
        self.assertEqual("contradicted", report["verification_history"][-1]["evidence_verdict"])
        self.assertEqual("original_material_located", report["provenance_status"])
        self.assertEqual([], report["gaps"])
        self.assertEqual(4, len(report["analysis_history"]))
        self.assertEqual(report["analysis_history"][0]["analysis"]["fragments"],
                         report["analysis_history"][2]["analysis"]["fragments"])
        psi_history = files["success-case-psi-history.json"]
        self.assertEqual(1, sum(item["status"] == "repair_requested" for item in psi_history))
        self.assertEqual([1, 2], [point["round"] for point in files["success-case-retrieval.json"]])
        second_round_critic = calls[15]["payload"]
        self.assertIsNotNone(second_round_critic["previous_analysis"])
        self.assertEqual("b", second_round_critic["verifier_feedback"]["tasks"][0]["locator"])
        self.assertTrue(all({m["version_id"] for m in call["payload"]["materials"]} == {"b"}
                            for call in calls if call["stage"] == "evidence"))

    def test_failed_call_stays_in_matrix_with_prior_checkpoint_and_usage(self):
        status, files, calls, _, _, _ = self.run_mock(include_failure=True)
        self.assertEqual(1, status)
        rows = {row["id"]: row for row in files["results.json"]}
        self.assertEqual({"success-case", "failed-case"}, set(rows))
        self.assertEqual("completed", rows["success-case"]["status"])
        failed = rows["failed-case"]
        self.assertEqual("error", failed["status"])
        self.assertIsNone(failed["prediction"])
        self.assertEqual("StagedSemanticError", failed["error_type"])
        self.assertIn(failed["error_stage"], {"evidence", "world"})
        self.assertEqual("sdk_error", failed["error_code"])
        self.assertEqual(17, failed["new_api_attempts"])
        self.assertEqual(16, failed["returned_responses"])
        self.assertEqual(17, failed["usage"]["model_calls"])
        self.assertEqual(160, failed["usage"]["input_tokens"])
        self.assertEqual(320, failed["usage"]["output_tokens"])
        self.assertEqual([1], [point["round"] for point in failed["checkpoint_hashes"]])
        self.assertEqual([1], [point["round"] for point in failed["checkpoints"]])
        self.assertEqual("sdk_error", files["failed-case-calls.json"][-1]["error_code"])
        self.assertEqual("RuntimeError", files["failed-case-calls.json"][-1]["error_type"])
        self.assertEqual("failed", files["failed-case-verification-history.json"][-1]["status"])
        self.assertEqual("verifier", files["failed-case-report.json"]["errors"][-1]["stage"])
        self.assertIn(files["failed-case-report.json"]["errors"][-1]["semantic_stage"],
                      {"evidence", "world"})
        self.assertEqual("has_errors", files["status.json"]["status"])
        self.assertEqual(2, files["status.json"]["scheduled_cases"])
        self.assertEqual(1, files["status.json"]["completed"])
        self.assertEqual(len(calls), files["status.json"]["new_api_attempts"])
        self.assertEqual(35, files["status.json"]["new_api_attempts"])
        self.assertEqual(34, files["status.json"]["returned_responses"])

    def test_client_constructor_failure_retains_every_scheduled_case(self):
        status, files, calls, _, _, _ = self.run_mock(include_failure=True, constructor_failure=True)
        self.assertEqual(1, status)
        self.assertEqual([], calls)
        self.assertEqual(2, len(files["results.json"]))
        self.assertTrue(all(row["status"] == "error" and row["prediction"] is None
                            and row["new_api_attempts"] == 0 for row in files["results.json"]))
        self.assertEqual(2, files["status.json"]["scheduled_cases"])
        self.assertEqual(0, files["status.json"]["completed"])

    def test_unexpected_returned_model_retains_response_and_invalidates_case(self):
        status, files, calls, _, _, _ = self.run_mock(model_mismatch=True)
        self.assertEqual(1, status)
        row, = files["results.json"]
        self.assertEqual("error", row["status"])
        self.assertIsNone(row["prediction"])
        self.assertEqual(1, row["new_api_attempts"])
        self.assertEqual(1, row["returned_responses"])
        self.assertEqual(1, len(calls))
        self.assertEqual("unexpected_model", files["success-case-calls.json"][0]["error_code"])

    def test_task_routed_first_round_returns_only_target_source_with_origin_attribution(self):
        target, materials = self.provider_fixture()
        provider = runner.TaskRoutedDevelopmentProvider(materials)
        origin = runner.p.Gap("origin:target", "Find the producing record.",
                              target_id=target.id)
        unrelated = runner.p.Gap("other", "Find an unrelated record.",
                                 target_id=target.id)

        hits = list(provider.search(target, (unrelated, origin), 1, 8))

        self.assertEqual(["b"], [hit.material.version_id for hit in hits])
        self.assertEqual([("origin:target",)], [hit.task_ids for hit in hits])
        self.assertTrue(all(isinstance(hit, runner.p.RetrievalHit) for hit in hits))
        self.assertEqual(["other"],
                         [item["task_id"] for item in provider.last_feedback])
        self.assertEqual("corpus_exhausted", provider.last_feedback[0]["status"])
        self.assertEqual([{"version_id": "b", "task_ids": ["origin:target"]}],
                         provider.history[0]["attribution"])

    def test_task_routed_exact_and_lexical_routes_never_fall_back_to_full_corpus(self):
        target, materials = self.provider_fixture()
        provider = runner.TaskRoutedDevelopmentProvider(materials)
        fetch = runner.p.Gap("fetch-a", "Fetch the cited record.",
            target_id=target.id, action="fetch", locator="https://example.org/a")
        reanalyse_a = runner.p.Gap("reanalyse-a", "Reanalyse the cited snapshot.",
            target_id=target.id, action="reanalyse", locator="a")
        reanalyse = runner.p.Gap("reanalyse-b", "Reanalyse the target snapshot.",
            target_id=target.id, action="reanalyse", locator="b")

        exact_hits = list(provider.search(
            target, (fetch, reanalyse_a, reanalyse), 2, 8))
        exact = {hit.material.version_id: hit.task_ids for hit in exact_hits}
        self.assertEqual({"a": ("fetch-a", "reanalyse-a"),
                          "b": ("reanalyse-b",)}, exact)

        repeated_hits = list(provider.search(
            target, (fetch, reanalyse_a, reanalyse), 3, 8))
        self.assertEqual({"a": ("reanalyse-a",), "b": ("reanalyse-b",)},
                         {hit.material.version_id: hit.task_ids
                          for hit in repeated_hits})
        self.assertEqual({"fetch-a"},
                         {item["task_id"] for item in provider.last_feedback})

        city = runner.p.Gap("search-city", "Find the permit ledger.",
            target_id=target.id, action="search", locator="permit ledger")
        harbor = runner.p.Gap("search-harbor", "Find the harbor bridge report.",
            target_id=target.id, action="search", locator="harbor bridge")
        lexical_provider = runner.TaskRoutedDevelopmentProvider(materials)
        lexical_hits = list(lexical_provider.search(target, (city, harbor), 2, 8))
        lexical = {hit.material.version_id: hit.task_ids for hit in lexical_hits}
        self.assertEqual({"a": ("search-city",), "b": ("search-harbor",)}, lexical)

        partial_fetch = runner.p.Gap("partial", "Fetch a guessed URL.",
            target_id=target.id, action="fetch", locator="https://example.org/a?guess=1")
        unknown = runner.p.Gap("unknown", "Find a lunar volcano report.",
            target_id=target.id, action="search", locator="lunar volcano")
        self.assertEqual([], list(lexical_provider.search(
            target, (partial_fetch, unknown), 3, 8)))
        self.assertEqual({"partial", "unknown"},
                         {item["task_id"] for item in lexical_provider.last_feedback})
        self.assertTrue(all(item["status"] == "corpus_exhausted"
                            for item in lexical_provider.last_feedback))
        self.assertEqual([], lexical_provider.history[-1]["returned"])

    def test_target_extension_plans_once_routes_views_and_reviews_both_rounds(self):
        status, files, calls, checkpoint_hashes, copied_hashes, input_hash = self.run_mock(
            target_extension=True)
        self.assertEqual(0, status)
        self.assertEqual(["claim_contract", "extension"], [call["stage"] for call in calls[:2]])
        self.assertEqual(1, sum(call["stage"] == "claim_contract" for call in calls))
        self.assertEqual(1, sum(call["stage"] == "extension" for call in calls))
        self.assertEqual(2, sum(call["stage"] == "judgement_critic" for call in calls))
        self.assertEqual(22, len(calls))
        row, = files["results.json"]
        self.assertEqual("completed", row["status"])
        self.assertEqual(22, row["new_api_attempts"])
        self.assertEqual(files["success-case-target-plan.json"]["sha256"], row["target_plan_sha256"])
        self.assertEqual(["claim_contract", "extension"],
                         [item["stage"] for item in files["success-case-target-planner-history.json"]])
        self.assertTrue(files["config.json"]["target_extension"])
        self.assertEqual("fixed_reanalysis", files["config.json"]["provider_mode"])
        self.assertFalse(files["config.json"]["strict_retrieval_attribution"])
        self.assertEqual("legacy", files["config.json"]["retrieval_attribution_mode"])
        self.assertTrue(files["config.json"]["trace_config"]["experimental_force_rounds"])
        self.assertEqual("target_extended_psi_development_v4_fixed_reanalysis",
                         files["config.json"]["experiment"])
        self.assertEqual(extension.PLAN_SCHEMA_VERSION,
                         files["config.json"]["target_plan_schema"])
        self.assertEqual(1, files["config.json"]["target_extension_output_repairs"])
        self.assertEqual(1, files["config.json"]["probe_result_structure_repairs"])
        self.assertIn("experiments/extended_semantic.py", copied_hashes)
        projections = files["success-case-target-plan.json"]["projections"]
        self.assertEqual({"atoms", "lineage", "critic", "evidence", "world"}, set(projections))
        self.assertEqual(1, len({view["plan_sha256"] for view in projections.values()}))
        for call in calls:
            if call["stage"] in projections:
                self.assertEqual(projections[call["stage"]], call["payload"]["target_plan"])
        report = files["success-case-report.json"]
        final = report["verification_history"][-1]
        self.assertEqual(len(projections["evidence"]["probes"]),
                         len(final["evidence_probe_results"]))
        self.assertEqual(len(projections["world"]["probes"]),
                         len(final["world_probe_results"]))

    def test_task_routed_v4_run_records_strict_attribution_end_to_end(self):
        status, files, calls, _, _, _ = self.run_mock(
            target_extension=True, provider_mode="task_routed")
        self.assertEqual(0, status)
        config = files["config.json"]
        self.assertEqual("task_routed", config["provider_mode"])
        self.assertTrue(config["strict_retrieval_attribution"])
        self.assertEqual("strict", config["retrieval_attribution_mode"])
        self.assertFalse(config["trace_config"]["experimental_force_rounds"])
        self.assertEqual("target_extended_psi_development_v4_task_routed",
                         config["experiment"])
        self.assertEqual("decision-probe-v4", config["target_plan_schema"])

        retrieval = files["success-case-retrieval.json"]
        self.assertEqual(["b"], retrieval[0]["returned"])
        self.assertEqual([{"version_id": "b",
                           "task_ids": ["origin:success-case"]}],
                         retrieval[0]["attribution"])
        issued_second = {item["id"] for item in retrieval[1]["tasks"]}
        returned_second = {item["version_id"]: set(item["task_ids"])
                           for item in retrieval[1]["attribution"]}
        self.assertEqual({"a"}, set(returned_second))
        self.assertTrue(all(task_ids and task_ids <= issued_second
                            for task_ids in returned_second.values()))

        report = files["success-case-report.json"]
        self.assertEqual("strict", report["retrieval_attribution_mode"])
        self.assertEqual(["b", "a", "b"],
                         [item["version_id"] for item in report["analysis_history"]])
        direct = [item for item in report["analysis_history"] if not item["revisit"]]
        revisits = [item for item in report["analysis_history"] if item["revisit"]]
        self.assertTrue(all(item["trigger_task_ids"] for item in direct))
        self.assertEqual(["b"], [item["version_id"] for item in revisits])
        self.assertNotIn("trigger_task_ids", revisits[0])
        atom_attribution = [call["payload"]["retrieval_attribution"]
                            for call in calls if call["stage"] == "atoms"]
        self.assertEqual(1, sum(item is None for item in atom_attribution))
        direct_attribution = [item for item in atom_attribution
                              if item is not None]
        self.assertTrue(all(item["trigger_task_ids"] and item["issued_tasks"]
                            for item in direct_attribution))
        atom_receipts = [call["payload"]["loop_receipt"]
                         for call in calls if call["stage"] == "atoms"]
        for attribution, receipt in zip(atom_attribution, atom_receipts):
            if attribution is None:
                self.assertIsNone(receipt["return_attribution"])
                self.assertEqual([], receipt["tasks"])
            else:
                self.assertIsNotNone(receipt["return_attribution"])
                self.assertEqual(attribution["trigger_task_ids"],
                                 [item["id"] for item in receipt["tasks"]])

    def test_task_routed_validator_rejects_forged_second_pass_driver_ids(self):
        status, files, _, _, _, _ = self.run_mock(
            target_extension=True, provider_mode="task_routed")
        self.assertEqual(0, status)
        report = files["success-case-report.json"]
        retrieval = files["success-case-retrieval.json"]

        forged = deepcopy(report)
        start = next(item for item in forged["operations"]
                     if item["round"] == 2 and
                     item["action"] == "verification_started")
        start["probe_owned_novel_returns"][0]["task_ids"] = ["forged-task"]
        with self.assertRaisesRegex(ValueError, "driver ledger"):
            runner._validate_outer_loop_execution(forged, retrieval, 2, "task_routed")

        forged = deepcopy(report)
        event = next(item for item in forged["operations"]
                     if item["round"] == 2 and
                     item["action"] == "probe_owned_novel_return_accepted")
        event["probe_ids"] = ["forged-probe"]
        with self.assertRaisesRegex(ValueError, "driver events"):
            runner._validate_outer_loop_execution(forged, retrieval, 2, "task_routed")

    def test_task_routed_no_active_gap_stops_after_one_real_verification(self):
        status, files, calls, _, _, _ = self.run_mock(
            target_extension=True, provider_mode="task_routed",
            routed_scenario="no_gap")
        self.assertEqual(0, status)
        row, = files["results.json"]
        report = files["success-case-report.json"]
        self.assertEqual("completed", row["status"])
        self.assertEqual("false", row["prediction"])
        self.assertEqual({"rounds": 1, "verification_calls": 1},
                         {key: row["engine_usage"][key]
                          for key in ("rounds", "verification_calls")})
        self.assertEqual([1], [item["round"] for item in row["checkpoints"]])
        self.assertEqual(1, len(files["success-case-retrieval.json"]))
        self.assertEqual("complete", report["stop_reason"])
        self.assertFalse(any(item["round"] > 1 for item in report["operations"]))
        self.assertEqual(1, sum(call["stage"] == "evidence" for call in calls))

    def test_task_routed_no_hit_records_exhaustion_without_fake_second_verification(self):
        status, files, calls, _, _, _ = self.run_mock(
            target_extension=True, provider_mode="task_routed",
            routed_scenario="no_hit")
        self.assertEqual(0, status)
        row, = files["results.json"]
        report = files["success-case-report.json"]
        retrieval = files["success-case-retrieval.json"]
        self.assertEqual("completed", row["status"])
        self.assertEqual({"rounds": 2, "verification_calls": 1},
                         {key: row["engine_usage"][key]
                          for key in ("rounds", "verification_calls")})
        self.assertEqual([1], [item["round"] for item in row["checkpoints"]])
        self.assertEqual(row["checkpoints"][0]["decision"], row["prediction"])
        self.assertEqual([], retrieval[1]["returned"])
        self.assertTrue(retrieval[1]["feedback"])
        issued = {item["id"] for item in retrieval[1]["tasks"]}
        exhausted = {item["task_id"] for item in retrieval[1]["feedback"]}
        self.assertEqual(issued, exhausted)
        self.assertEqual("provider_exhausted", report["stop_reason"])
        self.assertFalse(any(item["round"] == 2 and
                             item["action"] == "verification_started"
                             for item in report["operations"]))
        self.assertEqual(1, sum(call["stage"] == "evidence" for call in calls))

    def test_task_routed_provenance_only_hit_cannot_trigger_accuracy_second_pass(self):
        status, files, calls, _, _, _ = self.run_mock(
            target_extension=True, provider_mode="task_routed",
            routed_scenario="provenance_only")
        self.assertEqual(0, status)
        row, = files["results.json"]
        report = files["success-case-report.json"]
        retrieval = files["success-case-retrieval.json"]
        self.assertEqual({"rounds": 2, "verification_calls": 1},
                         {key: row["engine_usage"][key]
                          for key in ("rounds", "verification_calls")})
        self.assertEqual([1], [item["round"] for item in row["checkpoints"]])
        self.assertEqual(["a"], retrieval[1]["returned"])
        task_map = {item["id"]: item for item in retrieval[1]["tasks"]}
        hit_ids = retrieval[1]["attribution"][0]["task_ids"]
        self.assertTrue(all(task_map[task_id].get("probe_id") is None
                            for task_id in hit_ids))
        self.assertEqual("unresolved", report["decision_status"])
        self.assertEqual("no_probe_owned_novel_evidence", report["stop_reason"])
        self.assertEqual(1, sum(item["action"] == "verification_skipped"
                                for item in report["operations"]))
        self.assertFalse(any(item["round"] == 2 and
                             item["action"] == "verification_started"
                             for item in report["operations"]))
        self.assertEqual(1, sum(call["stage"] == "evidence" for call in calls))

    def test_task_routed_mode_requires_target_extension_before_any_artifact_write(self):
        with tempfile.TemporaryDirectory(prefix="staged-mode-") as folder:
            output = Path(folder) / "must-not-exist"
            args = SimpleNamespace(provider_mode="task_routed",
                                   target_extension=False, output=str(output))
            with self.assertRaisesRegex(ValueError, "requires --target-extension"):
                runner.run(args)
            self.assertFalse(output.exists())

    def test_v4_experiment_rejects_plan_schema_mismatch_before_artifact_write(self):
        with tempfile.TemporaryDirectory(prefix="staged-schema-") as folder:
            output = Path(folder) / "must-not-exist"
            args = SimpleNamespace(provider_mode="fixed_reanalysis",
                                   target_extension=True, output=str(output))
            with patch.object(runner, "PLAN_SCHEMA_VERSION", "decision-probe-v3"), \
                    self.assertRaisesRegex(ValueError, "requires decision-probe-v4"):
                runner.run(args)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
