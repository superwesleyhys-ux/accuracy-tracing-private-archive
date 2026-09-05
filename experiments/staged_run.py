"""Development run of staged decomposition with recorded inner and outer loops.

Uses supplied snapshots and never reads gold during inference. A completed run
means valid execution, not proven source correctness. Every failed case remains
in the scheduled denominator. Existing run directories cannot be overwritten.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import getpass
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import re
import shutil
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from loop_compare import load_inputs
from model_io import Budget, BudgetClient, write
from newsverify import provenance as p
from newsverify.decisions import present_decision, round_decisions
from staged_semantic import StagedDecomposer, StagedVerifier
from extended_semantic import PLAN_SCHEMA_VERSION, TargetPlanner, project_plan


class DevelopmentProvider:
    """Repeat eligible versions for the historical reanalysis-only experiment."""
    def __init__(self, materials):
        self.materials = tuple(materials)
        self.history = []

    def search(self, target, tasks, round_number, limit):
        chosen = self.materials[:limit]
        self.history.append({"round": round_number,
            "kind": "fixed_corpus_development_reanalysis",
            "tasks": [asdict(t) for t in tasks],
            "returned": [m.version_id for m in chosen]})
        return iter(chosen)


_LEXICAL_TOKEN = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)
_LEXICAL_STOPWORDS = frozenset({
    "about", "after", "against", "also", "and", "before", "could", "does",
    "evidence", "find", "for", "from", "into", "its", "locate", "material",
    "news", "one", "record", "search", "source", "that", "the", "their",
    "this", "through", "version", "what", "when", "where", "which", "with",
})


def _lexical_terms(value):
    """Return the frozen lexical routing terms for one search string."""
    if not isinstance(value, str):
        return ()
    terms = {
        token.casefold() for token in _LEXICAL_TOKEN.findall(value)
        if (len(token) >= 2 or token.isdigit()) and
        token.casefold() not in _LEXICAL_STOPWORDS
    }
    return tuple(sorted(terms))


class TaskRoutedDevelopmentProvider:
    """Route a frozen corpus only from tasks issued in the current round.

    Round one exposes exactly the immutable target source and attributes it to
    the runner's initial origin task. Later exact fetch/reanalyse tasks and
    deterministic lexical searches select candidates; no task means no return.
    This is a closed-corpus routing test, not an open-web search adapter.
    """

    def __init__(self, materials):
        self.materials = tuple(materials)
        self.by_version = {}
        for material in self.materials:
            if material.version_id in self.by_version:
                raise ValueError("task-routed corpus has duplicate version IDs")
            self.by_version[material.version_id] = material
        self.history = []
        self.last_feedback = ()
        self.returned_versions = set()

    @staticmethod
    def _feedback(task, detail):
        return {
            "task_id": task.id,
            "probe_id": task.probe_id,
            "status": "corpus_exhausted",
            "detail": detail,
        }

    def _lexical_matches(self, task):
        query_terms = _lexical_terms(task.locator or task.question)
        if not query_terms:
            return []
        threshold = 1 if len(query_terms) == 1 else 2
        ranked = []
        for material in self.materials:
            material_terms = set(_lexical_terms(" ".join((
                material.version_id, material.url, material.issuer,
                material.content,
            ))))
            overlap = set(query_terms) & material_terms
            if len(overlap) >= threshold:
                ranked.append((len(overlap), material.version_id, material))
        return [item[2] for item in sorted(
            ranked, key=lambda item: (-item[0], item[1]))]

    def _matches(self, task):
        if task.action == "fetch":
            return sorted((material for material in self.materials
                           if task.locator in {material.url, material.version_id}),
                          key=lambda material: material.version_id)
        if task.action == "reanalyse":
            material = self.by_version.get(task.locator)
            return [] if material is None else [material]
        if task.action == "search":
            return self._lexical_matches(task)
        return []

    def search(self, target, tasks, round_number, limit):
        tasks = tuple(tasks)
        self.last_feedback = ()
        if round_number == 1:
            origin_id = "origin:" + target.id
            initial = next((task for task in tasks if task.id == origin_id), None)
            material = self.by_version.get(target.source_version_id)
            hits = ([] if initial is None or material is None or limit <= 0 else
                    [p.RetrievalHit(material, (initial.id,))])
            feedback = [self._feedback(
                task, "round one admits only the target's initial origin task")
                for task in tasks if task.id != origin_id]
            if not hits:
                fallback = initial or (p.Gap(origin_id,
                    "Initial target source was unavailable", target_id=target.id))
                if all(item["task_id"] != fallback.id for item in feedback):
                    feedback.append(self._feedback(
                        fallback, "initial target source or origin task unavailable"))
            self.last_feedback = tuple(feedback)
            self.history.append({
                "round": round_number,
                "kind": "task_routed_fixed_corpus",
                "tasks": [asdict(task) for task in tasks],
                "returned": [hit.material.version_id for hit in hits],
                "attribution": [{"version_id": hit.material.version_id,
                                 "task_ids": list(hit.task_ids)} for hit in hits],
                "feedback": list(self.last_feedback),
            })
            self.returned_versions.update(hit.material.version_id for hit in hits)
            return iter(hits)

        task_order = {task.id: index for index, task in enumerate(tasks)}
        if len(task_order) != len(tasks):
            raise ValueError("task-routed provider received duplicate task IDs")
        matched_by_version = {}
        match_strength = {}
        matches_by_task = {}
        for task in tasks:
            matches = self._matches(task)
            matches_by_task[task.id] = matches
            eligible_matches = (matches if task.action == "reanalyse" else
                                [material for material in matches
                                 if material.version_id not in self.returned_versions])
            for rank, material in enumerate(eligible_matches):
                matched_by_version.setdefault(material.version_id, []).append(task.id)
                strength = (2 if task.action in {"fetch", "reanalyse"} else 1,
                            -rank)
                match_strength[material.version_id] = max(
                    strength, match_strength.get(material.version_id, (-1, 0)))
        ordered = sorted(matched_by_version, key=lambda version_id: (
            -match_strength[version_id][0], -match_strength[version_id][1],
            min(task_order[task_id] for task_id in matched_by_version[version_id]),
            version_id,
        ))
        selected_versions = ordered[:limit]
        hits = [p.RetrievalHit(
            self.by_version[version_id],
            tuple(sorted(set(matched_by_version[version_id]),
                         key=task_order.__getitem__)),
        ) for version_id in selected_versions]
        hit_tasks = {task_id for version_id in selected_versions
                     for task_id in matched_by_version[version_id]}
        feedback = []
        for task in tasks:
            if task.id in hit_tasks:
                continue
            matches = matches_by_task[task.id]
            if (task.action != "reanalyse" and matches and
                    all(material.version_id in self.returned_versions
                        for material in matches)):
                detail = "all exact or lexical matches were already returned in an earlier round"
            elif matches:
                detail = "novel matches exceeded this round's frozen document capacity"
            else:
                detail = "no exact or lexical match in the frozen corpus"
            feedback.append(self._feedback(task, detail))
        self.last_feedback = tuple(feedback)
        self.history.append({
            "round": round_number,
            "kind": "task_routed_fixed_corpus",
            "tasks": [asdict(task) for task in tasks],
            "returned": [hit.material.version_id for hit in hits],
            "attribution": [{"version_id": hit.material.version_id,
                             "task_ids": list(hit.task_ids)} for hit in hits],
            "feedback": list(self.last_feedback),
        })
        self.returned_versions.update(hit.material.version_id for hit in hits)
        return iter(hits)


def _validate_outer_loop_execution(report, retrieval, max_rounds, provider_mode):
    """Fail closed on the runner's fixed or genuinely adaptive outer-loop contract."""
    if (not isinstance(report, dict) or not isinstance(retrieval, list) or
            type(max_rounds) is not int or max_rounds < 1):
        raise ValueError("Outer-loop audit inputs are malformed")
    usage = report.get("usage")
    history = report.get("verification_history")
    operations = report.get("operations")
    if not all(isinstance(value, (dict, list)) for value in
               (usage, history, operations)):
        raise ValueError("Outer-loop audit collections are missing")
    verification_calls = usage.get("verification_calls")
    search_rounds = usage.get("rounds")
    accepted_rounds = [item.get("round") for item in history
                       if isinstance(item, dict)]
    if (type(verification_calls) is not int or type(search_rounds) is not int or
            verification_calls != len(history) or
            accepted_rounds != list(range(1, verification_calls + 1))):
        raise ValueError("Accepted verification history does not match engine usage")
    if provider_mode == "fixed_reanalysis":
        if (verification_calls != max_rounds or search_rounds != max_rounds or
                len(retrieval) != max_rounds or
                report.get("retrieval_attribution_mode") != "legacy-compatible"):
            raise ValueError("Fixed reanalysis did not complete every forced round")
        return
    if provider_mode != "task_routed":
        raise ValueError("Unknown outer-loop provider mode")
    if (not 1 <= verification_calls <= max_rounds or
            not verification_calls <= search_rounds <= max_rounds or
            len(retrieval) != search_rounds or
            report.get("retrieval_attribution_mode") != "strict" or
            [item.get("round") if isinstance(item, dict) else None for item in retrieval] !=
            list(range(1, search_rounds + 1))):
        raise ValueError("Task-routed execution is outside its adaptive round bounds")
    returned_before = set()
    duplicate_by_round = {}
    probe_owned_novel_by_round = {}
    for record in retrieval:
        if (not isinstance(record, dict) or
                not all(isinstance(record.get(key), list)
                        for key in ("tasks", "returned", "attribution", "feedback"))):
            raise ValueError("Task-routed retrieval round lacks complete outcome ledgers")
        task_map = {item.get("id"): item for item in record["tasks"]
                    if isinstance(item, dict) and isinstance(item.get("id"), str)}
        if len(task_map) != len(record["tasks"]):
            raise ValueError("Task-routed retrieval round repeats or malforms a task")
        returned = record.get("returned") if isinstance(record, dict) else None
        if (not isinstance(returned, list) or
                any(not isinstance(item, str) or not item for item in returned) or
                len(returned) != len(set(returned))):
            raise ValueError("Task-routed retrieval malformed a material return")
        if ([item.get("version_id") if isinstance(item, dict) else None
             for item in record["attribution"]] != returned):
            raise ValueError("Task-routed return and attribution ledgers disagree")
        hit_task_ids = set()
        duplicates = set(returned) & returned_before
        probe_owned_novel_drivers = []
        task_order = {item["id"]: index for index, item in enumerate(record["tasks"])}
        for item in record["attribution"]:
            task_ids = item.get("task_ids")
            if (not isinstance(task_ids, list) or not task_ids or
                    len(task_ids) != len(set(task_ids)) or
                    not set(task_ids) <= set(task_map)):
                raise ValueError("Task-routed return has invalid task attribution")
            if item["version_id"] in duplicates and any(
                    task_map[task_id].get("action") != "reanalyse"
                    for task_id in task_ids):
                raise ValueError("Only an explicit reanalyse task may return a seen version")
            if record["round"] > 1 and item["version_id"] not in duplicates:
                qualifying_task_ids = sorted((task_id for task_id in task_ids
                    if task_map[task_id].get("stage") == "verification" and
                    task_map[task_id].get("dimension") in {"evidence", "world"} and
                    task_map[task_id].get("probe_id") is not None and
                    task_map[task_id].get("action") in {"fetch", "search"}),
                    key=task_order.__getitem__)
                if qualifying_task_ids:
                    probe_owned_novel_drivers.append({
                        "version_id": item["version_id"],
                        "task_ids": qualifying_task_ids,
                        "probe_ids": list(dict.fromkeys(
                            task_map[task_id].get("probe_id")
                            for task_id in qualifying_task_ids)),
                    })
            hit_task_ids.update(task_ids)
        exhausted_task_ids = set()
        for item in record["feedback"]:
            task_id = item.get("task_id") if isinstance(item, dict) else None
            if (task_id not in task_map or item.get("status") != "corpus_exhausted" or
                    item.get("probe_id") != task_map[task_id].get("probe_id") or
                    task_id in exhausted_task_ids):
                raise ValueError("Task-routed exhaustion outcome is invalid")
            exhausted_task_ids.add(task_id)
        if (hit_task_ids & exhausted_task_ids or
                hit_task_ids | exhausted_task_ids != set(task_map)):
            raise ValueError("Every issued task needs exactly one hit-or-exhausted outcome")
        duplicate_by_round[record["round"]] = duplicates
        probe_owned_novel_by_round[record["round"]] = probe_owned_novel_drivers
        returned_before.update(returned)

    by_round = {}
    if [item.get("sequence") if isinstance(item, dict) else None
            for item in operations] != list(range(1, len(operations) + 1)):
        raise ValueError("Outer-loop operation sequence is not canonical")
    for operation in operations:
        if isinstance(operation, dict):
            by_round.setdefault(operation.get("round"), []).append(operation)
    for round_number in range(2, search_rounds + 1):
        driver_events = [{key: item.get(key) for key in
                          ("version_id", "task_ids", "probe_ids")}
                         for item in by_round.get(round_number, [])
                         if item.get("action") ==
                         "probe_owned_novel_return_accepted"]
        if driver_events != probe_owned_novel_by_round.get(round_number, []):
            raise ValueError("Later verification driver events are incomplete")
    for round_number in accepted_rounds[1:]:
        returned = retrieval[round_number - 1].get("returned", [])
        if not probe_owned_novel_by_round.get(round_number):
            raise ValueError(
                "A later verification ran without a probe-owned novel fetch/search hit")
        round_operations = by_round.get(round_number, [])
        verification_start_records = [item for item in round_operations
                                      if item.get("action") == "verification_started"]
        verification_starts = [item.get("sequence")
                               for item in verification_start_records]
        if len(verification_starts) != 1:
            raise ValueError("A later adaptive round lacks one verification start")
        verification_start = verification_starts[0]
        expected_drivers = probe_owned_novel_by_round[round_number]
        if verification_start_records[0].get(
                "probe_owned_novel_returns") != expected_drivers:
            raise ValueError("Later verification driver ledger is incomplete")
        if any(item.get("sequence", verification_start) >= verification_start
               for item in round_operations
               if item.get("action") == "probe_owned_novel_return_accepted"):
            raise ValueError("A later verification started before its driver was accepted")
        for version_id in returned:
            attributed = [item.get("sequence") for item in round_operations
                          if item.get("action") == "retrieval_attribution_validated" and
                          item.get("version_id") == version_id]
            lifecycle_action = ("duplicate_observed"
                                if version_id in duplicate_by_round.get(round_number, set())
                                else "snapshot_saved")
            saved = [item.get("sequence") for item in round_operations
                     if item.get("action") == lifecycle_action and
                     item.get("version_id") == version_id and item.get("eligible") is True]
            decomposed = [item.get("sequence") for item in round_operations
                          if item.get("action") == "decompose_completed" and
                          item.get("version_id") == version_id]
            if (len(attributed) != 1 or len(saved) != 1 or not decomposed or
                    not attributed[0] < saved[0] < min(decomposed) < verification_start):
                raise ValueError(
                    "A later task hit did not follow attribute-save-decompose-verify order")

    if search_rounds > verification_calls:
        final_retrieval = retrieval[-1]
        exhausted_stop = (not final_retrieval.get("returned") and
                          bool(final_retrieval.get("feedback")) and
                          report.get("stop_reason") == "provider_exhausted")
        non_probe_stop = (bool(final_retrieval.get("returned")) and
                          not probe_owned_novel_by_round.get(search_rounds) and
                          report.get("stop_reason") ==
                          "no_probe_owned_novel_evidence" and
                          report.get("decision_status") == "unresolved")
        if non_probe_stop:
            skipped = [item for item in by_round.get(search_rounds, [])
                       if item.get("action") == "verification_skipped"]
            non_probe_stop = (len(skipped) == 1 and
                              skipped[0].get("reason") ==
                              "no_probe_owned_novel_return")
        if (search_rounds != verification_calls + 1 or
                not (exhausted_stop or non_probe_stop)):
            raise ValueError("Adaptive no-hit stopping was not a single audited exhaustion round")
    elif verification_calls == 1 and search_rounds == 1 and report.get("stop_reason") != "complete":
        raise ValueError("A one-round task-routed run did not stop on a complete state")


def run(args):
    provider_mode = getattr(args, "provider_mode", "fixed_reanalysis")
    if provider_mode not in {"fixed_reanalysis", "task_routed"}:
        raise ValueError("Unknown development provider mode")
    extension_enabled = bool(getattr(args, "target_extension", False))
    if provider_mode == "task_routed" and not extension_enabled:
        raise ValueError("task_routed provider mode requires --target-extension")
    if extension_enabled and PLAN_SCHEMA_VERSION != "decision-probe-v4":
        raise ValueError(
            "v4 staged experiment requires decision-probe-v4 target plans")
    strict_retrieval_attribution = provider_mode == "task_routed"
    data = load_inputs(args.inputs)
    if args.cases:
        selected = set(args.cases.split(","))
        known = {c["target"]["id"] for c in data["cases"]}
        if not selected <= known:
            raise ValueError("Unknown selected development case")
        data = {**data, "cases": [c for c in data["cases"] if c["target"]["id"] in selected]}
    if not data["cases"]:
        raise ValueError("No development cases selected")
    if any(not isinstance(c["target"]["id"], str) or
           not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", c["target"]["id"])
           for c in data["cases"]):
        raise ValueError("Case IDs must be safe, bounded artifact filenames")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    budget = Budget(calls=args.max_calls, output_tokens=args.output_tokens,
                    per_call_output_tokens=args.per_call_tokens, seconds=args.seconds)
    trace_config = p.TraceConfig(max_rounds=args.rounds, max_documents=24,
        max_decomposition_calls=24,
        experimental_force_rounds=not strict_retrieval_attribution)
    used_experiments = ["staged_run.py", "staged_semantic.py", "extended_semantic.py", "model_io.py",
                        "loop_compare.py", "semantic_adapter.py"]
    source_files = list((ROOT / "newsverify").glob("*.py")) + [ROOT / "experiments" / n for n in used_experiments]
    experiment = ("target_extended_psi_development_v4_task_routed"
                  if strict_retrieval_attribution else
                  "target_extended_psi_development_v4_fixed_reanalysis"
                  if extension_enabled else "staged_psi_development_v1")
    config = {"experiment": experiment, "model": args.model,
        "reasoning_effort": args.reasoning_effort, "budget_per_case": asdict(budget),
        "trace_config": asdict(trace_config), "max_inner_repairs": args.max_repairs,
        "target_extension": extension_enabled,
        "target_plan_schema": PLAN_SCHEMA_VERSION if extension_enabled else None,
        "target_structure_repairs": args.max_repairs if extension_enabled else None,
        "target_extension_repairs": args.max_repairs if extension_enabled else None,
        "target_extension_output_repairs": args.max_repairs if extension_enabled else None,
        "material_stage_repairs": args.max_repairs,
        "judgement_repairs": args.max_repairs if extension_enabled else None,
        "probe_result_structure_repairs": args.max_repairs if extension_enabled else None,
        "target_extension_contract": ("one immutable target contract and decision-probe extension per case; "
            "one grounded result per routed probe; Python aggregates target logic; bounded target, "
            "result-structure and judgement repair; plan never counts as evidence" if extension_enabled else None),
        "workers": args.workers, "ordering_seed": 20260906,
        "case_ids": [c["target"]["id"] for c in data["cases"]],
        "scheduled_cases": len(data["cases"]), "gold_read_during_inference": False,
        "new_response_only": True, "automatic_transport_retries": 0,
        "dataset_status": "previously_seen_development_cases_not_hidden_benchmark",
        "loop_contract": ("every novel task-attributed material return uses atoms, lineage and critic; "
                          "verification is adaptive within the round cap"
                          if strict_retrieval_attribution else
                          "every material return uses atoms, lineage and critic; targeted bounded "
                          "inner repair; forced outer rounds"),
        "provider_mode": provider_mode,
        "strict_retrieval_attribution": strict_retrieval_attribution,
        "retrieval_attribution_mode": ("strict" if strict_retrieval_attribution
                                       else "legacy"),
        "provider": ("task-routed fixed eligible snapshots, no open-web collection"
                     if strict_retrieval_attribution else
                     "fixed eligible snapshots, no open-web collection"),
        "input_sha256": hashlib.sha256(Path(args.inputs).read_bytes()).hexdigest(),
        "source_sha256": {f.relative_to(ROOT).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
                          for f in sorted(source_files)}}
    write(output / "config.json", config)
    write(output / "inputs.json", data)
    for source in source_files:
        dest = output / "executed-code" / source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    if not os.environ.get("OPENAI_API_KEY"):
        write(output / "status.json", {"status": "blocked_missing_auth", "new_api_attempts": 0})
        return 2

    def one(case):
        identifier = case["target"]["id"]

        class RecordedClient(BudgetClient):
            def __init__(self):
                super().__init__(args.model, budget, reasoning_effort=args.reasoning_effort)
                self.recording_lock = threading.Lock()

            def call(self, *a, **kw):
                with self.recording_lock:
                    try:
                        result = super().call(*a, **kw)
                        record = self.records[-1]
                        if record.get("actual_model") != args.model:
                            record.update(error_type="UnexpectedModel", error_code="unexpected_model")
                            raise ValueError("Returned model differs from configured model")
                        if record["usage"]["output_tokens"] > record["max_completion_tokens"]:
                            record.update(error_type="OutputCapExceeded", error_code="per_call_cap_exceeded")
                            raise ValueError("Observed completion cap exceeded")
                        return result
                    finally:
                        write(output / f"{identifier}-calls.json", self.records)
                        print(json.dumps({"case": identifier, "calls": self.calls}), flush=True)

        client = planner = plan = psi = verifier = provider = None
        row = {"id": identifier, "assessment_mode": case["target"]["assessment_mode"],
            "status": "error", "prediction": None, "checkpoints": []}
        hashes = []

        def capture(checkpoint):
            text = p.canonical_checkpoint_json(checkpoint)
            round_number = checkpoint.state["usage"]["rounds"]
            (output / f"{identifier}-checkpoint-{round_number}.json").write_text(text + "\n")
            hashes.append({"round": round_number, "sha256": p.checkpoint_sha256(checkpoint)})

        try:
            client = RecordedClient()
            target = p.Target(**{**case["target"], "evidence_scope": tuple(case["target"]["evidence_scope"])})
            materials = [p.MaterialVersion(**m) for m in case["materials"]]
            cutoff = p._time(target.as_of, "as_of")
            eligible = [m for m in materials if not p._material_eligibility(m, cutoff)]
            target_plan = None
            if extension_enabled:
                planner = TargetPlanner(
                    client,
                    max_repairs=args.max_repairs,
                    max_structure_repairs=args.max_repairs,
                    max_output_repairs=args.max_repairs,
                )
                plan = planner.prepare(target)
                target_plan = {stage: project_plan(plan, stage).to_payload()
                               for stage in ("atoms", "lineage", "critic", "evidence", "world")}
                write(output / f"{identifier}-target-plan.json",
                      {**plan.to_payload(), "sha256": plan.sha256, "projections": target_plan})
                row["target_plan_sha256"] = plan.sha256
            psi = StagedDecomposer(client, max_repairs=args.max_repairs, target_plan=target_plan)
            verifier = StagedVerifier(
                client,
                target_plan=target_plan,
                max_repairs=args.max_repairs,
                max_structure_repairs=args.max_repairs,
            )
            provider = (TaskRoutedDevelopmentProvider(eligible)
                        if strict_retrieval_attribution else DevelopmentProvider(eligible))
            report = p.run_provenance(target, provider, psi, verifier, trace_config,
                checkpoint_callback=capture,
                strict_retrieval_attribution=strict_retrieval_attribution)
            retrieval_history = getattr(provider, "history", [])
            write(output / f"{identifier}-report.json", report)
            row["checkpoints"] = round_decisions(report)
            row["native_prediction"] = present_decision(report)
            row["engine_usage"] = report["usage"]
            if not row["native_prediction"]["assessment_valid"]:
                engine_errors = report.get("errors", [])
                if engine_errors:
                    engine_error = engine_errors[-1]
                    row["error_type"] = engine_error.get("type", "InvalidAssessment")
                    row["error_stage"] = (engine_error.get("semantic_stage") or
                                          engine_error.get("stage"))
                call_error = next((record for record in reversed(client.records)
                                   if record.get("error_type")), None)
                if call_error:
                    row["error_code"] = call_error.get("error_code", "model_call_failed")
                elif verifier.history:
                    row["error_code"] = verifier.history[-1].get(
                        "status", "engine_assessment_invalid")
                else:
                    row["error_code"] = "engine_assessment_invalid"
                raise ValueError("Engine did not accept a complete valid assessment")
            _validate_outer_loop_execution(
                report, retrieval_history, args.rounds, provider_mode)
            if any(r.get("error_type") for r in client.records):
                raise ValueError("A failed model call cannot be ignored")
            row["prediction"] = row["native_prediction"]["decision"]
            row["status"] = "completed"
        except Exception as exc:
            row.setdefault("error_type", type(exc).__name__)
            if getattr(exc, "stage", None):
                row.setdefault("error_stage", exc.stage)
        finally:
            row["usage"] = client.usage() if client else {k: 0 for k in
                ("model_calls", "input_tokens", "output_tokens", "seconds")}
            row["checkpoint_hashes"] = hashes
            records = client.records if client else []
            row["new_api_attempts"] = len(records)
            row["returned_responses"] = sum(bool(r.get("response_id")) for r in records)
            write(output / f"{identifier}-psi-history.json", getattr(psi, "history", []))
            write(output / f"{identifier}-verification-history.json", getattr(verifier, "history", []))
            write(output / f"{identifier}-target-planner-history.json", getattr(planner, "history", []))
            write(output / f"{identifier}-retrieval.json", getattr(provider, "history", []))
            write(output / f"{identifier}-result.json", row)
            print(json.dumps({"case": identifier, "status": row["status"]}), flush=True)
        return row

    cases = list(data["cases"])
    random.Random(20260906).shuffle(cases)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, case): case["target"]["id"] for case in cases}
        for future in as_completed(futures):
            rows.append(future.result())
            write(output / "results.json", sorted(rows, key=lambda r: r["id"]))
    write(output / "status.json", {"status": "completed" if all(r["status"] == "completed" for r in rows) else "has_errors",
        "scheduled_cases": len(cases), "completed": sum(r["status"] == "completed" for r in rows),
        "new_api_attempts": sum(r["new_api_attempts"] for r in rows),
        "returned_responses": sum(r["returned_responses"] for r in rows),
        "recorded_usage": {k: sum(r["usage"][k] for r in rows) for k in ["model_calls", "input_tokens", "output_tokens"]}})
    return 0 if all(r["status"] == "completed" for r in rows) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", help="Comma-separated development IDs; default all")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--per-call-tokens", type=int, default=4000)
    parser.add_argument("--output-tokens", type=int, default=64000)
    parser.add_argument("--max-calls", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=900)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-repairs", type=int, default=1)
    parser.add_argument("--target-extension", action="store_true",
                        help="Plan immutable target clauses and decision probes before material analysis")
    parser.add_argument("--provider-mode",
                        choices=("fixed_reanalysis", "task_routed"),
                        default="fixed_reanalysis",
                        help="Use historical fixed reanalysis or strict task-routed frozen retrieval")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prompt-key", action="store_true")
    args = parser.parse_args()
    if not (1 <= args.rounds <= 3 and 1 <= args.workers <= 2 and 0 <= args.max_repairs <= 1):
        parser.error("rounds=1..3, workers=1..2, max-repairs=0..1")
    if args.provider_mode == "task_routed" and not args.target_extension:
        parser.error("task_routed provider mode requires --target-extension")
    if not (500 <= args.per_call_tokens <= 16000 and args.output_tokens >= args.per_call_tokens
            and 1 <= args.max_calls <= 128 and 0 < args.seconds <= 1800):
        parser.error("Invalid bounded development budget")
    logging.disable(logging.CRITICAL)
    if args.prompt_key:
        if not sys.stdin.isatty():
            raise SystemExit("Echo-disabled terminal required")
        os.environ["OPENAI_API_KEY"] = getpass.getpass("API credential (hidden): ")
    try:
        return run(args)
    finally:
        if args.prompt_key:
            os.environ.pop("OPENAI_API_KEY", None)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "stopped", "error_type": type(exc).__name__}), flush=True)
        raise SystemExit(1)
