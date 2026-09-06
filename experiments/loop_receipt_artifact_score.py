"""Independent audit that v4 loop receipts reached retained model requests.

The provenance report and semantic-stage histories describe accepted engine
state.  They do not, by themselves, prove that the routing receipt was present
in the prompt sent to the model.  This module joins those ledgers to the raw
``*-calls.json`` request records and checks the exact JSON payloads.  It imports
no inference adapter and performs no network or API calls.

The audit deliberately reconstructs receipts from provider history and the
previous accepted probe assessment.  A PSI history row is used only to locate a
request in stage order; it is never accepted as evidence that a request carried
the receipt.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json


PLAN_SCHEMA_V4 = "decision-probe-v4"
PSI_STAGES = {"atoms", "lineage", "critic"}
LAYER_STAGES = {"evidence", "world"}
VERIFICATION_STAGES = LAYER_STAGES | {"judgement_critic"}
TASK_FIELDS = {
    "id", "question", "stage", "dimension", "blocking", "target_id",
    "basis", "decision_impact", "action", "locator", "probe_id",
}
RECEIPT_FIELDS = {
    "schema_version", "target_id", "return_attribution", "tasks",
    "probe_results",
}


class _DuplicateJSONKey(ValueError):
    pass


def _json_object(text, context):
    if not isinstance(text, str):
        raise ValueError(context + " is not retained as text")

    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise _DuplicateJSONKey(key)
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("non-finite JSON constant: " + value)

    try:
        value = json.loads(text, object_pairs_hook=object_pairs,
                           parse_constant=invalid_constant)
    except (json.JSONDecodeError, _DuplicateJSONKey, ValueError) as exc:
        raise ValueError(context + " is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(context + " must contain one JSON object")
    return value


def _request_digest(system, user):
    # Mirror the persisted transport format independently.  Keeping the exact
    # user string in this digest means whitespace-level request tampering is
    # visible even when it parses to the same object.
    value = json.dumps([system, user], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(value.encode()).hexdigest()


def _output_object(record, context):
    return _json_object(record.get("output"), context + " output")


def _classify_v4(payload):
    """Classify only bounded v4 semantic requests from their payload shape."""
    plan = payload.get("target_plan")
    if not isinstance(plan, dict) or plan.get("schema_version") != PLAN_SCHEMA_V4:
        return None
    stage = plan.get("stage")
    has_material = isinstance(payload.get("material"), dict)
    has_drafts = isinstance(payload.get("drafts"), dict)
    if has_material:
        if stage in {"atoms", "lineage"} and not has_drafts:
            return stage
        if stage == "critic" and has_drafts:
            return "critic"
    else:
        if stage in LAYER_STAGES and isinstance(payload.get("materials"), list):
            return stage
        if (stage == "critic" and has_drafts and
                isinstance(payload.get("materials"), list)):
            return "judgement_critic"
    raise ValueError("v4 model request has a stage/payload shape mismatch")


def _parse_calls(calls):
    if not isinstance(calls, list):
        raise ValueError("retained model calls must be an array")
    parsed = []
    for expected_sequence, record in enumerate(calls, 1):
        if (not isinstance(record, dict) or
                record.get("sequence") != expected_sequence or
                not isinstance(record.get("system"), str) or
                not isinstance(record.get("user"), str)):
            raise ValueError("retained model calls have invalid request order")
        expected_digest = _request_digest(record["system"], record["user"])
        if record.get("request_digest") != expected_digest:
            raise ValueError("retained model request digest does not match exact prompt text")
        payload = _json_object(record["user"], "retained model request")
        stage = _classify_v4(payload)
        parsed.append({"record": record, "payload": payload, "stage": stage})
    return parsed


def _validate_history(history, name, allowed_stages):
    if not isinstance(history, list):
        raise ValueError(name + " must be an array")
    for sequence, item in enumerate(history, 1):
        if (not isinstance(item, dict) or item.get("sequence") != sequence or
                item.get("stage") not in allowed_stages or
                not isinstance(item.get("status"), str)):
            raise ValueError(name + " has invalid stage request order")


def _join_history(parsed, history, stages, name):
    calls = [item for item in parsed if item["stage"] in stages]
    if len(calls) != len(history):
        raise ValueError(name + " does not join one-to-one to retained model requests")
    joined = []
    for call, ledger in zip(calls, history):
        if call["stage"] != ledger["stage"]:
            raise ValueError(name + " stage order differs from retained model requests")
        payload = call["payload"]
        target = payload.get("target")
        if not isinstance(target, dict):
            raise ValueError(name + " request lacks its immutable target")
        if call["stage"] in PSI_STAGES:
            material = payload.get("material")
            if (not isinstance(material, dict) or
                    ledger.get("material") != material.get("version_id")):
                raise ValueError(name + " material differs from retained model request")
        elif ledger.get("material") is not None:
            raise ValueError(name + " verifier row unexpectedly names one material")

        # Bind an accepted PSI ledger to the corresponding retained response as
        # well as its request.  This prevents reordering two same-stage requests
        # and then relying on the history's probe_checks alone.
        if ledger.get("status") == "accepted" and call["stage"] in {
                "atoms", "lineage"}:
            output = _output_object(call["record"], name)
            if output.get("probe_checks") != ledger.get("probe_checks"):
                raise ValueError(name + " accepted probe ledger differs from retained response")
        if ledger.get("status") == "accepted" and call["stage"] == "critic":
            output = _output_object(call["record"], name)
            if output.get("decision") != "accept":
                raise ValueError(name + " accepted critic differs from retained response")
        joined.append({**call, "ledger": ledger})
    return joined


def _task_map(record):
    tasks = record.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("strict retrieval round lacks issued tasks")
    result = {}
    for task in tasks:
        if (not isinstance(task, dict) or set(task) != TASK_FIELDS or
                not isinstance(task.get("id"), str) or not task["id"] or
                task["id"] in result):
            raise ValueError("strict retrieval round has an invalid task snapshot")
        result[task["id"]] = task
    return result


def _retrieval_hits(retrieval, target_id):
    if not isinstance(retrieval, list):
        raise ValueError("strict retrieval history must be an array")
    hits = []
    for expected_round, record in enumerate(retrieval, 1):
        if (not isinstance(record, dict) or record.get("round") != expected_round or
                record.get("kind") != "task_routed_fixed_corpus"):
            raise ValueError("strict retrieval rounds are not consecutive task-routed rounds")
        tasks = _task_map(record)
        returned, attribution = record.get("returned"), record.get("attribution")
        if (not isinstance(returned, list) or not isinstance(attribution, list) or
                len(returned) != len(attribution)):
            raise ValueError("strict retrieval round lacks a positional hit ledger")
        for version_id, hit in zip(returned, attribution):
            if (not isinstance(hit, dict) or set(hit) != {"version_id", "task_ids"} or
                    hit.get("version_id") != version_id or
                    not isinstance(hit.get("task_ids"), list) or
                    not hit["task_ids"] or len(hit["task_ids"]) != len(set(hit["task_ids"])) or
                    any(task_id not in tasks for task_id in hit["task_ids"])):
                raise ValueError("strict retrieval hit lacks exact task attribution")
            positions = [list(tasks).index(task_id) for task_id in hit["task_ids"]]
            if positions != sorted(positions):
                raise ValueError("strict retrieval hit changes issued task order")
            snapshots = [deepcopy(tasks[task_id]) for task_id in hit["task_ids"]]
            if any(task.get("target_id") not in (None, target_id) for task in snapshots):
                raise ValueError("strict retrieval hit contains another target's task")
            probe_ids = list(dict.fromkeys(
                task.get("probe_id") for task in snapshots
                if task.get("probe_id") is not None))
            hits.append({
                "round": expected_round,
                "version_id": version_id,
                "task_ids": list(hit["task_ids"]),
                "probe_ids": probe_ids,
                "tasks": snapshots,
            })
    return hits


def _direct_revisions(report, hits):
    if not isinstance(report, dict) or report.get("retrieval_attribution_mode") != "strict":
        raise ValueError("v4 receipt audit requires a strict task-routed report")
    history = report.get("analysis_history")
    if not isinstance(history, list):
        raise ValueError("strict report lacks analysis history")
    direct = [item for item in history if isinstance(item, dict) and
              "trigger_task_ids" in item]
    if len(direct) != len(hits):
        raise ValueError("direct analysis revisions do not match attributed provider hits")
    for revision, hit in zip(direct, hits):
        if (revision.get("accepted") is not True or revision.get("revisit") is not False or
                revision.get("round") != hit["round"] or
                revision.get("version_id") != hit["version_id"] or
                revision.get("trigger_task_ids") != hit["task_ids"] or
                revision.get("trigger_probe_ids") != hit["probe_ids"]):
            raise ValueError("direct analysis revision changes its provider attribution")
    return history, direct


def _prior_probe_results(report, hit):
    verification = report.get("verification_history")
    if not isinstance(verification, list):
        raise ValueError("strict report lacks accepted verification history")
    earlier = [item for item in verification if isinstance(item, dict) and
               type(item.get("round")) is int and item["round"] < hit["round"]]
    if not hit["probe_ids"]:
        return []
    if not earlier:
        raise ValueError("probe-owned return lacks a prior accepted verification")
    latest = max(earlier, key=lambda item: item["round"])
    requested = {}
    for task in hit["tasks"]:
        probe_id = task.get("probe_id")
        if probe_id is None:
            continue
        layer = task.get("dimension")
        if task.get("stage") != "verification" or layer not in LAYER_STAGES:
            raise ValueError("probe-owned return has a non-layer verification task")
        requested.setdefault((layer, probe_id), []).append(task["id"])
    if list(dict.fromkeys(key[1] for key in requested)) != hit["probe_ids"]:
        raise ValueError("trigger probe IDs differ from attributed task snapshots")
    results = []
    for (layer, probe_id), task_ids in requested.items():
        candidates = [item for item in latest.get(layer + "_probe_results", ())
                      if isinstance(item, dict) and item.get("probe_id") == probe_id]
        if len(candidates) != 1:
            raise ValueError("probe-owned return lacks one prior probe result")
        item = candidates[0]
        basis = item.get("basis")
        if (item.get("status") not in {
                "supported", "contradicted", "conflicting", "unresolved"} or
                not isinstance(basis, list) or
                not isinstance(item.get("rationale"), str) or
                not item["rationale"].strip()):
            raise ValueError("prior probe result is malformed")
        aggregate = latest.get(layer + "_verdict", latest.get("verdict"))
        if aggregate not in {"supported", "contradicted", "conflicting", "unresolved"}:
            raise ValueError("prior probe aggregate verdict is malformed")
        results.append({
            "probe_id": probe_id,
            "claim_id": item.get("claim_id"),
            "layer": layer,
            "status": item["status"],
            "basis": deepcopy(basis),
            "rationale": item["rationale"],
            "referent_relation": item.get("referent_relation", "not_applicable"),
            "aggregate_verdict": aggregate,
            "verification_round": latest["round"],
            "task_ids": list(task_ids),
        })
    return results


def _base_attribution(hit):
    return {
        "version_id": hit["version_id"],
        "trigger_task_ids": list(hit["task_ids"]),
        "trigger_probe_ids": list(hit["probe_ids"]),
        "issued_tasks": deepcopy(hit["tasks"]),
    }


def _empty_receipt(target_id):
    return {"schema_version": "loop-receipt-v1", "target_id": target_id,
            "return_attribution": None, "tasks": [], "probe_results": []}


def _analysis_gap_ids(analysis):
    gaps = analysis.get("gaps", ()) if isinstance(analysis, dict) else ()
    if not isinstance(gaps, list):
        raise ValueError("accepted analysis has a malformed gap ledger")
    result = set()
    for gap in gaps:
        if (not isinstance(gap, dict) or not isinstance(gap.get("id"), str) or
                not gap["id"]):
            raise ValueError("accepted analysis has a malformed gap")
        result.add(gap["id"])
    return result


def _analysis_resolution_ids(analysis):
    resolutions = analysis.get("resolutions", ()) if isinstance(analysis, dict) else ()
    if not isinstance(resolutions, list):
        raise ValueError("accepted analysis has a malformed resolution ledger")
    result = set()
    for resolution in resolutions:
        if (not isinstance(resolution, dict) or
                not isinstance(resolution.get("gap_id"), str) or
                not resolution["gap_id"]):
            raise ValueError("accepted analysis has a malformed resolution")
        result.add(resolution["gap_id"])
    return result


def _terminal_origin_path_state(current, target):
    """Independently rebuild terminal reachability and incomplete chains."""
    source = target.get("source_version_id")
    available = set(current)
    adjacency = {}
    origins = set()
    direct_kinds = {"quotes", "cites", "reprints", "translates", "derives"}
    for analysis in current.values():
        relations = analysis.get("relations", ()) if isinstance(analysis, dict) else ()
        origin_rows = analysis.get("origins", ()) if isinstance(analysis, dict) else ()
        if not isinstance(relations, list) or not isinstance(origin_rows, list):
            raise ValueError("accepted analysis has malformed graph findings")
        for relation in relations:
            if (isinstance(relation, dict) and relation.get("status") == "direct" and
                    relation.get("kind") in direct_kinds and
                    relation.get("from_version") in available and
                    relation.get("to_version") in available):
                adjacency.setdefault(relation["from_version"], set()).add(
                    relation["to_version"])
        for origin in origin_rows:
            if (isinstance(origin, dict) and
                    origin.get("target_id") == target.get("id") and
                    origin.get("version_id") in available):
                origins.add(origin["version_id"])
    def reachable(start):
        reached, pending = set(), [start] if start in available else []
        while pending:
            version_id = pending.pop()
            if version_id in reached:
                continue
            reached.add(version_id)
            pending.extend(adjacency.get(version_id, ()))
        return reached

    candidates = reachable(source) & origins
    downstream = {version_id: reachable(version_id) for version_id in candidates}
    terminal = {version_id for version_id in candidates
                if not ((downstream[version_id] - {version_id}) & candidates)}
    incomplete = (candidates != origins or
                  any(not (path & terminal) for path in downstream.values()))
    return bool(terminal), incomplete


def _expected_active_annotations(report, retrieval, analysis_history, target_id):
    """Replay active-at-decomposition for the exact round-start issued tasks.

    Round-start membership comes from the provider ledger already joined to the
    report by the main scorer.  Within a round, complete material replacements
    and explicit source-backed resolutions are replayed before the next direct
    return.  ``inspect-lineage`` is a registered no-gap fallback, not an active
    gap, so its annotation is always false.
    """
    target = report.get("target")
    if not isinstance(target, dict) or target.get("id") != target_id:
        raise ValueError("strict report lacks the immutable target for active replay")
    tasks_by_round = {record["round"]: _task_map(record) for record in retrieval}
    current = {}
    accepted = [item for item in analysis_history if isinstance(item, dict) and
                item.get("accepted") is True]
    annotations = []
    for round_number in range(1, len(retrieval) + 1):
        task_map = tasks_by_round[round_number]
        round_start = set(task_map)
        round_start.discard("inspect-lineage")
        material_owned_before = set().union(
            *(_analysis_gap_ids(analysis) for analysis in current.values())) if current else set()
        persistent = round_start - material_owned_before
        active = set(round_start)
        for revision in (item for item in accepted
                         if item.get("round") == round_number):
            task_ids = revision.get("trigger_task_ids")
            if task_ids is not None:
                if (not isinstance(task_ids, list) or
                        any(task_id not in task_map for task_id in task_ids)):
                    raise ValueError("direct revision has tasks outside its retrieval round")
                annotations.append([task_id in active for task_id in task_ids])
            analysis = revision.get("analysis")
            if not isinstance(analysis, dict):
                raise ValueError("accepted revision lacks analysis for active replay")
            current[revision.get("version_id")] = analysis
            owned = set().union(
                *(_analysis_gap_ids(value) for value in current.values())) if current else set()
            resolved = set().union(
                *(_analysis_resolution_ids(value) for value in current.values())) if current else set()
            active = (persistent | owned) - resolved
            active.discard("inspect-lineage")
            terminal_path, incomplete_chain = _terminal_origin_path_state(current, target)
            if incomplete_chain:
                # The runtime reopens this runner-owned task even when an old
                # or current material supplies an explicit closure.
                active.add("lineage:" + target_id)
            elif terminal_path:
                active.discard("lineage:" + target_id)
    direct_count = sum("trigger_task_ids" in item for item in accepted)
    if len(annotations) != direct_count:
        raise ValueError("active replay does not cover every direct revision")
    return annotations


def _validate_direct_receipt(payload, hit, report, target_id, expected_active):
    if payload.get("retrieval_attribution") != _base_attribution(hit):
        raise ValueError("model request retrieval_attribution differs from provider hit")
    receipt = payload.get("loop_receipt")
    if not isinstance(receipt, dict) or set(receipt) != RECEIPT_FIELDS:
        raise ValueError("model request lacks the bounded loop receipt")
    expected_return = {
        "version_id": hit["version_id"],
        "trigger_task_ids": list(hit["task_ids"]),
        "trigger_probe_ids": list(hit["probe_ids"]),
    }
    if (receipt.get("schema_version") != "loop-receipt-v1" or
            receipt.get("target_id") != target_id or
            receipt.get("return_attribution") != expected_return):
        raise ValueError("model request loop receipt changes return attribution")
    receipt_tasks = receipt.get("tasks")
    if not isinstance(receipt_tasks, list) or len(receipt_tasks) != len(hit["tasks"]):
        raise ValueError("model request loop receipt lacks exact task snapshots")
    if len(expected_active) != len(hit["tasks"]):
        raise ValueError("active replay does not cover every attributed task")
    for position, (actual, snapshot, is_active) in enumerate(zip(
            receipt_tasks, hit["tasks"], expected_active)):
        if (not isinstance(actual, dict) or
                {key: actual.get(key) for key in TASK_FIELDS} != snapshot or
                set(actual) != TASK_FIELDS | {"active_at_decomposition", "issued_order"} or
                actual.get("active_at_decomposition") is not is_active or
                actual.get("issued_order") != position):
            raise ValueError("model request loop receipt task snapshot is not exact")
    expected_results = _prior_probe_results(report, hit)
    if receipt.get("probe_results") != expected_results:
        raise ValueError("model request loop receipt changes the prior probe result")
    return deepcopy(receipt)


def _transactions(psi_joined, analysis_history):
    transactions = []
    pending = []
    for call in psi_joined:
        pending.append(call)
        ledger = call["ledger"]
        if call["stage"] == "critic" and ledger.get("status") == "accepted":
            stages = {item["stage"] for item in pending
                      if item["ledger"].get("status") == "accepted"}
            round_material = {(item["ledger"].get("round"),
                               item["ledger"].get("material")) for item in pending}
            if not {"atoms", "lineage", "critic"} <= stages or len(round_material) != 1:
                raise ValueError("accepted PSI transaction is incomplete or crosses materials")
            transactions.append(pending)
            pending = []
    if pending:
        raise ValueError("PSI history ends with an unreviewed model transaction")
    revisions = [item for item in analysis_history if isinstance(item, dict) and
                 item.get("accepted") is True]
    if len(transactions) != len(revisions):
        raise ValueError("accepted PSI transactions do not match analysis revisions")
    for transaction, revision in zip(transactions, revisions):
        ledger = transaction[0]["ledger"]
        if (ledger.get("round") != revision.get("round") or
                ledger.get("material") != revision.get("version_id")):
            raise ValueError("accepted PSI transaction is joined to the wrong revision")
    return list(zip(transactions, revisions))


def _project_receipts(receipts, layer):
    projected = []
    for receipt in receipts:
        results = [deepcopy(item) for item in receipt["probe_results"]
                   if item["layer"] == layer]
        if not results:
            continue
        task_ids = {task_id for result in results for task_id in result["task_ids"]}
        tasks = [deepcopy(item) for item in receipt["tasks"]
                 if item["id"] in task_ids]
        ordered_ids = [item["id"] for item in tasks]
        probe_ids = list(dict.fromkeys(item["probe_id"] for item in results))
        projected.append({
            "schema_version": receipt["schema_version"],
            "target_id": receipt["target_id"],
            "return_attribution": {
                "version_id": receipt["return_attribution"]["version_id"],
                "trigger_task_ids": ordered_ids,
                "trigger_probe_ids": probe_ids,
            },
            "tasks": tasks,
            "probe_results": results,
        })
    return projected


def _zero_metrics():
    return {
        "available": 0,
        "retained_calls": 0,
        "request_digests_verified": 0,
        "psi_history_calls": 0,
        "psi_calls_joined": 0,
        "verification_history_calls": 0,
        "verification_calls_joined": 0,
        "accepted_material_transactions": 0,
        "direct_attributed_transactions": 0,
        "direct_stage_calls": 0,
        "direct_stage_calls_with_exact_receipt": 0,
        "provenance_only_direct_transactions": 0,
        "revisit_transactions": 0,
        "revisit_stage_calls_with_empty_receipt": 0,
        "layer_calls": 0,
        "layer_calls_with_exact_projection": 0,
        "layer_receipt_deliveries": 0,
        "second_pass_layer_calls": 0,
        "second_pass_receipt_deliveries": 0,
        "cross_layer_leaks": 0,
        "breaks": 0,
    }


def audit_loop_receipt_artifacts(calls, psi_history, verification_call_history,
                                 report, retrieval, target_id, *, required=True):
    """Audit exact v4 receipt delivery using retained request artifacts only.

    ``verification_call_history`` is the StagedVerifier request ledger, while
    ``report['verification_history']`` contains accepted engine assessments.
    Both are required because they answer different questions.
    """
    metrics = _zero_metrics()
    if not required and all(value in (None, []) for value in (
            calls, psi_history, verification_call_history, retrieval)):
        return metrics
    if not isinstance(target_id, str) or not target_id:
        raise ValueError("loop receipt artifact audit requires one target ID")
    _validate_history(psi_history, "PSI history", PSI_STAGES)
    _validate_history(verification_call_history, "verification call history",
                      VERIFICATION_STAGES)
    parsed = _parse_calls(calls)
    psi_joined = _join_history(parsed, psi_history, PSI_STAGES, "PSI history")
    verification_joined = _join_history(
        parsed, verification_call_history, VERIFICATION_STAGES,
        "verification call history")
    if any(item["payload"]["target"].get("id") != target_id
           for item in psi_joined + verification_joined):
        raise ValueError("retained model request belongs to another target")

    hits = _retrieval_hits(retrieval, target_id)
    analysis_history, direct = _direct_revisions(report, hits)
    pairs = _transactions(psi_joined, analysis_history)
    active_annotations = _expected_active_annotations(
        report, retrieval, analysis_history, target_id)
    direct_position = 0
    hits_by_identity = {}
    for hit in hits:
        identity = (hit["round"], hit["version_id"], tuple(hit["task_ids"]),
                    tuple(hit["probe_ids"]))
        hits_by_identity.setdefault(identity, []).append(hit)

    receipts_by_round = {}
    counters = Counter()
    for transaction, revision in pairs:
        counters["accepted_material_transactions"] += 1
        is_direct = "trigger_task_ids" in revision
        if is_direct:
            identity = (revision["round"], revision["version_id"],
                        tuple(revision["trigger_task_ids"]),
                        tuple(revision["trigger_probe_ids"]))
            candidates = hits_by_identity.get(identity, [])
            if not candidates:
                raise ValueError("accepted direct PSI transaction has no provider hit")
            hit = candidates.pop(0)
            canonical_receipt = None
            expected_active = active_annotations[direct_position]
            direct_position += 1
            for call in transaction:
                receipt = _validate_direct_receipt(
                    call["payload"], hit, report, target_id, expected_active)
                if canonical_receipt is None:
                    canonical_receipt = receipt
                elif receipt != canonical_receipt:
                    raise ValueError("one direct PSI transaction received inconsistent receipts")
                counters["direct_stage_calls"] += 1
                counters["direct_stage_calls_with_exact_receipt"] += 1
            counters["direct_attributed_transactions"] += 1
            if not hit["probe_ids"]:
                if canonical_receipt["probe_results"]:
                    raise ValueError("provenance-only return exposes a prior probe result")
                counters["provenance_only_direct_transactions"] += 1
            receipts_by_round.setdefault(hit["round"], []).append(canonical_receipt)
        else:
            if revision.get("revisit") is not True:
                raise ValueError("non-attributed accepted PSI transaction is not a revisit")
            for call in transaction:
                if (call["payload"].get("retrieval_attribution") is not None or
                        call["payload"].get("loop_receipt") != _empty_receipt(target_id)):
                    raise ValueError("dependency revisit received an attributed loop receipt")
                counters["revisit_stage_calls_with_empty_receipt"] += 1
            counters["revisit_transactions"] += 1
    if any(values for values in hits_by_identity.values()):
        raise ValueError("provider hit lacks an accepted direct PSI transaction")
    if counters["direct_attributed_transactions"] != len(direct):
        raise ValueError("direct transaction count differs from report attribution")
    if direct_position != len(active_annotations):
        raise ValueError("direct transaction count differs from active replay")

    for call in verification_joined:
        stage, ledger, payload = call["stage"], call["ledger"], call["payload"]
        if stage == "judgement_critic":
            if any(key in payload for key in (
                    "current_round_receipts", "loop_receipt", "retrieval_attribution")):
                raise ValueError("judgement critic received a layer routing receipt")
            continue
        round_number = ledger.get("round")
        if type(round_number) is not int or round_number <= 0:
            raise ValueError("verification model request lacks its engine round")
        expected = _project_receipts(receipts_by_round.get(round_number, []), stage)
        if payload.get("current_round_receipts") != expected:
            raise ValueError("verification model request lacks the exact layer receipt projection")
        for receipt in payload["current_round_receipts"]:
            if (any(result.get("layer") != stage for result in receipt["probe_results"]) or
                    any(task.get("dimension") != stage for task in receipt["tasks"])):
                raise ValueError("verification receipt leaks a probe from another layer")
        counters["layer_calls"] += 1
        counters["layer_calls_with_exact_projection"] += 1
        counters["layer_receipt_deliveries"] += len(expected)
        if round_number > 1:
            counters["second_pass_layer_calls"] += 1
            counters["second_pass_receipt_deliveries"] += len(expected)

    metrics.update({
        "available": 1,
        "retained_calls": len(calls),
        "request_digests_verified": len(calls),
        "psi_history_calls": len(psi_history),
        "psi_calls_joined": len(psi_joined),
        "verification_history_calls": len(verification_call_history),
        "verification_calls_joined": len(verification_joined),
    })
    for key in counters:
        metrics[key] = counters[key]
    return metrics
