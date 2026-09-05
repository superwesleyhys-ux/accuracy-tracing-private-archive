"""Offline adversarial audit of v4 receipts in retained model-call payloads."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from loop_receipt_artifact_score import audit_loop_receipt_artifacts

try:
    from tests import test_staged_run as _staged_run_tests
except ModuleNotFoundError:  # unittest discovery also imports tests as top-level modules.
    import test_staged_run as _staged_run_tests


def request_digest(record):
    value = json.dumps([record["system"], record["user"]],
                       sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(value.encode()).hexdigest()


def payload(record):
    return json.loads(record["user"])


def replace_payload(record, value, *, rehash=True):
    record["user"] = json.dumps(value, ensure_ascii=False)
    if rehash:
        record["request_digest"] = request_digest(record)


class LoopReceiptArtifactScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = _staged_run_tests.StagedRunMockTests(
            "test_task_routed_v4_run_records_strict_attribution_end_to_end")
        status, files, _, _, _, _ = fixture.run_mock(
            target_extension=True, provider_mode="task_routed")
        if status != 0:
            raise AssertionError("offline task-routed fixture did not complete")
        cls.base_files = files

    def setUp(self):
        self.files = deepcopy(self.base_files)

    def audit(self, *, required=True):
        return audit_loop_receipt_artifacts(
            self.files["success-case-calls.json"],
            self.files["success-case-psi-history.json"],
            self.files["success-case-verification-history.json"],
            self.files["success-case-report.json"],
            self.files["success-case-retrieval.json"],
            "success-case", required=required)

    def calls(self, stage):
        return [record for record in self.files["success-case-calls.json"]
                if isinstance(payload(record).get("target_plan"), dict) and
                payload(record)["target_plan"].get("stage") == stage]

    def test_actual_offline_runner_proves_receipt_reached_raw_model_calls(self):
        result = self.audit()
        self.assertTrue(result["available"])
        self.assertEqual(result["retained_calls"],
                         result["request_digests_verified"])
        self.assertEqual(result["psi_history_calls"], result["psi_calls_joined"])
        self.assertEqual(result["verification_history_calls"],
                         result["verification_calls_joined"])
        self.assertEqual(2, result["direct_attributed_transactions"])
        self.assertEqual(result["direct_stage_calls"],
                         result["direct_stage_calls_with_exact_receipt"])
        self.assertEqual(1, result["provenance_only_direct_transactions"])
        self.assertEqual(1, result["revisit_transactions"])
        self.assertEqual(2, result["second_pass_layer_calls"])
        self.assertEqual(1, result["second_pass_receipt_deliveries"])
        self.assertEqual(0, result["cross_layer_leaks"])
        self.assertEqual(0, result["breaks"])

    def test_exact_request_digest_rejects_text_level_tampering(self):
        record = next(item for item in self.calls("atoms")
                      if payload(item)["material"]["version_id"] == "a")
        value = payload(record)
        value["loop_receipt"]["schema_version"] = "forged"
        replace_payload(record, value, rehash=False)
        with self.assertRaisesRegex(ValueError, "request digest"):
            self.audit()

    def test_recomputed_digest_cannot_hide_attribution_snapshot_tampering(self):
        record = next(item for item in self.calls("atoms")
                      if payload(item)["material"]["version_id"] == "a")
        value = payload(record)
        value["retrieval_attribution"]["issued_tasks"][-1]["question"] += " forged"
        replace_payload(record, value)
        with self.assertRaisesRegex(ValueError, "provider hit"):
            self.audit()

    def test_psi_history_alone_cannot_claim_a_receipt_reached_the_model(self):
        record = next(item for item in self.calls("atoms")
                      if payload(item)["material"]["version_id"] == "a")
        value = payload(record)
        value["loop_receipt"] = {
            "schema_version": "loop-receipt-v1", "target_id": "success-case",
            "return_attribution": None, "tasks": [], "probe_results": [],
        }
        replace_payload(record, value)
        # The independent PSI ledger is intentionally left untouched.
        with self.assertRaisesRegex(ValueError, "return attribution"):
            self.audit()

    def test_prior_probe_result_tampering_is_rejected_after_rehash(self):
        record = next(item for item in self.calls("lineage")
                      if payload(item)["material"]["version_id"] == "a")
        value = payload(record)
        value["loop_receipt"]["probe_results"][0]["status"] = "supported"
        replace_payload(record, value)
        with self.assertRaisesRegex(ValueError, "prior probe result"):
            self.audit()

    def test_loop_receipt_task_snapshot_and_order_are_exact(self):
        record = next(item for item in self.calls("critic")
                      if payload(item).get("material", {}).get("version_id") == "a")
        value = payload(record)
        value["loop_receipt"]["tasks"][0]["issued_order"] = 2
        replace_payload(record, value)
        with self.assertRaisesRegex(ValueError, "task snapshot"):
            self.audit()

    def test_second_pass_must_receive_exact_layer_projection(self):
        evidence = next(item for item in self.calls("evidence")
                        if payload(item)["current_round_receipts"])
        value = payload(evidence)
        value["current_round_receipts"] = []
        replace_payload(evidence, value)
        with self.assertRaisesRegex(ValueError, "exact layer receipt projection"):
            self.audit()

    def test_cross_layer_receipt_leak_is_rejected(self):
        evidence = next(item for item in self.calls("evidence")
                        if payload(item)["current_round_receipts"])
        evidence_receipt = deepcopy(payload(evidence)["current_round_receipts"])
        world = self.calls("world")[-1]
        value = payload(world)
        value["current_round_receipts"] = evidence_receipt
        replace_payload(world, value)
        with self.assertRaisesRegex(ValueError, "exact layer receipt projection"):
            self.audit()

    def test_provenance_only_return_is_empty_at_verifier_boundary(self):
        atoms = next(item for item in self.calls("atoms")
                     if payload(item)["material"]["version_id"] == "b" and
                     payload(item)["retrieval_attribution"] is not None)
        provenance_receipt = deepcopy(payload(atoms)["loop_receipt"])
        first_evidence = self.calls("evidence")[0]
        value = payload(first_evidence)
        value["current_round_receipts"] = [provenance_receipt]
        replace_payload(first_evidence, value)
        with self.assertRaisesRegex(ValueError, "exact layer receipt projection"):
            self.audit()

    def test_dependency_revisit_must_have_an_explicitly_empty_receipt(self):
        direct = next(item for item in self.calls("atoms")
                      if payload(item)["material"]["version_id"] == "a")
        revisit = next(item for item in self.calls("atoms")
                       if payload(item)["retrieval_attribution"] is None)
        value = payload(revisit)
        value["retrieval_attribution"] = deepcopy(payload(direct)["retrieval_attribution"])
        value["loop_receipt"] = deepcopy(payload(direct)["loop_receipt"])
        replace_payload(revisit, value)
        with self.assertRaisesRegex(ValueError, "dependency revisit"):
            self.audit()

    def test_accepted_probe_ledger_is_joined_to_retained_response(self):
        record = next(item for item in self.calls("atoms")
                      if payload(item)["material"]["version_id"] == "a")
        output = json.loads(record["output"])
        output["probe_checks"][0]["status"] = "absent"
        record["output"] = json.dumps(output)
        with self.assertRaisesRegex(ValueError, "retained response"):
            self.audit()

    def test_history_stage_reordering_cannot_rebind_requests(self):
        history = self.files["success-case-psi-history.json"]
        history[0]["stage"], history[1]["stage"] = (
            history[1]["stage"], history[0]["stage"])
        with self.assertRaisesRegex(ValueError, "stage order"):
            self.audit()

    def test_strict_json_rejects_duplicate_payload_keys(self):
        record = self.calls("evidence")[0]
        # Preserve a valid request digest so the rejection specifically proves
        # duplicate-key-safe parsing rather than stale-hash detection.
        record["user"] = record["user"][:-1] + ',"target":{} }'
        record["request_digest"] = request_digest(record)
        with self.assertRaisesRegex(ValueError, "strict JSON"):
            self.audit()

    def test_optional_absence_returns_an_explicit_unavailable_ledger(self):
        result = audit_loop_receipt_artifacts(
            [], [], [], None, [], "success-case", required=False)
        self.assertFalse(result["available"])
        self.assertEqual(0, result["breaks"])


if __name__ == "__main__":
    unittest.main()
