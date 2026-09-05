"""Offline diagnostics contracts; no SDK, credentials, or network are needed."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


# Load this checkout's transport even when another suite imported model_io first.
_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "_transport_diagnostics_model_io", _ROOT / "experiments" / "model_io.py")
model_io = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = model_io
_SPEC.loader.exec_module(model_io)

BUDGET = model_io.Budget(calls=24, output_tokens=36000,
                        per_call_output_tokens=2500, seconds=600)
SCHEMA = {"type": "object", "properties": {"value": {"type": "integer"}},
          "required": ["value"], "additionalProperties": False}


def response(content='{"value":1}', finish_reason="stop", refusal=None,
             input_tokens=17, output_tokens=8, reasoning_tokens=None,
             missing_usage=False):
    usage = SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens)
    if reasoning_tokens is not None:
        usage.completion_tokens_details = SimpleNamespace(reasoning_tokens=reasoning_tokens)
    return SimpleNamespace(
        id="synthetic-response", model="synthetic-model", usage=None if missing_usage else usage,
        choices=[SimpleNamespace(finish_reason=finish_reason,
                                 message=SimpleNamespace(content=content, refusal=refusal))])


class SyntheticSDKError(Exception):
    pass


class TransportDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        clock = patch.object(model_io.time, "monotonic", return_value=100.0)
        self.clock = clock.start()
        self.addCleanup(clock.stop)

    def client(self, reply=None, budget=BUDGET, transport=None):
        if transport is None:
            transport = Mock(return_value=response() if reply is None else reply)
        return model_io.BudgetClient("synthetic-model", budget, transport=transport), transport

    def assert_failure(self, client, transport, error_code, error_type="RuntimeError"):
        with self.assertRaises(RuntimeError) as raised:
            client.call("Synthetic system", "Synthetic input", SCHEMA)
        self.assertEqual("Model call failed: " + error_type, str(raised.exception))
        self.assertEqual(1, transport.call_count)
        self.assertEqual(1, client.calls)
        self.assertEqual(1, len(client.records))
        record = client.records[0]
        self.assertEqual(error_code, record["error_code"])
        self.assertEqual(error_type, record["error_type"])
        return record, raised.exception

    def test_length_with_empty_output_records_reasoning_and_charges_once(self):
        client, transport = self.client(response(content=None, finish_reason="length",
            output_tokens=2500, reasoning_tokens=2500))
        record, _ = self.assert_failure(client, transport, "incomplete_model_output")
        self.assertEqual("length", record["finish_reason"])
        self.assertIs(False, record["has_refusal"])
        self.assertEqual(2500, record["reasoning_tokens"])
        self.assertEqual("", record["output"])
        self.assertEqual({"input_tokens": 17, "output_tokens": 2500}, record["usage"])
        self.assertEqual(2500, client.used_output)
        self.assertEqual(17, client.used_input)
        self.assertEqual(2500, transport.call_args.kwargs["max_completion_tokens"])
        self.assertEqual(600, transport.call_args.kwargs["timeout"])
        self.assertEqual("synthetic-response", record["response_id"])

    def test_refusal_is_distinct_and_does_not_log_refusal_text(self):
        marker = "SYNTHETIC_REFUSAL_PRIVATE_TEXT_91A"
        client, transport = self.client(response(content=None, refusal=marker))
        record, raised = self.assert_failure(client, transport, "model_refusal")
        self.assertEqual("stop", record["finish_reason"])
        self.assertIs(True, record["has_refusal"])
        self.assertNotIn(marker, json.dumps(client.records) + str(raised))
        self.assertEqual(8, client.used_output)

    def test_empty_stopped_json_output_is_distinct_from_length(self):
        client, transport = self.client(response(content=""))
        record, _ = self.assert_failure(client, transport, "invalid_json", "JSONDecodeError")
        self.assertEqual("stop", record["finish_reason"])
        self.assertIs(False, record["has_refusal"])
        self.assertEqual("", record["output"])
        self.assertEqual(8, client.used_output)

    def test_missing_usage_keeps_returned_finish_metadata_without_inventing_usage(self):
        client, transport = self.client(response(missing_usage=True))
        record, _ = self.assert_failure(client, transport, "missing_usage")
        self.assertEqual("stop", record["finish_reason"])
        self.assertIs(False, record["has_refusal"])
        self.assertEqual("synthetic-response", record["response_id"])
        self.assertNotIn("usage", record)
        self.assertNotIn("reasoning_tokens", record)
        self.assertEqual(0, client.used_input)
        self.assertEqual(0, client.used_output)

    def test_observed_output_overrun_keeps_actual_usage_and_original_request_cap(self):
        client, transport = self.client(response(output_tokens=101),
                                        budget=replace(BUDGET, output_tokens=100))
        record, _ = self.assert_failure(client, transport, "output_budget_exceeded")
        self.assertEqual(100, transport.call_args.kwargs["max_completion_tokens"])
        self.assertEqual(101, record["usage"]["output_tokens"])
        self.assertEqual(101, client.used_output)
        self.assertEqual("stop", record["finish_reason"])

    def test_observed_timeout_overrun_keeps_actual_usage_and_original_timeout(self):
        def delayed(**kwargs):
            self.clock.return_value = 701.0
            return response()
        transport = Mock(side_effect=delayed)
        client, transport = self.client(transport=transport)
        record, _ = self.assert_failure(client, transport, "timeout_budget_exceeded")
        self.assertEqual(600, transport.call_args.kwargs["timeout"])
        self.assertEqual(601, record["seconds"])
        self.assertEqual(8, client.used_output)
        self.assertEqual(17, client.used_input)
        self.assertEqual("stop", record["finish_reason"])

    def test_sdk_exception_is_one_attempt_and_does_not_copy_remote_message(self):
        marker = "SYNTHETIC_REMOTE_SECRET_TEXT_83B"
        transport = Mock(side_effect=SyntheticSDKError(marker))
        client, transport = self.client(transport=transport)
        record, raised = self.assert_failure(client, transport, "sdk_error", "SyntheticSDKError")
        self.assertNotIn(marker, json.dumps(client.records) + str(raised))
        self.assertNotIn("response_id", record)
        self.assertNotIn("usage", record)
        self.assertEqual(0, client.used_input)
        self.assertEqual(0, client.used_output)

    def test_sdk_error_code_attributes_log_only_the_two_known_values(self):
        for code in ("insufficient_quota", "rate_limit_exceeded"):
            with self.subTest(code=code):
                exc = SyntheticSDKError("SYNTHETIC_REMOTE_PRIVATE_MESSAGE")
                exc.code = code
                client, transport = self.client(transport=Mock(side_effect=exc))
                record, raised = self.assert_failure(client, transport, "sdk_error", "SyntheticSDKError")
                self.assertEqual(code, record["sdk_error_code"])
                self.assertNotIn("SYNTHETIC_REMOTE_PRIVATE_MESSAGE", json.dumps(client.records) + str(raised))

    def test_sdk_error_code_falls_back_to_nested_or_flat_body_without_string_attribute(self):
        for body, code_attribute, expected in (
            ({"error": {"code": "insufficient_quota"}}, None, "insufficient_quota"),
            ({"code": "rate_limit_exceeded"}, 429, "rate_limit_exceeded"),
        ):
            with self.subTest(body=body):
                exc = SyntheticSDKError("SYNTHETIC_REMOTE_PRIVATE_MESSAGE")
                exc.code = code_attribute
                exc.body = {**body, "private": "SYNTHETIC_REMOTE_PRIVATE_BODY"}
                client, transport = self.client(transport=Mock(side_effect=exc))
                record, raised = self.assert_failure(client, transport, "sdk_error", "SyntheticSDKError")
                self.assertEqual(expected, record["sdk_error_code"])
                logged = json.dumps(client.records) + str(raised)
                self.assertNotIn("SYNTHETIC_REMOTE_PRIVATE_MESSAGE", logged)
                self.assertNotIn("SYNTHETIC_REMOTE_PRIVATE_BODY", logged)

    def test_unknown_sdk_codes_are_omitted_and_never_copied_to_public_errors(self):
        marker = "SYNTHETIC_UNTRUSTED_CODE_PRIVATE_TEXT_40D"
        for attributes in (
            {"code": marker, "body": {"error": {"code": "insufficient_quota"}}},
            {"body": {"error": {"code": marker}}},
            {"body": {"code": marker}},
        ):
            with self.subTest(source=list(attributes)):
                exc = SyntheticSDKError(marker)
                for name, value in attributes.items():
                    setattr(exc, name, value)
                client, transport = self.client(transport=Mock(side_effect=exc))
                record, raised = self.assert_failure(client, transport, "sdk_error", "SyntheticSDKError")
                self.assertNotIn("sdk_error_code", record)
                self.assertNotIn(marker, json.dumps(client.records) + str(raised))

    def test_success_returns_same_json_and_request_without_error_diagnostics(self):
        client, transport = self.client()
        result = client.call("Synthetic system", "Synthetic input", SCHEMA)
        self.assertEqual({"value": 1}, result)
        self.assertEqual(1, transport.call_count)
        kwargs = transport.call_args.kwargs
        self.assertEqual({"model", "messages", "max_completion_tokens", "timeout", "response_format"},
                         set(kwargs))
        self.assertEqual(2500, kwargs["max_completion_tokens"])
        self.assertEqual(600, kwargs["timeout"])
        self.assertEqual({"type": "json_schema", "json_schema": {
            "name": "result", "strict": True, "schema": SCHEMA}}, kwargs["response_format"])
        record = client.records[0]
        self.assertEqual("stop", record["finish_reason"])
        self.assertIs(False, record["has_refusal"])
        self.assertNotIn("reasoning_tokens", record)
        self.assertNotIn("error_code", record)
        self.assertNotIn("error_type", record)
        self.assertEqual({"model_calls": 1, "input_tokens": 17, "output_tokens": 8, "seconds": 0},
                         client.usage())

    def test_optional_reasoning_usage_preserves_zero_as_a_reported_value(self):
        for tokens in (0, 3):
            with self.subTest(reasoning_tokens=tokens):
                client, _ = self.client(response(reasoning_tokens=tokens))
                self.assertEqual({"value": 1}, client.call("system", "input", SCHEMA))
                self.assertEqual(tokens, client.records[0]["reasoning_tokens"])
                self.assertNotIn("error_code", client.records[0])

    def test_remaining_output_cap_and_preflight_stop_behavior_are_unchanged(self):
        transport = Mock(side_effect=[response(output_tokens=7), response(output_tokens=5)])
        client, transport = self.client(transport=transport,
            budget=replace(BUDGET, output_tokens=12, per_call_output_tokens=10))
        self.assertEqual({"value": 1}, client.call("system", "first", SCHEMA))
        self.assertEqual({"value": 1}, client.call("system", "second", SCHEMA))
        self.assertEqual([10, 5], [call.kwargs["max_completion_tokens"]
                                   for call in transport.call_args_list])
        with self.assertRaisesRegex(RuntimeError, "^Per-arm resource budget exhausted$"):
            client.call("system", "third", SCHEMA)
        self.assertEqual(2, transport.call_count)
        self.assertEqual(2, client.calls)
        self.assertEqual(2, len(client.records))
        self.assertEqual(12, client.used_output)
        self.assertTrue(all("error_code" not in record for record in client.records))


if __name__ == "__main__":
    unittest.main()
