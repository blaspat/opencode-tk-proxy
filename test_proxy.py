"""Tests for opencode-tk-proxy compression (tools array, structural JSON, skip counters).

Run: ~/.hermes/hermes-agent/venv/bin/python -m unittest -v test_proxy
"""
import json
import os
import sys
import tempfile
import threading
import unittest
import asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx
from fastapi.testclient import TestClient

import proxy


class _FakeCtx:
    def __init__(self, resp):
        self.resp = resp
        self.exited = False

    async def __aenter__(self):
        return self.resp

    async def __aexit__(self, *a):
        self.exited = True


class _FakeUpstream:
    """Minimal stand-in for httpx.AsyncClient: records the request, returns a canned response."""
    def __init__(self, resp):
        self.resp = resp
        self.last = None

    def stream(self, **kwargs):
        self.last = kwargs
        return _FakeCtx(self.resp)

    async def aclose(self):
        pass


def _resp(content=b'{"ok": true}', status=200, ctype="application/json"):
    req = httpx.Request("POST", "http://fake/v1/chat/completions")
    return httpx.Response(status, headers={"content-type": ctype}, content=content, request=req)


class CompressorTests(unittest.TestCase):
    def test_tools_preserve_structure(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "search",
                "description": "x" * 500,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "q": {"type": "string", "description": "y" * 300},
                        "n": {"type": "integer", "enum": [1, 2, 3]},
                    },
                    "required": ["q"],
                },
            },
        }]
        new_tools, saved = proxy._compress_tools(tools)
        fn = new_tools[0]["function"]
        self.assertGreater(saved, 0)
        self.assertEqual(fn["name"], "search")
        self.assertLess(len(fn["description"]), 130)
        self.assertEqual(fn["parameters"]["required"], ["q"])
        self.assertEqual(fn["parameters"]["properties"]["n"]["enum"], [1, 2, 3])
        json.dumps(new_tools)  # must stay valid JSON

    def test_json_content_structural(self):
        big = {"status": "ok", "items": [{"id": i, "data": "z" * 5000} for i in range(5)], "count": 5}
        text = json.dumps(big)
        out = proxy._compress_json_content(text)
        self.assertLess(len(out), len(text))
        parsed = json.loads(out)  # must stay parseable
        self.assertEqual(parsed["count"], 5)
        self.assertEqual(parsed["items"][0]["id"], 0)
        self.assertLess(len(parsed["items"][0]["data"]), 1000)
        self.assertEqual(set(parsed.keys()), {"status", "items", "count"})

    def test_json_content_no_gain_returns_original(self):
        self.assertEqual(proxy._compress_json_content('{"a":1}'), '{"a":1}')
        not_json = "hello world " * 100
        self.assertEqual(proxy._compress_json_content(not_json), not_json)

    def test_maybe_json_content(self):
        self.assertTrue(proxy._maybe_json_content(json.dumps({"a": ["x" * 2000]})))
        self.assertFalse(proxy._maybe_json_content('{"a": 1}'))       # too small
        self.assertFalse(proxy._maybe_json_content("plain " * 300))    # not JSON

    def test_skip_counters(self):
        skipped = {}
        out, saved = proxy._try_compress_message({"role": "tool", "content": "short"}, skipped)
        self.assertEqual(saved, 0)
        self.assertEqual(skipped.get("too_small"), 1)

        skipped = {}
        proxy._try_compress_message(
            {"role": "assistant", "content": "x" * 500, "tool_calls": [{"id": "c1"}]}, skipped)
        self.assertEqual(skipped.get("tool_calls"), 1)

        skipped = {}
        proxy._try_compress_message({"role": "tool", "content": "def f():\n    pass\n" * 50}, skipped)
        self.assertEqual(skipped.get("verbatim"), 1)

    def test_big_json_tool_result_gets_compressed(self):
        big = {"rows": [{"id": i, "payload": "p" * 3000} for i in range(20)]}
        msg = {"role": "tool", "tool_call_id": "t1", "content": json.dumps(big)}
        skipped = {}
        out, saved = proxy._try_compress_message(msg, skipped)
        self.assertGreater(saved, 0)
        content, _, _ = out["content"].partition("\n[ccr:")
        parsed = json.loads(content)  # marker appended AFTER valid JSON
        self.assertEqual(len(parsed["rows"]), 20)


class ProxyEndToEndTests(unittest.TestCase):
    def setUp(self):
        proxy.REQUEST_STATS.clear()
        proxy._recovery_local = threading.local()
        tmp = tempfile.mkdtemp()
        proxy._RECOVERY_DB = os.path.join(tmp, "recovery.db")

    def test_full_request_tools_and_json_compressed(self):
        big_tool = {"rows": [{"id": i, "data": "d" * 4000} for i in range(10)]}
        payload = {
            "model": "test",
            "messages": [
                {"role": "system", "content": "S" * 1500},
                {"role": "tool", "tool_call_id": "t1", "content": json.dumps(big_tool)},
                {"role": "user", "content": "summary"},
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "fetch",
                    "description": "F" * 600,
                    "parameters": {
                        "type": "object",
                        "properties": {"url": {"type": "string", "description": "u" * 400}},
                        "required": ["url"],
                    },
                },
            }],
        }
        upstream = _FakeUpstream(_resp(b'{"ok": true}'))
        with TestClient(proxy.app) as client:
            proxy._http = upstream  # after lifespan startup clobbers _http
            r = client.post("/v1/chat/completions", json=payload)
        self.assertEqual(r.status_code, 200)
        sent = json.loads(upstream.last["content"])

        # tools array compressed but structure intact
        fn = sent["tools"][0]["function"]
        self.assertLess(len(fn["description"]), 130)
        self.assertEqual(fn["name"], "fetch")
        # tool result JSON message compressed
        tool_msg = [m for m in sent["messages"] if m["role"] == "tool"][0]
        self.assertLess(len(tool_msg["content"]), len(json.dumps(big_tool)))
        # small user message untouched
        user_msg = [m for m in sent["messages"] if m["role"] == "user"][0]
        self.assertEqual(user_msg["content"], "summary")

        self.assertIn("x-compressed", r.headers)
        entry = proxy.REQUEST_STATS[-1]
        self.assertGreater(entry["tools_saved_chars"], 0)
        self.assertIn("skipped", entry)

    def test_upstream_usage_captured_non_stream(self):
        usage_body = json.dumps({
            "id": "gen-1", "object": "chat.completion", "model": "mimo-v2.5",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 123, "completion_tokens": 4, "total_tokens": 127,
                      "prompt_tokens_details": {"cached_tokens": 100}},
            "cost": "0.0001",
        }).encode()
        payload = {"model": "test", "messages": [
            {"role": "tool", "tool_call_id": "t1",
             "content": json.dumps({"rows": [{"d": "q" * 4000} for _ in range(8)]})},
            {"role": "user", "content": "hi"},
        ]}
        upstream = _FakeUpstream(_resp(usage_body))
        with TestClient(proxy.app) as client:
            proxy._http = upstream
            r = client.post("/v1/chat/completions", json=payload)
        self.assertEqual(r.status_code, 200)
        entry = proxy.REQUEST_STATS[-1]
        self.assertEqual(entry["upstream_prompt_tokens"], 123)
        self.assertEqual(entry["upstream_cost"], "0.0001")
        self.assertEqual(entry["upstream_usage"]["prompt_tokens_details"]["cached_tokens"], 100)

    def test_token_estimate_and_backfill(self):
        self.assertEqual(proxy._estimate_tokens(""), 0)
        self.assertEqual(proxy._estimate_tokens("hello world"), 2)  # o200k
        self.assertGreater(proxy._estimate_tokens("x" * 10000), 100)

        proxy.REQUEST_STATS.clear()
        req_id = "test-abc"
        proxy.REQUEST_STATS.append({"req_id": req_id, "chars_saved": 1})
        orig = json.dumps({"rows": [{"id": i, "data": "y" * 3000} for i in range(10)]}).encode()
        comp = proxy._compress_json_content(orig.decode()).encode()
        asyncio.run(proxy._backfill_token_stats(req_id, orig, comp))
        entry = proxy.REQUEST_STATS[-1]
        self.assertGreater(entry["tokens_est_original"], 0)
        self.assertGreater(entry["tokens_est_saved"], 0)
        self.assertLess(entry["tokens_est_compressed"], entry["tokens_est_original"])
        self.assertEqual(entry["tokens_est_method"], "tiktoken")

    def test_non_json_body_passthrough(self):
        upstream = _FakeUpstream(_resp(b'{"ok": true}'))
        with TestClient(proxy.app) as client:
            proxy._http = upstream
            r = client.post("/v1/chat/completions", content=b"not json at all", headers={"content-type": "text/plain"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(upstream.last["content"], b"not json at all")

    def test_responses_api_flat_tools_and_input_compressed(self):
        big_json = json.dumps({"rows": [{"d": "q" * 4000} for _ in range(8)]})
        payload = {
            "model": "test",
            "instructions": "I" * 800,
            "input": [
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "U" * 1500}]},
                {"type": "function_call", "call_id": "c1", "name": "fetch",
                 "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": big_json},
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "go"}]},
            ],
            "tools": [{
                "type": "function",
                "name": "fetch",
                "description": "x" * 500,
                "parameters": {"type": "object", "properties": {
                    "q": {"type": "string", "description": "y" * 300}}},
            }],
        }
        upstream = _FakeUpstream(_resp(b'{"usage": {"input_tokens": 42}}'))
        with TestClient(proxy.app) as client:
            proxy._http = upstream
            r = client.post("/v1/responses", json=payload)
        self.assertEqual(r.status_code, 200)
        sent = json.loads(upstream.last["content"])
        # flat tools stay flat, description trimmed, schema compressed
        tool = sent["tools"][0]
        self.assertEqual(tool["name"], "fetch")
        self.assertLess(len(tool["description"]), 130)
        self.assertLess(len(json.dumps(tool["parameters"])),
                        len(json.dumps(payload["tools"][0]["parameters"])))
        # big function_call_output compressed + recovery handle
        outs = [i for i in sent["input"] if i.get("type") == "function_call_output"]
        self.assertEqual(len(outs), 1)
        self.assertLess(len(outs[0]["output"]), len(big_json))
        self.assertIn("[ccr:", outs[0]["output"])
        # function_call + small messages untouched
        self.assertIn({"type": "function_call", "call_id": "c1", "name": "fetch",
                       "arguments": "{}"}, sent["input"])
        # stats recorded for responses path
        entry = proxy.REQUEST_STATS[-1]
        self.assertEqual(entry["path"], "v1/responses")
        self.assertGreater(entry["chars_saved"], 0)
        self.assertEqual(entry["upstream_prompt_tokens"], 42)

    def test_responses_api_reasoning_never_touched(self):
        reasoning = {"type": "reasoning", "summary": [{"type": "summary_text",
                                                      "text": "R" * 2000}]}
        payload = {"model": "test", "input": [reasoning]}
        upstream = _FakeUpstream(_resp(b'{"usage": {"input_tokens": 1}}'))
        with TestClient(proxy.app) as client:
            proxy._http = upstream
            r = client.post("/v1/responses", json=payload)
        self.assertEqual(r.status_code, 200)
        sent = json.loads(upstream.last["content"])
        self.assertEqual(sent["input"][0], reasoning)

    def test_responses_api_string_input_compressed(self):
        payload = {"model": "test", "input": "hello world " * 300}
        upstream = _FakeUpstream(_resp(b'{"usage": {"input_tokens": 1}}'))
        with TestClient(proxy.app) as client:
            proxy._http = upstream
            r = client.post("/v1/responses", json=payload)
        self.assertEqual(r.status_code, 200)
        sent = json.loads(upstream.last["content"])
        self.assertLess(len(sent["input"]), len(payload["input"]))
        self.assertIn("[ccr:", sent["input"])


    def test_query_aware_compression(self):
        """Latest user message threads through as query anchor for tool results."""
        from proxy import _extract_query
        payload = {"messages": [
            {"role": "user", "content": "Find the retry logic in worker.py"},
            {"role": "tool", "content": "\n".join(f"noise {i}" for i in range(50)) +
                                       "\nretry logic found at line 42" +
                                       "\n".join(f"noise2 {i}" for i in range(50))},
        ]}
        self.assertIn("retry logic", _extract_query(payload) or "")
        self.assertIsNone(_extract_query({"messages": [{"role": "user", "content": "hi"}]}))

if __name__ == "__main__":
    unittest.main()