import asyncio
import importlib

import httpx
import pytest

api = importlib.import_module("backend.app")


@pytest.mark.parametrize("failure", ["connection", "503", "401", "quota"])
def test_openai_retries_transient_failures_only(monkeypatch, failure):
    monkeypatch.setenv("ULCA_LLM_PROVIDER", "openai")
    monkeypatch.setattr(api, "get_openai_credentials", lambda: ("private-test-key", "test-model"))
    calls = []
    original_client = httpx.AsyncClient

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            if failure == "connection":
                raise httpx.ConnectError("private diagnostic", request=request)
            status = 503 if failure == "503" else 401 if failure == "401" else 429
            return httpx.Response(status, json={"error": {"code": "insufficient_quota" if failure == "quota" else "test"}})
        return httpx.Response(200, json={"output": [{"content": [{"type": "output_text", "text": "ok"}]}]})

    monkeypatch.setattr(api.httpx, "AsyncClient", lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs))
    if failure in ("401", "quota"):
        with pytest.raises(httpx.HTTPStatusError) as caught:
            asyncio.run(api.complete("test", []))
        assert len(calls) == 1
        reason = api.llm_failure_reason(caught.value)
        assert "private" not in reason
        assert "429" in reason if failure == "quota" else "401" in reason
    else:
        assert asyncio.run(api.complete("test", [])) == "ok"
        assert len(calls) == 2
