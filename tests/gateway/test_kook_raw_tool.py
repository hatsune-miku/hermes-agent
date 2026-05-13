import asyncio
import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock


def _install_fake_khl(monkeypatch):
    fake = types.ModuleType("khl")

    class Bot:
        def __init__(self, token=""):
            self.token = token
            self.client = SimpleNamespace(gate=SimpleNamespace(request=AsyncMock(return_value={"ok": True})))

    fake.Bot = Bot
    monkeypatch.setitem(sys.modules, "khl", fake)
    return fake


def test_kook_raw_request_get_uses_query_params(monkeypatch):
    fake = _install_fake_khl(monkeypatch)
    monkeypatch.setenv("KOOK_TOKEN", "token-123")
    from plugins.platforms.kook.tools import handle_kook_raw_request

    result = json.loads(handle_kook_raw_request("GET", "/api/v3/user/me", '{"page": 1}'))

    assert result == {"ok": True}
    bot = fake.Bot(token="token-123")
    assert bot.token == "token-123"


def test_kook_raw_request_get_calls_gate_with_params(monkeypatch):
    fake = _install_fake_khl(monkeypatch)
    monkeypatch.setenv("KOOK_TOKEN", "token-123")
    created = []

    class Bot:
        def __init__(self, token=""):
            self.token = token
            self.client = SimpleNamespace(gate=SimpleNamespace(request=AsyncMock(return_value={"data": []})))
            created.append(self)

    fake.Bot = Bot
    from plugins.platforms.kook.tools import handle_kook_raw_request

    result = json.loads(handle_kook_raw_request("GET", "/api/v3/guild/list", '{"page": 2}'))

    assert result == {"data": []}
    created[0].client.gate.request.assert_awaited_once_with("GET", "guild/list", params={"page": 2})


def test_kook_raw_request_post_uses_json_body(monkeypatch):
    fake = _install_fake_khl(monkeypatch)
    monkeypatch.setenv("KOOK_TOKEN", "token-123")
    created = []

    class Bot:
        def __init__(self, token=""):
            self.client = SimpleNamespace(gate=SimpleNamespace(request=AsyncMock(return_value={"msg_id": "m1"})))
            created.append(self)

    fake.Bot = Bot
    from plugins.platforms.kook.tools import handle_kook_raw_request

    result = json.loads(handle_kook_raw_request("POST", "/api/v3/message/create", '{"content": "hi"}'))

    assert result == {"msg_id": "m1"}
    created[0].client.gate.request.assert_awaited_once_with(
        "POST", "message/create", json={"content": "hi"}
    )


def test_kook_raw_request_rejects_non_api_v3_endpoint(monkeypatch):
    _install_fake_khl(monkeypatch)
    monkeypatch.setenv("KOOK_TOKEN", "token-123")
    from plugins.platforms.kook.tools import handle_kook_raw_request

    result = json.loads(handle_kook_raw_request("GET", "https://evil.example/api/v3/user/me", "{}"))

    assert "error" in result
    assert "/api/v3/" in result["error"]


def test_kook_raw_request_rejects_invalid_json_body(monkeypatch):
    _install_fake_khl(monkeypatch)
    monkeypatch.setenv("KOOK_TOKEN", "token-123")
    from plugins.platforms.kook.tools import handle_kook_raw_request

    result = json.loads(handle_kook_raw_request("GET", "/api/v3/user/me", "not-json"))

    assert "error" in result
    assert "valid JSON" in result["error"]
