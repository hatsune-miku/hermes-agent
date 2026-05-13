import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock


def _install_fake_khl(monkeypatch, *, return_value=None):
    fake = types.ModuleType("khl")

    class Bot:
        def __init__(self, token=""):
            self.token = token
            self.client = SimpleNamespace(
                gate=SimpleNamespace(
                    request_endpoint=AsyncMock(return_value=return_value if return_value is not None else {"ok": True})
                )
            )

    fake.Bot = Bot
    monkeypatch.setitem(sys.modules, "khl", fake)
    return fake


def test_kook_raw_request_get_passes_body_as_params(monkeypatch):
    fake = _install_fake_khl(monkeypatch, return_value={"data": []})
    monkeypatch.setenv("KOOK_TOKEN", "token-123")
    created = []

    original_bot = fake.Bot

    class CapturingBot(original_bot):
        def __init__(self, token=""):
            super().__init__(token=token)
            created.append(self)

    fake.Bot = CapturingBot

    from plugins.platforms.kook.tools import handle_kook_raw_request

    result = json.loads(handle_kook_raw_request({"method": "GET", "endpoint": "/api/v3/guild/list", "body": '{"page": 2}'}))

    assert result == {"data": []}
    created[0].client.gate.request_endpoint.assert_awaited_once_with(
        "GET", "/api/v3/guild/list", params={"page": 2}
    )


def test_kook_raw_request_post_passes_body_as_json(monkeypatch):
    fake = _install_fake_khl(monkeypatch, return_value={"msg_id": "m1"})
    monkeypatch.setenv("KOOK_TOKEN", "token-123")
    created = []

    original_bot = fake.Bot

    class CapturingBot(original_bot):
        def __init__(self, token=""):
            super().__init__(token=token)
            created.append(self)

    fake.Bot = CapturingBot

    from plugins.platforms.kook.tools import handle_kook_raw_request

    result = json.loads(handle_kook_raw_request({"method": "POST", "endpoint": "/api/v3/message/create", "body": '{"content": "hi"}'}))

    assert result == {"msg_id": "m1"}
    created[0].client.gate.request_endpoint.assert_awaited_once_with(
        "POST", "/api/v3/message/create", json={"content": "hi"}
    )


def test_kook_raw_request_rejects_unknown_method(monkeypatch):
    _install_fake_khl(monkeypatch)
    monkeypatch.setenv("KOOK_TOKEN", "token-123")
    from plugins.platforms.kook.tools import handle_kook_raw_request

    result = json.loads(handle_kook_raw_request({"method": "PURGE", "endpoint": "/api/v3/user/me", "body": "{}"}))

    assert "error" in result
    assert "method" in result["error"].lower()


def test_kook_raw_request_rejects_invalid_json_body(monkeypatch):
    _install_fake_khl(monkeypatch)
    monkeypatch.setenv("KOOK_TOKEN", "token-123")
    from plugins.platforms.kook.tools import handle_kook_raw_request

    result = json.loads(handle_kook_raw_request({"method": "GET", "endpoint": "/api/v3/user/me", "body": "not-json"}))

    assert "error" in result
    assert "valid JSON" in result["error"]


def test_kook_raw_request_requires_token(monkeypatch):
    _install_fake_khl(monkeypatch)
    monkeypatch.delenv("KOOK_TOKEN", raising=False)
    from plugins.platforms.kook.tools import handle_kook_raw_request

    result = json.loads(handle_kook_raw_request({"method": "GET", "endpoint": "/api/v3/user/me", "body": "{}"}))

    assert "error" in result
    assert "KOOK_TOKEN" in result["error"]
