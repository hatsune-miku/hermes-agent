import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock


def _install_fake_khl(monkeypatch, *, channel_send=None, user_send=None, upload_url="https://cdn.kook.example/x.png"):
    fake = types.ModuleType("khl")

    class MessageTypes:
        TEXT = 1
        IMG = 2
        FILE = 4
        KMD = 9

    channel = SimpleNamespace(send=channel_send or AsyncMock(return_value={"msg_id": "ch-msg"}))
    user = SimpleNamespace(send=user_send or AsyncMock(return_value={"msg_id": "dm-msg"}))

    class Bot:
        def __init__(self, token=""):
            self.token = token
            self.client = SimpleNamespace(
                gate=SimpleNamespace(request=AsyncMock(), requester=SimpleNamespace(_cs=SimpleNamespace(close=AsyncMock()))),
                create_asset=AsyncMock(return_value=upload_url),
                fetch_public_channel=AsyncMock(return_value=channel),
                fetch_user=AsyncMock(return_value=user),
            )

    fake.Bot = Bot
    fake.MessageTypes = MessageTypes
    monkeypatch.setitem(sys.modules, "khl", fake)
    return fake, channel, user


def test_send_file_behind_link_downloads_and_sends_to_channel(monkeypatch, tmp_path):
    fake, channel, _ = _install_fake_khl(monkeypatch)
    monkeypatch.setenv("KOOK_TOKEN", "token-123")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "channel-1")
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "kook")

    downloads = {}

    def fake_download(url, dest):
        downloads["url"] = url
        downloads["dest"] = dest
        dest.write_bytes(b"file-bytes")

    monkeypatch.setattr("plugins.platforms.kook.tools._download_to_path", fake_download)
    monkeypatch.setattr("plugins.platforms.kook.tools._scratch_dir", lambda: tmp_path)

    from plugins.platforms.kook.tools import handle_send_file_behind_link

    result = json.loads(handle_send_file_behind_link({"url": "https://example.com/x.png", "file_name": "renamed.png"}))

    assert result == {"msg_id": "ch-msg"}
    assert downloads["url"] == "https://example.com/x.png"
    assert downloads["dest"].name == "renamed.png"
    channel.send.assert_awaited_once_with("https://cdn.kook.example/x.png", type=fake.MessageTypes.IMG)


def test_send_file_behind_link_uses_dm_when_dm_chat(monkeypatch, tmp_path):
    fake, channel, user = _install_fake_khl(monkeypatch)
    monkeypatch.setenv("KOOK_TOKEN", "token-123")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "user-2")
    monkeypatch.setenv("HERMES_SESSION_USER_ID", "user-2")
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "kook")

    monkeypatch.setattr("plugins.platforms.kook.tools._download_to_path", lambda url, dest: dest.write_bytes(b"x"))
    monkeypatch.setattr("plugins.platforms.kook.tools._scratch_dir", lambda: tmp_path)

    from plugins.platforms.kook.tools import handle_send_file_behind_link

    result = json.loads(handle_send_file_behind_link({"url": "https://example.com/x.png", "file_name": "name.png"}))

    assert result == {"msg_id": "dm-msg"}
    channel.send.assert_not_called()
    user.send.assert_awaited_once_with("https://cdn.kook.example/x.png", type=fake.MessageTypes.IMG)


def test_send_file_behind_link_requires_chat_id(monkeypatch, tmp_path):
    _install_fake_khl(monkeypatch)
    monkeypatch.setenv("KOOK_TOKEN", "token-123")
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
    monkeypatch.setattr("plugins.platforms.kook.tools._download_to_path", lambda *a, **k: None)
    monkeypatch.setattr("plugins.platforms.kook.tools._scratch_dir", lambda: tmp_path)

    from plugins.platforms.kook.tools import handle_send_file_behind_link

    result = json.loads(handle_send_file_behind_link({"url": "https://example.com/x.png", "file_name": "x.png"}))

    assert "error" in result
    assert "chat" in result["error"].lower()


def test_send_file_behind_link_rejects_non_http(monkeypatch, tmp_path):
    _install_fake_khl(monkeypatch)
    monkeypatch.setenv("KOOK_TOKEN", "token-123")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "channel-1")
    monkeypatch.setattr("plugins.platforms.kook.tools._download_to_path", lambda *a, **k: None)
    monkeypatch.setattr("plugins.platforms.kook.tools._scratch_dir", lambda: tmp_path)

    from plugins.platforms.kook.tools import handle_send_file_behind_link

    result = json.loads(handle_send_file_behind_link({"url": "ftp://nope/x", "file_name": "x.png"}))

    assert "error" in result
    assert "http" in result["error"].lower()


def test_send_file_behind_link_closes_session_after_send(monkeypatch, tmp_path):
    fake, _, _ = _install_fake_khl(monkeypatch)
    monkeypatch.setenv("KOOK_TOKEN", "token-123")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "channel-1")
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "kook")

    cs_close = AsyncMock()
    requester = SimpleNamespace(_cs=SimpleNamespace(close=cs_close))
    original_bot = fake.Bot

    class BotWithRequester(original_bot):
        def __init__(self, token=""):
            super().__init__(token=token)
            self.client.gate = SimpleNamespace(request_endpoint=AsyncMock(), requester=requester)

    fake.Bot = BotWithRequester
    monkeypatch.setattr("plugins.platforms.kook.tools._download_to_path", lambda url, dest: dest.write_bytes(b"x"))
    monkeypatch.setattr("plugins.platforms.kook.tools._scratch_dir", lambda: tmp_path)

    from plugins.platforms.kook.tools import handle_send_file_behind_link

    handle_send_file_behind_link({"url": "https://example.com/x.png", "file_name": "x.png"})

    cs_close.assert_awaited_once()


def test_send_file_behind_link_closes_session_even_on_failure(monkeypatch, tmp_path):
    fake, _, _ = _install_fake_khl(monkeypatch)
    monkeypatch.setenv("KOOK_TOKEN", "token-123")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "channel-1")

    cs_close = AsyncMock()
    requester = SimpleNamespace(_cs=SimpleNamespace(close=cs_close))
    original_bot = fake.Bot

    class BotWithRequester(original_bot):
        def __init__(self, token=""):
            super().__init__(token=token)
            self.client.gate = SimpleNamespace(request_endpoint=AsyncMock(), requester=requester)
            self.client.create_asset = AsyncMock(side_effect=RuntimeError("upload failed"))

    fake.Bot = BotWithRequester
    monkeypatch.setattr("plugins.platforms.kook.tools._download_to_path", lambda url, dest: dest.write_bytes(b"x"))
    monkeypatch.setattr("plugins.platforms.kook.tools._scratch_dir", lambda: tmp_path)

    from plugins.platforms.kook.tools import handle_send_file_behind_link

    result = json.loads(handle_send_file_behind_link({"url": "https://example.com/x.png", "file_name": "x.png"}))

    assert "error" in result
    cs_close.assert_awaited_once()
