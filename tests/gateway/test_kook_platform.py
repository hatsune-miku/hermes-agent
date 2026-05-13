import asyncio
import inspect
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType


class _Context:
    def __init__(self):
        self.kwargs = None
        self.tools = []

    def register_platform(self, **kwargs):
        self.kwargs = kwargs

    def register_tool(self, **kwargs):
        self.tools.append(kwargs)


def _install_fake_khl(monkeypatch):
    fake = types.ModuleType("khl")

    class MessageTypes:
        TEXT = 1
        IMG = 2
        FILE = 4
        KMD = 9

    class ChannelPrivacyTypes:
        GROUP = "GROUP"
        PERSON = "PERSON"

    class _MessageAPI:
        @staticmethod
        def update(**kwargs):
            return ("message.update", kwargs)

    class _DirectMessageAPI:
        @staticmethod
        def update(**kwargs):
            return ("direct-message.update", kwargs)

    class Bot:
        def __init__(self, token=""):
            self.token = token
            self.client = SimpleNamespace(
                fetch_me=AsyncMock(return_value=SimpleNamespace(id="bot-1")),
                fetch_public_channel=AsyncMock(),
                fetch_user=AsyncMock(),
                create_asset=AsyncMock(),
                gate=SimpleNamespace(exec_req=AsyncMock(return_value={})),
            )
            self._message_handler = None

        def on_message(self, *args, **kwargs):
            def decorator(func):
                # Replicate khl.py client.register validation:
                # handler must have exactly one param with a RawMessage subclass annotation
                params = list(inspect.signature(func).parameters.values())
                if len(params) != 1 or not (
                    params[0].annotation is not inspect.Parameter.empty
                    and isinstance(params[0].annotation, type)
                    and issubclass(params[0].annotation, fake.RawMessage)
                ):
                    raise TypeError(
                        "handler must have one and only one param, "
                        "and the param inherits RawMessage"
                    )
                self._message_handler = func
                return func
            return decorator

        async def start(self):
            await asyncio.Event().wait()

        async def close(self):
            return None

    class RawMessage:
        pass

    class Message(RawMessage):
        pass

    fake.api = SimpleNamespace(Message=_MessageAPI, DirectMessage=_DirectMessageAPI)
    fake.RawMessage = RawMessage
    fake.Message = Message
    fake.Bot = Bot
    fake.MessageTypes = MessageTypes
    fake.ChannelPrivacyTypes = ChannelPrivacyTypes
    monkeypatch.setitem(sys.modules, "khl", fake)
    return fake


def _config(**extra):
    cfg = PlatformConfig(enabled=True)
    cfg.extra = extra
    return cfg


def test_register_exposes_kook_platform_metadata():
    from plugins.platforms.kook import register

    ctx = _Context()
    register(ctx)

    assert ctx.kwargs["name"] == "kook"
    assert ctx.kwargs["label"] == "KOOK"
    assert ctx.kwargs["cron_deliver_env_var"] == "KOOK_HOME_CHANNEL"
    assert ctx.kwargs["allowed_users_env"] == "KOOK_ALLOWED_USERS"
    assert ctx.kwargs["allow_all_env"] == "KOOK_ALLOW_ALL_USERS"
    assert ctx.kwargs["max_message_length"] == 2000
    assert 'hermes-agent[kook]' in ctx.kwargs["install_hint"]


def test_register_exposes_kook_raw_request_tool():
    from plugins.platforms.kook import register

    ctx = _Context()
    register(ctx)

    tool_names = [tool["name"] for tool in ctx.tools]
    assert "kook_raw_request" in tool_names
    assert "send_file_behind_link" in tool_names
    raw = next(t for t in ctx.tools if t["name"] == "kook_raw_request")
    send_file = next(t for t in ctx.tools if t["name"] == "send_file_behind_link")
    assert raw["toolset"] == "kook"
    assert send_file["toolset"] == "kook"
    assert callable(send_file["handler"])
    assert callable(send_file["check_fn"])
    assert send_file["schema"]["name"] == "send_file_behind_link"


def test_env_enablement_reads_token_and_home_channel(monkeypatch):
    from plugins.platforms.kook.adapter import _env_enablement

    monkeypatch.setenv("KOOK_TOKEN", "token-123")
    monkeypatch.setenv("KOOK_HOME_CHANNEL", "channel-456")

    assert _env_enablement() == {
        "token": "token-123",
        "home_channel": {"chat_id": "channel-456", "name": "channel-456"},
    }


def test_validate_config_accepts_env_or_extra_token(monkeypatch):
    from plugins.platforms.kook.adapter import validate_config

    monkeypatch.delenv("KOOK_TOKEN", raising=False)
    assert validate_config(_config()) is False
    assert validate_config(_config(token="from-extra")) is True
    monkeypatch.setenv("KOOK_TOKEN", "from-env")
    assert validate_config(_config()) is True


def test_connect_fetches_bot_identity_and_ignores_self_messages(monkeypatch):
    _install_fake_khl(monkeypatch)
    from khl import ChannelPrivacyTypes
    from plugins.platforms.kook.adapter import KookAdapter

    async def _run():
        adapter = KookAdapter(_config(token="token-123"))
        captured = []

        async def capture(event):
            captured.append(event)

        adapter.handle_message = capture
        assert await adapter.connect() is True
        assert adapter._bot_user_id == "bot-1"

        msg = SimpleNamespace(
            content="self echo",
            channel_type=ChannelPrivacyTypes.GROUP,
            msg_id="message-self",
            target_id="channel-1",
            author_id="bot-1",
            author=SimpleNamespace(id="bot-1", username="bot", bot=False),
            ctx=SimpleNamespace(channel=SimpleNamespace(id="channel-1", name="general"), guild=None),
        )
        await adapter._on_kook_message(msg)
        await adapter.disconnect()

        assert captured == []

    asyncio.run(_run())


def test_send_uses_kmarkdown_channel_send(monkeypatch):
    _install_fake_khl(monkeypatch)
    from khl import MessageTypes
    from plugins.platforms.kook.adapter import KookAdapter

    async def _run():
        adapter = KookAdapter(_config(token="token-123"))
        channel = SimpleNamespace(send=AsyncMock(return_value={"msg_id": "msg-1"}))
        adapter._bot = SimpleNamespace(client=SimpleNamespace(fetch_public_channel=AsyncMock(return_value=channel)))

        result = await adapter.send("channel-1", "hello")

        assert result.success is True
        assert result.message_id == "msg-1"
        channel.send.assert_awaited_once_with("hello", type=MessageTypes.KMD)

    asyncio.run(_run())


def test_send_dm_uses_fetch_user_and_user_send(monkeypatch):
    _install_fake_khl(monkeypatch)
    from khl import MessageTypes
    from plugins.platforms.kook.adapter import KookAdapter

    async def _run():
        adapter = KookAdapter(_config(token="token-123"))
        user_obj = SimpleNamespace(send=AsyncMock(return_value={"msg_id": "dm-msg-1"}))
        adapter._bot = SimpleNamespace(
            client=SimpleNamespace(fetch_user=AsyncMock(return_value=user_obj), create_asset=AsyncMock())
        )

        result = await adapter.send("user-99", "hello DM", metadata={"chat_type": "dm"})

        assert result.success is True
        assert result.message_id == "dm-msg-1"
        adapter._bot.client.fetch_user.assert_awaited_once_with("user-99")
        user_obj.send.assert_awaited_once_with("hello DM", type=MessageTypes.KMD)

    asyncio.run(_run())


def test_group_message_ignores_messages_without_bot_mention(monkeypatch):
    _install_fake_khl(monkeypatch)
    from khl import ChannelPrivacyTypes
    from plugins.platforms.kook.adapter import KookAdapter

    async def _run():
        adapter = KookAdapter(_config(token="token-123"))
        adapter._bot_user_id = "bot-1"
        captured = []

        async def capture(event):
            captured.append(event)

        adapter.handle_message = capture
        msg = SimpleNamespace(
            content="hello from kook",
            mention=[],
            channel_type=ChannelPrivacyTypes.GROUP,
            msg_id="message-1",
            target_id="channel-1",
            author_id="user-1",
            author=SimpleNamespace(id="user-1", username="alice", nickname="Alice", bot=False),
            ctx=SimpleNamespace(channel=SimpleNamespace(id="channel-1", name="general"), guild=None),
        )

        await adapter._on_kook_message(msg)

        assert captured == []

    asyncio.run(_run())


def test_group_message_dispatches_when_bot_is_mentioned(monkeypatch):
    _install_fake_khl(monkeypatch)
    from khl import ChannelPrivacyTypes
    from plugins.platforms.kook.adapter import KookAdapter

    async def _run():
        adapter = KookAdapter(_config(token="token-123"))
        adapter._bot_user_id = "bot-1"
        captured = []

        async def capture(event):
            captured.append(event)

        adapter.handle_message = capture
        msg = SimpleNamespace(
            content="hello from kook",
            mention=["bot-1"],
            channel_type=ChannelPrivacyTypes.GROUP,
            msg_id="message-1",
            target_id="channel-1",
            author_id="user-1",
            author=SimpleNamespace(id="user-1", username="alice", nickname="Alice", bot=False),
            ctx=SimpleNamespace(
                channel=SimpleNamespace(id="channel-1", name="general"),
                guild=SimpleNamespace(id="guild-1"),
            ),
        )

        await adapter._on_kook_message(msg)

        assert len(captured) == 1
        event = captured[0]
        assert event.text == "hello from kook"
        assert event.message_type == MessageType.TEXT
        assert event.message_id == "message-1"
        assert event.source.platform.value == "kook"
        assert event.source.chat_id == "channel-1"
        assert event.source.chat_name == "general"
        assert event.source.chat_type == "group"
        assert event.source.user_id == "user-1"
        assert event.source.user_name == "Alice"
        assert event.source.guild_id == "guild-1"

    asyncio.run(_run())


def test_private_message_uses_author_id_as_chat_id(monkeypatch):
    _install_fake_khl(monkeypatch)
    from khl import ChannelPrivacyTypes
    from plugins.platforms.kook.adapter import KookAdapter

    async def _run():
        adapter = KookAdapter(_config(token="token-123"))
        captured = []

        async def capture(event):
            captured.append(event)

        adapter.handle_message = capture
        msg = SimpleNamespace(
            content="dm hello",
            channel_type=ChannelPrivacyTypes.PERSON,
            msg_id="message-2",
            target_id="private-chat-code",
            author_id="user-2",
            author=SimpleNamespace(id="user-2", username="bob", bot=False),
            ctx=SimpleNamespace(channel=SimpleNamespace(id="private-chat-code", name="bob"), guild=None),
        )

        await adapter._on_kook_message(msg)

        assert len(captured) == 1
        assert captured[0].source.chat_id == "user-2"
        assert captured[0].source.chat_type == "dm"
        assert captured[0].source.user_name == "bob"
        assert "user-2" in adapter._dm_chat_ids

    asyncio.run(_run())


def test_reply_to_inbound_dm_uses_cached_chat_type(monkeypatch):
    _install_fake_khl(monkeypatch)
    from khl import ChannelPrivacyTypes, MessageTypes
    from plugins.platforms.kook.adapter import KookAdapter

    async def _run():
        adapter = KookAdapter(_config(token="token-123"))
        captured = []

        async def capture(event):
            captured.append(event)

        adapter.handle_message = capture
        msg = SimpleNamespace(
            content="dm hello",
            channel_type=ChannelPrivacyTypes.PERSON,
            msg_id="message-2",
            target_id="private-chat-code",
            author_id="user-2",
            author=SimpleNamespace(id="user-2", username="bob", bot=False),
            ctx=SimpleNamespace(channel=SimpleNamespace(id="private-chat-code", name="bob"), guild=None),
        )
        await adapter._on_kook_message(msg)

        user_obj = SimpleNamespace(send=AsyncMock(return_value={"msg_id": "dm-reply-1"}))
        adapter._bot = SimpleNamespace(
            client=SimpleNamespace(fetch_user=AsyncMock(return_value=user_obj), fetch_public_channel=AsyncMock())
        )

        result = await adapter.send("user-2", "reply without metadata")

        assert result.success is True
        assert result.message_id == "dm-reply-1"
        adapter._bot.client.fetch_user.assert_awaited_once_with("user-2")
        adapter._bot.client.fetch_public_channel.assert_not_called()
        user_obj.send.assert_awaited_once_with("reply without metadata", type=MessageTypes.KMD)

    asyncio.run(_run())


def test_get_chat_info_reports_cached_dm_without_fetching_channel(monkeypatch):
    _install_fake_khl(monkeypatch)
    from plugins.platforms.kook.adapter import KookAdapter

    async def _run():
        adapter = KookAdapter(_config(token="token-123"))
        adapter._dm_chat_ids.add("user-2")
        adapter._bot = SimpleNamespace(client=SimpleNamespace(fetch_public_channel=AsyncMock()))

        info = await adapter.get_chat_info("user-2")

        assert info == {"name": "user-2", "type": "dm", "chat_id": "user-2"}
        adapter._bot.client.fetch_public_channel.assert_not_called()

    asyncio.run(_run())


def test_send_image_file_uploads_asset_and_preserves_dm_metadata(monkeypatch):
    _install_fake_khl(monkeypatch)
    from khl import MessageTypes
    from plugins.platforms.kook.adapter import KookAdapter

    async def _run():
        adapter = KookAdapter(_config(token="token-123"))
        user_obj = SimpleNamespace(send=AsyncMock(return_value={"msg_id": "img-msg-1"}))
        adapter._bot = SimpleNamespace(
            client=SimpleNamespace(
                fetch_user=AsyncMock(return_value=user_obj),
                fetch_public_channel=AsyncMock(),
                create_asset=AsyncMock(return_value="https://cdn.kook.example/image.png"),
            )
        )

        result = await adapter.send_image_file("user-2", "local.png", metadata={"chat_type": "dm"})

        assert result.success is True
        assert result.message_id == "img-msg-1"
        adapter._bot.client.create_asset.assert_awaited_once_with(Path("local.png"))
        adapter._bot.client.fetch_user.assert_awaited_once_with("user-2")
        adapter._bot.client.fetch_public_channel.assert_not_called()
        user_obj.send.assert_awaited_once_with("https://cdn.kook.example/image.png", type=MessageTypes.IMG)

    asyncio.run(_run())


def test_send_remote_image_skips_upload_and_sends_caption(monkeypatch):
    _install_fake_khl(monkeypatch)
    from khl import MessageTypes
    from plugins.platforms.kook.adapter import KookAdapter

    async def _run():
        adapter = KookAdapter(_config(token="token-123"))
        channel = SimpleNamespace(send=AsyncMock(return_value={"msg_id": "img-msg-2"}))
        adapter._bot = SimpleNamespace(
            client=SimpleNamespace(fetch_public_channel=AsyncMock(return_value=channel), create_asset=AsyncMock())
        )

        result = await adapter.send_image("channel-1", "https://example.com/image.png", caption="caption")

        assert result.success is True
        assert result.message_id == "img-msg-2"
        adapter._bot.client.create_asset.assert_not_called()
        channel.send.assert_any_await("https://example.com/image.png", type=MessageTypes.IMG)
        channel.send.assert_any_await("caption", type=MessageTypes.KMD)

    asyncio.run(_run())


def test_send_document_uploads_asset(monkeypatch):
    _install_fake_khl(monkeypatch)
    from khl import MessageTypes
    from plugins.platforms.kook.adapter import KookAdapter

    async def _run():
        adapter = KookAdapter(_config(token="token-123"))
        channel = SimpleNamespace(send=AsyncMock(return_value={"msg_id": "file-msg-1"}))
        adapter._bot = SimpleNamespace(
            client=SimpleNamespace(
                fetch_public_channel=AsyncMock(return_value=channel),
                create_asset=AsyncMock(return_value="https://cdn.kook.example/file.txt"),
            )
        )

        result = await adapter.send_document("channel-1", "report.txt", caption="report")

        assert result.success is True
        assert result.message_id == "file-msg-1"
        adapter._bot.client.create_asset.assert_awaited_once_with(Path("report.txt"))
        channel.send.assert_any_await("https://cdn.kook.example/file.txt", type=MessageTypes.FILE)
        channel.send.assert_any_await("report", type=MessageTypes.KMD)

    asyncio.run(_run())


# ── Ambient group context ──────────────────────────────────────────────────────

def test_non_mention_group_message_stored_in_ambient_history(monkeypatch):
    _install_fake_khl(monkeypatch)
    from khl import ChannelPrivacyTypes
    from plugins.platforms.kook.adapter import KookAdapter

    async def _run():
        adapter = KookAdapter(_config(token="token-123"))
        adapter._bot_user_id = "bot-1"
        captured = []

        async def capture(event):
            captured.append(event)

        adapter.handle_message = capture
        msg = SimpleNamespace(
            content="just chatting",
            mention=[],
            channel_type=ChannelPrivacyTypes.GROUP,
            msg_id="ambient-1",
            target_id="channel-1",
            author_id="user-1",
            author=SimpleNamespace(id="user-1", username="alice", nickname="Alice", bot=False),
            ctx=SimpleNamespace(channel=SimpleNamespace(id="channel-1", name="general"), guild=None),
        )
        await adapter._on_kook_message(msg)

        assert captured == []
        assert "channel-1" in adapter._group_history
        entries = list(adapter._group_history["channel-1"])
        assert len(entries) == 1
        assert entries[0]["user"] == "Alice"
        assert entries[0]["text"] == "just chatting"

    asyncio.run(_run())


def test_mention_message_prepends_ambient_history_then_clears(monkeypatch):
    _install_fake_khl(monkeypatch)
    from khl import ChannelPrivacyTypes
    from plugins.platforms.kook.adapter import KookAdapter

    async def _run():
        adapter = KookAdapter(_config(token="token-123"))
        adapter._bot_user_id = "bot-1"
        captured = []

        async def capture(event):
            captured.append(event)

        adapter.handle_message = capture

        for i in range(2):
            ambient = SimpleNamespace(
                content=f"ambient message {i}",
                mention=[],
                channel_type=ChannelPrivacyTypes.GROUP,
                msg_id=f"ambient-{i}",
                target_id="channel-1",
                author_id="user-1",
                author=SimpleNamespace(id="user-1", username="alice", nickname="Alice", bot=False),
                ctx=SimpleNamespace(channel=SimpleNamespace(id="channel-1", name="general"), guild=None),
            )
            await adapter._on_kook_message(ambient)

        mention_msg = SimpleNamespace(
            content="hey bot, summarise",
            mention=["bot-1"],
            channel_type=ChannelPrivacyTypes.GROUP,
            msg_id="mention-1",
            target_id="channel-1",
            author_id="user-2",
            author=SimpleNamespace(id="user-2", username="bob", nickname="Bob", bot=False),
            ctx=SimpleNamespace(
                channel=SimpleNamespace(id="channel-1", name="general"),
                guild=SimpleNamespace(id="guild-1"),
            ),
        )
        await adapter._on_kook_message(mention_msg)

        assert len(captured) == 1
        event_text = captured[0].text
        assert "ambient message 0" in event_text
        assert "ambient message 1" in event_text
        assert "hey bot, summarise" in event_text
        assert len(adapter._group_history.get("channel-1", [])) == 0

    asyncio.run(_run())


def test_ambient_history_capped_at_max_size(monkeypatch):
    _install_fake_khl(monkeypatch)
    from khl import ChannelPrivacyTypes
    from plugins.platforms.kook.adapter import KookAdapter

    async def _run():
        adapter = KookAdapter(_config(token="token-123"))
        adapter._bot_user_id = "bot-1"
        adapter.handle_message = AsyncMock()

        for i in range(30):
            msg = SimpleNamespace(
                content=f"msg {i}",
                mention=[],
                channel_type=ChannelPrivacyTypes.GROUP,
                msg_id=f"m-{i}",
                target_id="channel-1",
                author_id="user-1",
                author=SimpleNamespace(id="user-1", username="alice", nickname=None, bot=False),
                ctx=SimpleNamespace(channel=SimpleNamespace(id="channel-1", name="general"), guild=None),
            )
            await adapter._on_kook_message(msg)

        assert len(adapter._group_history["channel-1"]) <= 20

    asyncio.run(_run())


def test_edit_message_updates_channel_message(monkeypatch):
    _install_fake_khl(monkeypatch)
    from plugins.platforms.kook.adapter import KookAdapter

    async def _run():
        adapter = KookAdapter(_config(token="token-123"))
        gate = SimpleNamespace(exec_req=AsyncMock(return_value={}))
        adapter._bot = SimpleNamespace(client=SimpleNamespace(gate=gate))

        result = await adapter.edit_message("channel-1", "msg-1", "updated")

        assert result.success is True
        assert result.message_id == "msg-1"
        gate.exec_req.assert_awaited_once_with(
            ("message.update", {"msg_id": "msg-1", "content": "updated"})
        )

    asyncio.run(_run())


def test_edit_message_uses_dm_api_for_cached_dm_chat(monkeypatch):
    _install_fake_khl(monkeypatch)
    from plugins.platforms.kook.adapter import KookAdapter

    async def _run():
        adapter = KookAdapter(_config(token="token-123"))
        adapter._dm_chat_ids.add("user-2")
        gate = SimpleNamespace(exec_req=AsyncMock(return_value={}))
        adapter._bot = SimpleNamespace(client=SimpleNamespace(gate=gate))

        result = await adapter.edit_message("user-2", "dm-msg-1", "updated dm")

        assert result.success is True
        assert result.message_id == "dm-msg-1"
        gate.exec_req.assert_awaited_once_with(
            ("direct-message.update", {"msg_id": "dm-msg-1", "content": "updated dm"})
        )

    asyncio.run(_run())


def test_edit_message_marks_rate_limit_errors_retryable(monkeypatch):
    _install_fake_khl(monkeypatch)
    from plugins.platforms.kook.adapter import KookAdapter

    async def _run():
        adapter = KookAdapter(_config(token="token-123"))
        gate = SimpleNamespace(exec_req=AsyncMock(side_effect=Exception("429 rate limited")))
        adapter._bot = SimpleNamespace(client=SimpleNamespace(gate=gate))

        result = await adapter.edit_message("channel-1", "msg-1", "updated")

        assert result.success is False
        assert result.retryable is True
        assert "429" in result.error

    asyncio.run(_run())
