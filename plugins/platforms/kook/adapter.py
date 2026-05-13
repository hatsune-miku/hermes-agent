import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from gateway.config import Platform
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 2000


def _khl():
    from khl import Bot, ChannelPrivacyTypes, Message, MessageTypes

    return Bot, ChannelPrivacyTypes, Message, MessageTypes


class KookAdapter(BasePlatformAdapter):
    def __init__(self, config, **kwargs):
        super().__init__(config=config, platform=Platform("kook"))
        extra = getattr(config, "extra", {}) or {}
        self.token = (
            os.getenv("KOOK_TOKEN")
            or extra.get("token")
            or getattr(config, "token", "")
            or ""
        )
        self.max_message_length = int(
            extra.get("max_message_length") or MAX_MESSAGE_LENGTH
        )
        self._bot = None
        self._bot_task: Optional[asyncio.Task] = None
        self._bot_user_id: Optional[str] = None
        self._dm_chat_ids: set[str] = set()

    async def connect(self) -> bool:
        try:
            Bot, _, _, _ = _khl()
        except Exception:
            self._set_fatal_error(
                "missing_dependency", "khl.py is not installed", retryable=False
            )
            return False
        if not self.token:
            self._set_fatal_error(
                "missing_token", "KOOK_TOKEN is required", retryable=False
            )
            return False

        self._bot = Bot(token=self.token)
        try:
            me = await self._bot.client.fetch_me()
            self._bot_user_id = str(me.id)
        except Exception as exc:
            logger.warning("KOOK: failed to fetch bot identity: %s", exc)
        self._register_handlers()
        self._bot_task = asyncio.create_task(self._run_bot())
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        task = self._bot_task
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        bot = self._bot
        close = getattr(bot, "close", None) if bot else None
        if close:
            result = close()
            if asyncio.iscoroutine(result):
                await result
        self._bot_task = None
        self._bot = None
        self._mark_disconnected()

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        if not self._bot:
            return SendResult(success=False, error="KOOK bot is not connected")
        try:
            target = await self._resolve_send_target(chat_id, metadata)
            _, _, _, MessageTypes = _khl()
            message = await target.send(content, type=MessageTypes.KMD)
            return SendResult(
                success=True, message_id=self._extract_message_id(message)
            )
        except Exception as exc:
            logger.warning("KOOK: failed to send message to %s: %s", chat_id, exc)
            return SendResult(success=False, error=str(exc))

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        if not self._bot:
            return SendResult(success=False, error="KOOK bot is not connected")
        try:
            target = await self._resolve_send_target(chat_id, metadata)
            _, _, _, MessageTypes = _khl()
            asset_url = image_url
            if not image_url.startswith(("http://", "https://")):
                asset_url = await self._bot.client.create_asset(Path(image_url))
            message = await target.send(asset_url, type=MessageTypes.IMG)
            if caption:
                await target.send(caption, type=MessageTypes.KMD)
            return SendResult(
                success=True, message_id=self._extract_message_id(message)
            )
        except Exception as exc:
            logger.warning("KOOK: failed to send image to %s: %s", chat_id, exc)
            return SendResult(success=False, error=str(exc))

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        return await self.send_image(
            chat_id=chat_id,
            image_url=image_path,
            caption=caption,
            reply_to=reply_to,
            metadata=kwargs.get("metadata"),
        )

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        if not self._bot:
            return SendResult(success=False, error="KOOK bot is not connected")
        try:
            target = await self._resolve_send_target(chat_id, kwargs.get("metadata"))
            _, _, _, MessageTypes = _khl()
            asset_url = await self._bot.client.create_asset(Path(file_path))
            message = await target.send(asset_url, type=MessageTypes.FILE)
            if caption:
                await target.send(caption, type=MessageTypes.KMD)
            return SendResult(
                success=True, message_id=self._extract_message_id(message)
            )
        except Exception as exc:
            logger.warning("KOOK: failed to send document to %s: %s", chat_id, exc)
            return SendResult(success=False, error=str(exc))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        if chat_id in self._dm_chat_ids:
            return {"name": chat_id, "type": "dm", "chat_id": chat_id}
        if not self._bot:
            return {"name": chat_id, "type": "channel", "chat_id": chat_id}
        try:
            channel = await self._bot.client.fetch_public_channel(chat_id)
            return {
                "name": getattr(channel, "name", chat_id),
                "type": "channel",
                "chat_id": chat_id,
            }
        except Exception:
            return {"name": chat_id, "type": "channel", "chat_id": chat_id}

    def _register_handlers(self) -> None:
        _, _, Message, _ = _khl()

        @self._bot.on_message()
        async def _handle_message(msg: Message) -> None:
            await self._on_kook_message(msg)

    async def _run_bot(self) -> None:
        try:
            await self._bot.start()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("KOOK: bot runtime error: %s", exc)
            self._set_fatal_error("connection_lost", str(exc), retryable=True)
            await self._notify_fatal_error()

    async def _on_kook_message(self, msg) -> None:
        author = getattr(msg, "author", None)
        if getattr(author, "bot", False):
            return
        author_id = str(
            getattr(author, "id", None) or getattr(msg, "author_id", "") or ""
        )
        if self._bot_user_id and author_id == self._bot_user_id:
            return

        text = str(getattr(msg, "content", "") or "").strip()
        if not text:
            return

        channel_type = getattr(msg, "channel_type", None)
        try:
            _, ChannelPrivacyTypes, _, _ = _khl()
            is_dm = channel_type == ChannelPrivacyTypes.PERSON
        except Exception:
            is_dm = str(channel_type).upper() == "PERSON"
        ctx = getattr(msg, "ctx", None)
        channel = getattr(ctx, "channel", None)
        guild = getattr(ctx, "guild", None)
        if not is_dm and not self._is_bot_mentioned(msg):
            return

        if is_dm:
            chat_id = author_id or str(getattr(msg, "target_id", ""))
            self._dm_chat_ids.add(chat_id)
            chat_name = _display_name(author) or chat_id
            chat_type = "dm"
            guild_id = None
        else:
            chat_id = str(getattr(channel, "id", None) or getattr(msg, "target_id", ""))
            chat_name = getattr(channel, "name", None) or chat_id
            chat_type = "group"
            guild_id = getattr(guild, "id", None)

        source = self.build_source(
            chat_id=chat_id,
            chat_name=chat_name,
            chat_type=chat_type,
            user_id=author_id,
            user_name=_display_name(author) or author_id,
            guild_id=guild_id,
            message_id=getattr(msg, "msg_id", None),
        )
        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=msg,
            message_id=str(getattr(msg, "msg_id", "") or ""),
            timestamp=datetime.now(),
        )
        await self.handle_message(event)

    def _is_bot_mentioned(self, msg) -> bool:
        if not self._bot_user_id:
            return False
        mentions = getattr(msg, "mention", None) or []
        return str(self._bot_user_id) in {str(user_id) for user_id in mentions}

    async def _resolve_send_target(
        self, chat_id: str, metadata: Optional[dict[str, Any]] = None
    ):
        metadata = metadata or {}
        target = metadata.get("target") or metadata.get("channel")
        if target is not None:
            return target
        if metadata.get("chat_type") == "dm" or chat_id in self._dm_chat_ids:
            return await self._bot.client.fetch_user(chat_id)
        return await self._bot.client.fetch_public_channel(chat_id)

    @staticmethod
    def _extract_message_id(message: Any) -> Optional[str]:
        if isinstance(message, dict):
            return message.get("msg_id") or message.get("id")
        return getattr(message, "msg_id", None) or getattr(message, "id", None)


def _display_name(author: Any) -> str:
    return str(
        getattr(author, "nickname", None)
        or getattr(author, "username", None)
        or getattr(author, "id", None)
        or ""
    )


def check_kook_requirements() -> bool:
    try:
        _khl()
        return True
    except Exception:
        return False


def validate_config(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    return bool(
        os.getenv("KOOK_TOKEN") or extra.get("token") or getattr(config, "token", "")
    )


def is_connected(config) -> bool:
    return validate_config(config)


def _env_enablement() -> dict | None:
    token = os.getenv("KOOK_TOKEN", "").strip()
    if not token:
        return None
    seed = {"token": token}
    home = os.getenv("KOOK_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {"chat_id": home, "name": home}
    return seed


def register(ctx) -> None:
    print("Registering KOOK platform")
    ctx.register_platform(
        name="kook",
        label="KOOK",
        adapter_factory=lambda cfg: KookAdapter(cfg),
        check_fn=check_kook_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["KOOK_TOKEN"],
        install_hint='Install KOOK support with: pip install "hermes-agent[kook]"',
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="KOOK_HOME_CHANNEL",
        allowed_users_env="KOOK_ALLOWED_USERS",
        allow_all_env="KOOK_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="🎮",
        pii_safe=False,
        allow_update_command=True,
        platform_hint=(
            "You are chatting via KOOK. KOOK supports KMarkdown and media messages. "
            "Keep responses concise for chat channels."
        ),
    )
