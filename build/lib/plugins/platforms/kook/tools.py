import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SEND_FILE_BEHIND_LINK_SCHEMA = {
    "name": "send_file_behind_link",
    "description": (
        "When you need to send a file or image but you only have a direct URL, "
        "you MUST use this tool to deliver it. The tool automatically downloads the URL, "
        "renames it to ``file_name``, and sends it as a KOOK file attachment "
        "to the current chat, in the correct way. DO NOT DO THIS BY YOURSELF, "
        "let the tool handle it automatically."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Direct HTTP(S) URL of the file to download and send.",
            },
            "file_name": {
                "type": "string",
                "description": "Filename shown to the recipient. Include the extension.",
            },
        },
        "required": ["url", "file_name"],
    },
}


def handle_send_file_behind_link(body, **kwargs) -> str:
    try:
        url = body.get("url") if isinstance(body, dict) else ""
        file_name = body.get("file_name") if isinstance(body, dict) else ""
        print("send_file_behind_link", "url=", url, "file_name=", file_name)

        result = asyncio.run(
            _send_file_behind_link(url=url or "", file_name=file_name or "")
        )
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.warning("send_file_behind_link failed: %s", exc)
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


def _scratch_dir() -> Path:
    return Path(tempfile.gettempdir())


def _download_to_path(url: str, dest: Path) -> None:
    import httpx

    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
        response.raise_for_status()
        with dest.open("wb") as fp:
            for chunk in response.iter_bytes():
                if chunk:
                    fp.write(chunk)


async def _send_file_behind_link(url: str, file_name: str) -> Any:
    token = os.getenv("KOOK_TOKEN", "").strip()
    if not token:
        raise ValueError("KOOK_TOKEN is not configured")

    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        raise ValueError("url must be an http:// or https:// URL")

    name = (file_name or "").strip()
    if not name:
        raise ValueError("file_name is required")
    if "/" in name or "\\" in name:
        raise ValueError("file_name must not contain path separators")

    from gateway.session_context import get_session_env

    chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "").strip()
    if not chat_id:
        raise ValueError(
            "no active chat context; cannot determine where to send the file"
        )
    user_id = get_session_env("HERMES_SESSION_USER_ID", "").strip()
    is_dm = bool(user_id) and user_id == chat_id

    print(
        "send_file_behind_link",
        "url=",
        url,
        "file_name=",
        name,
        "is_dm=",
        is_dm,
        "chat_id=",
        chat_id,
        "user_id=",
        user_id,
    )

    dest = _scratch_dir() / name
    _download_to_path(url, dest)

    from khl import Bot, MessageTypes

    bot = Bot(token=token)
    try:
        asset_url = await bot.client.create_asset(dest)
        if is_dm:
            target = await bot.client.fetch_user(chat_id)
        else:
            target = await bot.client.fetch_public_channel(chat_id)

        print("asset_url=", asset_url)
        is_image = name.lower().endswith((
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".jfif",
            ".webp",
        ))
        print("is_image=", is_image)

        if is_image:
            ret = await target.send(asset_url, type=MessageTypes.IMG)
        else:
            ret = await target.send(asset_url, type=MessageTypes.FILE)
        print("send_file_behind_link", "ret=", ret)
        return ret
    finally:
        await _close_bot_session(bot)


async def _close_bot_session(bot) -> None:
    requester = getattr(
        getattr(getattr(bot, "client", None), "gate", None), "requester", None
    )
    cs = getattr(requester, "_cs", None)
    if cs is None:
        return
    try:
        await cs.close()
    except Exception:
        logger.debug("KOOK: failed to close aiohttp session", exc_info=True)


def check_send_file_behind_link_requirements() -> bool:
    return bool(os.getenv("KOOK_TOKEN", "").strip())
