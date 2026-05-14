import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

KOOK_RAW_REQUEST_SCHEMA = {
    "name": "kook_raw_request",
    "description": (
        "Call KOOK OpenAPI using the configured KOOK bot token. "
        "Endpoint must start with /api/. GET/DELETE/HEAD bodies are sent as query params; "
        "POST/PUT/PATCH bodies are sent as JSON request bodies."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "description": "HTTP method, e.g. GET, POST, PUT, PATCH, DELETE, HEAD.",
            },
            "endpoint": {
                "type": "string",
                "description": "KOOK API endpoint beginning with /api/, e.g. /api/v3/user/me.",
            },
            "body": {
                "type": "string",
                "description": "Serialized JSON object used as query params for GET/DELETE/HEAD or JSON body for POST/PUT/PATCH.",
            },
        },
        "required": ["method", "endpoint", "body"],
    },
}

_QUERY_METHODS = {"GET", "DELETE", "HEAD"}
_BODY_METHODS = {"POST", "PUT", "PATCH"}


def check_kook_raw_request_requirements() -> bool:
    return bool(os.getenv("KOOK_TOKEN", "").strip())


# method: str, endpoint: str, body: str,
def handle_kook_raw_request(body, **kwargs) -> str:
    print("kook raw request", body)
    method = body["method"]
    endpoint = body["endpoint"]
    body = body["body"]

    if not method:
        raise ValueError("method is required")

    if not endpoint:
        raise ValueError("endpoint is required")

    if not body:
        body = "{}"

    try:
        result = asyncio.run(
            _kook_raw_request(method=method, endpoint=endpoint, body=body)
        )
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


async def _kook_raw_request(method: str, endpoint: str, body: str) -> Any:
    token = os.getenv("KOOK_TOKEN", "").strip()
    if not token:
        raise ValueError("KOOK_TOKEN is not configured")

    normalized_method = (method or "").strip().upper()
    if normalized_method not in _QUERY_METHODS | _BODY_METHODS:
        raise ValueError("method must be one of GET, DELETE, HEAD, POST, PUT, PATCH")

    payload = _parse_body(body)

    from khl import Bot

    bot = Bot(token=token)
    kwargs = (
        {"params": payload}
        if normalized_method in _QUERY_METHODS
        else {"json": payload}
    )
    return await bot.client.gate.request_endpoint(normalized_method, endpoint, **kwargs)


def _parse_body(body: str) -> dict[str, Any]:
    if body is None or str(body).strip() == "":
        return {}
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"body must be valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("body must be a JSON object")
    return parsed


SEND_FILE_BEHIND_LINK_SCHEMA = {
    "name": "send_file_behind_link",
    "description": (
        "When you need to send a file or image but you only have a direct URL, "
        "you MUST use this tool to deliver it. The tool automatically downloads the URL, "
        "renames it to ``file_name``, and sends it as a KOOK file attachment "
        "to the current chat, in the correct way."
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
    asset_url = await bot.client.create_asset(dest)
    if is_dm:
        target = await bot.client.fetch_user(chat_id)
    else:
        target = await bot.client.fetch_public_channel(chat_id)

    is_image = name.lower().endswith((
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".jfif",
        ".webp",
    ))
    print(
        "send_file_behind_link",
        "is_image=",
        is_image,
        "target=",
        target,
    )

    if is_image:
        return await target.send(asset_url, MessageTypes.IMAGE)
    else:
        return await target.send(asset_url, MessageTypes.FILE)


def check_send_file_behind_link_requirements() -> bool:
    return bool(os.getenv("KOOK_TOKEN", "").strip())
