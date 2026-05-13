import asyncio
import json
import os
from typing import Any


KOOK_RAW_REQUEST_SCHEMA = {
    "name": "kook_raw_request",
    "description": (
        "Call KOOK OpenAPI using the configured KOOK bot token. "
        "Endpoint must start with /api/v3/. GET/DELETE/HEAD bodies are sent as query params; "
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
                "description": "KOOK API endpoint beginning with /api/v3/, e.g. /api/v3/user/me.",
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


def handle_kook_raw_request(method: str, endpoint: str, body: str) -> str:
    try:
        result = asyncio.run(_kook_raw_request(method=method, endpoint=endpoint, body=body))
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

    route = _normalize_endpoint(endpoint)
    payload = _parse_body(body)

    from khl import Bot

    bot = Bot(token=token)
    kwargs = {"params": payload} if normalized_method in _QUERY_METHODS else {"json": payload}
    return await bot.client.gate.request(normalized_method, route, **kwargs)


def _normalize_endpoint(endpoint: str) -> str:
    value = (endpoint or "").strip()
    if not value.startswith("/api/v3/"):
        raise ValueError("endpoint must start with /api/v3/")
    route = value[len("/api/v3/") :].strip("/")
    if not route:
        raise ValueError("endpoint must include a path after /api/v3/")
    return route


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
