import asyncio
import json
import os
from typing import Any

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
