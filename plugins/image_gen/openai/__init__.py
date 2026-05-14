"""OpenAI image generation backend.

Exposes OpenAI's ``gpt-image-2`` model at three quality tiers as an
:class:`ImageGenProvider` implementation. The tiers are implemented as
three virtual model IDs so the ``hermes tools`` model picker and the
``image_gen.model`` config key behave like any other multi-model backend:

    gpt-image-2-low     ~15s   fastest, good for iteration
    gpt-image-2-medium  ~40s   default — balanced
    gpt-image-2-high    ~2min  slowest, highest fidelity

All three hit the same underlying API model (``gpt-image-2``) with a
different ``quality`` parameter. Output is base64 JSON → saved under
``$HERMES_HOME/cache/images/``.

Endpoint resolution:

Set ``IMAGE_GEN_OPENAI_BASEURL`` to route image generation through a custom
OpenAI-compatible endpoint. When unset, the OpenAI SDK default endpoint is
used.

Credentials are read from ``IMAGE_GEN_OPENAI_API_KEY`` so image generation can
use a dedicated OpenAI-compatible key independent from the chat/model provider.
For image-conditioned requests, ``IMAGE_GEN_OPENAI_RESPONSES_MODEL`` can
override the Responses API host model. When a custom base URL is configured,
image-conditioned requests use ``/images/generations`` instead and pass
reference images through the request body for OpenAI-compatible gateways that
do not implement the Responses API.

Selection precedence (first hit wins):

1. ``OPENAI_IMAGE_MODEL`` env var (escape hatch for scripts / tests)
2. ``image_gen.openai.model`` in ``config.yaml``
3. ``image_gen.model`` in ``config.yaml`` (when it's one of our tier IDs)
4. :data:`DEFAULT_MODEL` — ``gpt-image-2-medium``
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------
#
# All three IDs resolve to the same underlying API model with a different
# ``quality`` setting. ``api_model`` is what gets sent to OpenAI;
# ``quality`` is the knob that changes generation time and output fidelity.

API_MODEL = "gpt-image-2-vip"
API_KEY_ENV = "IMAGE_GEN_OPENAI_API_KEY"
BASE_URL_ENV = "IMAGE_GEN_OPENAI_BASEURL"
RESPONSES_MODEL_ENV = "IMAGE_GEN_OPENAI_RESPONSES_MODEL"
RESPONSES_MODEL = "gpt-5.5"
MAX_INPUT_IMAGE_BYTES = 20 * 1024 * 1024
VALID_IMAGE_ACTIONS = {"auto", "generate", "edit"}

_MODELS: Dict[str, Dict[str, Any]] = {
    "gpt-image-2-low": {
        "display": "GPT Image 2 (Low)",
        "speed": "~15s",
        "strengths": "Fast iteration, lowest cost",
        "quality": "low",
    },
    "gpt-image-2-medium": {
        "display": "GPT Image 2 (Medium)",
        "speed": "~40s",
        "strengths": "Balanced — default",
        "quality": "medium",
    },
    "gpt-image-2-high": {
        "display": "GPT Image 2 (High)",
        "speed": "~2min",
        "strengths": "Highest fidelity, strongest prompt adherence",
        "quality": "high",
    },
}

DEFAULT_MODEL = "gpt-image-2-medium"

_SIZES = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}


def _load_openai_config() -> Dict[str, Any]:
    """Read ``image_gen`` from config.yaml (returns {} on any failure)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    """Decide which tier to use and return ``(model_id, meta)``."""
    env_override = os.environ.get("OPENAI_IMAGE_MODEL")
    if env_override and env_override in _MODELS:
        return env_override, _MODELS[env_override]

    cfg = _load_openai_config()
    openai_cfg = cfg.get("openai") if isinstance(cfg.get("openai"), dict) else {}
    candidate: Optional[str] = None
    if isinstance(openai_cfg, dict):
        value = openai_cfg.get("model")
        if isinstance(value, str) and value in _MODELS:
            candidate = value
    if candidate is None:
        top = cfg.get("model")
        if isinstance(top, str) and top in _MODELS:
            candidate = top

    if candidate is not None:
        return candidate, _MODELS[candidate]

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


def _resolve_base_url() -> Optional[str]:
    """Return the configured OpenAI image-gen base URL, if any."""
    base_url = os.environ.get(BASE_URL_ENV)
    if not isinstance(base_url, str):
        return None
    base_url = base_url.strip()
    return base_url or None


def _resolve_responses_model() -> str:
    """Return the host model for Responses API image-conditioned requests."""
    override = os.environ.get(RESPONSES_MODEL_ENV)
    if isinstance(override, str) and override.strip():
        return override.strip()
    return RESPONSES_MODEL


def _build_openai_client(openai_module: Any, base_url: Optional[str] = None) -> Any:
    """Build an OpenAI client, honoring the image-gen-specific base URL."""
    api_key = os.environ.get(API_KEY_ENV)
    client_kwargs: Dict[str, Any] = {
        "api_key": api_key,
        "default_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
        },
    }
    if base_url is None:
        base_url = _resolve_base_url()
    if base_url:
        client_kwargs["base_url"] = base_url

    return openai_module.OpenAI(**client_kwargs)


def _coerce_string_list(value: Any) -> List[str]:
    """Accept either a single string or list-like input and return clean strings."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    result: List[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
    return result


def _data_url_for_image_path(raw_path: str) -> str:
    """Return a data URL for a local image file path."""
    path = Path(raw_path).expanduser()
    if not path.is_file():
        raise ValueError(
            f"Input image path does not exist or is not a file: {raw_path}"
        )

    size = path.stat().st_size
    if size > MAX_INPUT_IMAGE_BYTES:
        limit_mb = MAX_INPUT_IMAGE_BYTES // (1024 * 1024)
        raise ValueError(f"Input image is too large: {raw_path} exceeds {limit_mb}MB")

    mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
    if not mime_type.startswith("image/"):
        raise ValueError(f"Input file is not recognized as an image: {raw_path}")

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _validate_image_url(value: str) -> str:
    """Return a supported image URL/data URL or raise a clear error."""
    if value.startswith("data:image/"):
        return value

    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return value

    raise ValueError(
        "Input image URLs must be fully qualified http(s) URLs or data:image/* URLs"
    )


def _build_input_image_content(
    *,
    prompt: str,
    image_urls: Any = None,
    image_paths: Any = None,
    file_ids: Any = None,
) -> List[Dict[str, Any]]:
    """Build Responses API message content with text plus input images."""
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]

    for image_url in _coerce_string_list(image_urls):
        content.append(
            {
                "type": "input_image",
                "image_url": _validate_image_url(image_url),
            }
        )

    for image_path in _coerce_string_list(image_paths):
        content.append(
            {
                "type": "input_image",
                "image_url": _data_url_for_image_path(image_path),
            }
        )

    for file_id in _coerce_string_list(file_ids):
        content.append({"type": "input_image", "file_id": file_id})

    return content


def _input_image_count(content: List[Dict[str, Any]]) -> int:
    return sum(1 for item in content if item.get("type") == "input_image")


def _build_image_generation_extra_body(
    input_content: List[Dict[str, Any]],
    *,
    action: str,
) -> Dict[str, Any]:
    """Build extra request body fields for custom ``images.generate`` gateways."""
    image_urls: List[str] = []
    file_ids: List[str] = []
    for item in input_content:
        if item.get("type") != "input_image":
            continue
        image_url = item.get("image_url")
        if isinstance(image_url, str) and image_url:
            image_urls.append(image_url)
        file_id = item.get("file_id")
        if isinstance(file_id, str) and file_id:
            file_ids.append(file_id)

    extra_body: Dict[str, Any] = {}
    if image_urls:
        extra_body["image_urls"] = image_urls
    if file_ids:
        extra_body["file_ids"] = file_ids
    if action and action != "auto":
        extra_body["action"] = action
    return extra_body


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _extract_response_image(response: Any) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(b64_image, revised_prompt)`` from a Responses API result."""
    for item in _field(response, "output") or []:
        if _field(item, "type") != "image_generation_call":
            continue
        result = _field(item, "result")
        if isinstance(result, str) and result:
            revised = _field(item, "revised_prompt")
            return result, revised if isinstance(revised, str) else None
    return None, None


def _create_image_response(
    client: Any,
    *,
    model: str,
    input_content: List[Dict[str, Any]],
    tool: Dict[str, Any],
) -> Any:
    return client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": input_content,
            }
        ],
        tools=[tool],
        tool_choice={"type": "image_generation"},
    )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class OpenAIImageGenProvider(ImageGenProvider):
    """OpenAI ``images.generate`` backend — gpt-image-2 at low/medium/high."""

    @property
    def name(self) -> str:
        return "openai"

    @property
    def display_name(self) -> str:
        return "OpenAI"

    def is_available(self) -> bool:
        if not os.environ.get(API_KEY_ENV):
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": "varies",
            }
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "OpenAI",
            "badge": "paid",
            "tag": "gpt-image-2 at low/medium/high quality tiers",
            "env_vars": [
                {
                    "key": API_KEY_ENV,
                    "prompt": "OpenAI image generation API key",
                    "url": "https://platform.openai.com/api-keys",
                },
            ],
            "post_setup_hint": (
                f"Set {BASE_URL_ENV} to use a custom OpenAI-compatible base URL."
            ),
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="openai",
                aspect_ratio=aspect,
            )

        if not os.environ.get(API_KEY_ENV):
            return error_response(
                error=(
                    f"{API_KEY_ENV} not set. Run `hermes tools` → Image "
                    "Generation → OpenAI to configure, or `hermes setup` "
                    "to add the key."
                ),
                error_type="auth_required",
                provider="openai",
                aspect_ratio=aspect,
            )

        try:
            import openai
        except ImportError:
            return error_response(
                error="openai Python package not installed (pip install openai)",
                error_type="missing_dependency",
                provider="openai",
                aspect_ratio=aspect,
            )

        tier_id, meta = _resolve_model()
        size = _SIZES.get(aspect, _SIZES["square"])

        action = kwargs.get("action")
        if isinstance(action, str):
            action = action.strip().lower() or "auto"
        else:
            action = "auto"
        if action not in VALID_IMAGE_ACTIONS:
            return error_response(
                error="action must be one of: auto, generate, edit",
                error_type="invalid_argument",
                provider="openai",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            input_content = _build_input_image_content(
                prompt=prompt,
                image_urls=kwargs.get("image_urls") or kwargs.get("image_url"),
                image_paths=kwargs.get("image_paths") or kwargs.get("image_path"),
                file_ids=kwargs.get("file_ids") or kwargs.get("file_id"),
            )
        except ValueError as exc:
            return error_response(
                error=str(exc),
                error_type="invalid_argument",
                provider="openai",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        input_image_count = _input_image_count(input_content)
        has_input_images = input_image_count > 0

        # gpt-image-2 returns b64_json unconditionally and REJECTS
        # ``response_format`` as an unknown parameter. Don't send it.
        payload: Dict[str, Any] = {
            "model": API_MODEL,
            "prompt": prompt,
            "size": size,
            "n": 1,
            "quality": meta["quality"],
        }

        try:
            base_url = _resolve_base_url()
            client = _build_openai_client(openai, base_url=base_url)

            if has_input_images:
                if base_url:
                    extra_body = _build_image_generation_extra_body(
                        input_content,
                        action=action,
                    )
                    if extra_body:
                        payload["extra_body"] = extra_body
                else:
                    tool: Dict[str, Any] = {
                        "type": "image_generation",
                        "model": API_MODEL,
                        "size": size,
                        "quality": meta["quality"],
                        "output_format": "png",
                        "action": action,
                    }
                    response = _create_image_response(
                        client,
                        model=_resolve_responses_model(),
                        input_content=input_content,
                        tool=tool,
                    )
                    b64, revised_prompt = _extract_response_image(response)
                    if not b64:
                        return error_response(
                            error=(
                                "OpenAI response contained no "
                                "image_generation_call result"
                            ),
                            error_type="empty_response",
                            provider="openai",
                            model=tier_id,
                            prompt=prompt,
                            aspect_ratio=aspect,
                        )
                    try:
                        saved_path = save_b64_image(b64, prefix=f"openai_{tier_id}")
                    except Exception as exc:
                        return error_response(
                            error=f"Could not save image to cache: {exc}",
                            error_type="io_error",
                            provider="openai",
                            model=tier_id,
                            prompt=prompt,
                            aspect_ratio=aspect,
                        )

                    extra: Dict[str, Any] = {
                        "size": size,
                        "quality": meta["quality"],
                        "action": action,
                        "input_image_count": input_image_count,
                    }
                    if revised_prompt:
                        extra["revised_prompt"] = revised_prompt

                    return success_response(
                        image=str(saved_path),
                        model=tier_id,
                        prompt=prompt,
                        aspect_ratio=aspect,
                        provider="openai",
                        extra=extra,
                    )

            response = client.images.generate(**payload)
        except Exception as exc:
            logger.debug("OpenAI image generation failed", exc_info=True)
            return error_response(
                error=f"OpenAI image generation failed: {exc}",
                error_type="api_error",
                provider="openai",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        data = getattr(response, "data", None) or []
        if not data:
            return error_response(
                error="OpenAI returned no image data",
                error_type="empty_response",
                provider="openai",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        first = data[0]
        b64 = getattr(first, "b64_json", None)
        url = getattr(first, "url", None)
        revised_prompt = getattr(first, "revised_prompt", None)

        if b64:
            try:
                saved_path = save_b64_image(b64, prefix=f"openai_{tier_id}")
            except Exception as exc:
                return error_response(
                    error=f"Could not save image to cache: {exc}",
                    error_type="io_error",
                    provider="openai",
                    model=tier_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            image_ref = str(saved_path)
        elif url:
            # Defensive — gpt-image-2 returns b64 today, but fall back
            # gracefully if the API ever changes.
            image_ref = url
        else:
            return error_response(
                error="OpenAI response contained neither b64_json nor URL",
                error_type="empty_response",
                provider="openai",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        extra: Dict[str, Any] = {"size": size, "quality": meta["quality"]}
        if has_input_images:
            extra["action"] = action
            extra["input_image_count"] = input_image_count
        if revised_prompt:
            extra["revised_prompt"] = revised_prompt

        return success_response(
            image=image_ref,
            model=tier_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="openai",
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``OpenAIImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(OpenAIImageGenProvider())
