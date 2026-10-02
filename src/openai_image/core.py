"""Core request validation and OpenAI image API client."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import unicodedata
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "gpt-image-2.5-flare"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OUTPUT_DIR = "."

VALID_QUALITIES = frozenset({"low", "medium", "high", "xhigh", "max", "auto"})
VALID_FORMATS = frozenset({"png", "jpeg", "webp"})
VALID_MODERATION = frozenset({"auto", "low"})
VALID_BACKGROUND = frozenset({"auto", "opaque", "transparent"})
VALID_TRANSPORTS = frozenset({"auto", "json", "multipart"})


class OpenAIImageCLIError(Exception):
    """A validation, filesystem, transport, or response error."""

    def __init__(self, message: str, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = dict(details or {})


class OpenAIAPIError(OpenAIImageCLIError):
    """An error response from the OpenAI API."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        request_id: str | None = None,
        code: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.status = status
        self.request_id = request_id
        self.code = code


@dataclass(frozen=True)
class ImageReference:
    kind: str
    value: str
    original: str


@dataclass(frozen=True)
class ImageRequest:
    cwd: Path
    model: str
    prompt: str
    n: int
    size: str
    parsed_size: Mapping[str, Any]
    quality: str
    output_format: str
    output_compression: int | None
    background: str | None
    moderation: str | None
    transport: str
    endpoint: str
    reference_images: tuple[str, ...]
    reference_objects: tuple[ImageReference, ...]
    mask: str | None
    mask_object: ImageReference | None
    output_dir: Path
    output_prefix: str
    save_metadata: bool
    dry_run: bool
    user: str | None
    warnings: tuple[str, ...]


def _strip_leading_at(value: str) -> str:
    return value.removeprefix("@")


def _expand_path(value: str, cwd: Path) -> Path:
    path = Path(_strip_leading_at(value)).expanduser()
    return path.resolve() if path.is_absolute() else (cwd / path).resolve()


def classify_reference(value: Any, cwd: str | Path | None = None) -> ImageReference:
    if not isinstance(value, str) or not value.strip():
        raise OpenAIImageCLIError("Image references must be non-empty strings.")
    original = value.strip()
    stripped = _strip_leading_at(original)
    if re.match(r"^https?://", stripped, re.IGNORECASE):
        return ImageReference("url", stripped, original)
    if re.match(r"^data:image/[a-z0-9.+-]+;base64,", stripped, re.IGNORECASE):
        return ImageReference("data_url", stripped, original)
    if re.match(r"^file-[A-Za-z0-9_-]+$", stripped):
        return ImageReference("file_id", stripped, original)
    return ImageReference(
        "local", str(_expand_path(original, Path(cwd or Path.cwd()))), original
    )


def _parse_integer(value: Any, name: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise OpenAIImageCLIError(f"{name} must be an integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise OpenAIImageCLIError(f"{name} must be an integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise OpenAIImageCLIError(f"{name} must be an integer.")
    if isinstance(value, str) and not re.fullmatch(r"[+-]?\d+", value.strip()):
        raise OpenAIImageCLIError(f"{name} must be an integer.")
    return parsed


def _parse_boolean(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise OpenAIImageCLIError(f"Invalid boolean value: {value}")


def parse_size(size: Any) -> dict[str, Any] | None:
    if size is None or size == "":
        return None
    text = str(size).strip().lower()
    if text == "auto":
        return {"kind": "auto", "value": "auto"}
    match = re.fullmatch(r"(\d{2,5})x(\d{2,5})", text)
    if not match:
        raise OpenAIImageCLIError(
            f'Invalid size "{size}". Use "auto" or WIDTHxHEIGHT, for example 1024x1024.'
        )
    width, height = map(int, match.groups())
    return {
        "kind": "dimensions",
        "value": f"{width}x{height}",
        "width": width,
        "height": height,
        "pixels": width * height,
        "aspect_ratio": width / height,
    }


def _is_gpt_image_25(model: str) -> bool:
    return re.match(r"^gpt-image-2\.5-(?:flare|sunburst)(?:-|$)", model) is not None


def _is_gpt_image_2(model: str) -> bool:
    return model == "gpt-image-2" or model.startswith("gpt-image-2-")


def _validate_gpt_image_size(size: Mapping[str, Any] | None, model: str) -> list[str]:
    if not size or size["kind"] == "auto":
        return []
    width, height = size["width"], size["height"]
    pixels, ratio = size["pixels"], size["aspect_ratio"]
    if width % 16 or height % 16:
        raise OpenAIImageCLIError(
            f"{model} size must have both edges divisible by 16; got {width}x{height}."
        )
    if ratio > 3 or ratio < 1 / 3:
        raise OpenAIImageCLIError(
            f"{model} aspect ratio must be between 1:3 and 3:1; got {width}x{height}."
        )
    if pixels < 655_360:
        raise OpenAIImageCLIError(
            f"{model} size must be at least 655,360 pixels; got {pixels:,}."
        )
    if pixels > 8_294_400:
        raise OpenAIImageCLIError(
            f"{model} size must not exceed 8,294,400 pixels; got {pixels:,}."
        )
    if width > 3840 or height > 3840:
        raise OpenAIImageCLIError(
            f"{model} maximum edge is 3840px; got {width}x{height}."
        )
    if pixels > 3_686_400:
        return [
            (
                f"Requested {width}x{height} ({pixels:,} pixels), above 2560x1440; "
                "OpenAI describes this range as experimental."
            )
        ]
    return []


def _sanitize_filename_part(value: Any, fallback: str = "image") -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    without_marks = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", without_marks)
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")[:80]
    return cleaned or fallback


def _slug_from_prompt(prompt: str) -> str:
    words = re.sub(r"[^a-z0-9\s-]", " ", prompt.lower()).strip().split()[:8]
    return _sanitize_filename_part("-".join(words))


def _timestamp_for_filename() -> str:
    stamp = (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
    return stamp.replace(":", "-").replace(".", "-")


def _string_values(*values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        if value is None:
            continue
        items = value if isinstance(value, (list, tuple)) else [value]
        result.extend(text for item in items if (text := str(item).strip()))
    return result


def _default_output_dir() -> str:
    return (
        os.getenv("OPENAI_IMAGE_OUTPUT_DIR")
        or DEFAULT_OUTPUT_DIR
    )


def normalize_image_request(
    input_data: Mapping[str, Any] | None = None, *, cwd: str | Path | None = None
) -> ImageRequest:
    data = input_data or {}
    if not isinstance(data, Mapping):
        raise OpenAIImageCLIError("Image request must be an object.")
    working_dir = Path(cwd or data.get("cwd") or Path.cwd()).expanduser().resolve()
    prompt = data.get("prompt", "")
    prompt = prompt.strip() if isinstance(prompt, str) else ""
    if not prompt:
        raise OpenAIImageCLIError("prompt is required.")

    model = str(data.get("model") or DEFAULT_MODEL).strip()
    n_value = data.get("n") if data.get("n") is not None else data.get("count")
    parsed_n = _parse_integer(n_value, "n")
    n = 1 if parsed_n is None else parsed_n
    if not 1 <= n <= 10:
        raise OpenAIImageCLIError("n must be between 1 and 10.")

    quality = str(data.get("quality", "medium")).strip().lower()
    if quality not in VALID_QUALITIES:
        raise OpenAIImageCLIError(
            "quality must be one of: low, medium, high, xhigh, max, auto."
        )
    if quality in {"xhigh", "max"} and not _is_gpt_image_25(model):
        raise OpenAIImageCLIError(
            "quality xhigh and max require gpt-image-2.5-flare or gpt-image-2.5-sunburst."
        )

    format_value = data.get("output_format")
    if format_value is None:
        format_value = data.get("format", "png")
    output_format = str(format_value).strip().lower()
    if output_format not in VALID_FORMATS:
        raise OpenAIImageCLIError("output_format must be one of: png, jpeg, webp.")

    compression = _parse_integer(
        data.get("output_compression", data.get("compression")), "output_compression"
    )
    if compression is not None and not 0 <= compression <= 100:
        raise OpenAIImageCLIError("output_compression must be between 0 and 100.")
    if compression is not None and output_format == "png":
        raise OpenAIImageCLIError(
            "output_compression is only supported for jpeg or webp outputs."
        )

    background_value = data.get("background")
    background = (
        None if background_value is None else str(background_value).strip().lower()
    )
    if background is not None and background not in VALID_BACKGROUND:
        raise OpenAIImageCLIError(
            "background must be one of: auto, opaque, transparent."
        )
    if (
        _is_gpt_image_2(model)
        and not _is_gpt_image_25(model)
        and background == "transparent"
    ):
        raise OpenAIImageCLIError(
            "gpt-image-2 does not support transparent backgrounds; use a GPT Image 2.5 model."
        )
    if background == "transparent" and output_format == "jpeg":
        raise OpenAIImageCLIError("transparent backgrounds require png or webp output.")

    moderation_value = data.get("moderation")
    moderation = (
        None if moderation_value is None else str(moderation_value).strip().lower()
    )
    if moderation is not None and moderation not in VALID_MODERATION:
        raise OpenAIImageCLIError("moderation must be one of: auto, low.")

    transport = str(data.get("transport") or "auto").strip().lower()
    if transport not in VALID_TRANSPORTS:
        raise OpenAIImageCLIError("transport must be one of: auto, json, multipart.")

    parsed_size = parse_size(data.get("size") or "1024x1024")
    assert parsed_size is not None
    size = parsed_size["value"]
    warnings: list[str] = []
    if _is_gpt_image_2(model) or _is_gpt_image_25(model):
        warnings.extend(_validate_gpt_image_size(parsed_size, model))

    references = _string_values(
        data.get("reference_images"), data.get("references"), data.get("reference")
    )
    if len(references) > 16:
        raise OpenAIImageCLIError("reference_images supports at most 16 images.")
    reference_objects = tuple(
        classify_reference(ref, working_dir) for ref in references
    )
    mask_value = data.get("mask")
    mask = (
        str(mask_value).strip()
        if mask_value is not None and str(mask_value).strip()
        else None
    )
    mask_object = classify_reference(mask, working_dir) if mask else None
    if mask_object and not reference_objects:
        raise OpenAIImageCLIError("mask requires at least one reference image.")

    endpoint = "edits" if reference_objects or mask_object else "generations"
    output_dir_value = str(data.get("output_dir") or _default_output_dir())
    output_dir_path = Path(output_dir_value).expanduser()
    output_dir = (
        output_dir_path.resolve()
        if output_dir_path.is_absolute()
        else (working_dir / output_dir_path).resolve()
    )
    default_prefix = f"{_timestamp_for_filename()}_{_slug_from_prompt(prompt)}"
    output_prefix = _sanitize_filename_part(data.get("output_prefix") or default_prefix)
    save_metadata = _parse_boolean(data.get("save_metadata"))
    dry_run = _parse_boolean(data.get("dry_run"))
    user_value = data.get("user")

    return ImageRequest(
        cwd=working_dir,
        model=model,
        prompt=prompt,
        n=n,
        size=size,
        parsed_size=parsed_size,
        quality=quality,
        output_format=output_format,
        output_compression=compression,
        background=background,
        moderation=moderation,
        transport=transport,
        endpoint=endpoint,
        reference_images=tuple(references),
        reference_objects=reference_objects,
        mask=mask,
        mask_object=mask_object,
        output_dir=output_dir,
        output_prefix=output_prefix,
        save_metadata=True if save_metadata is None else save_metadata,
        dry_run=False if dry_run is None else dry_run,
        user=None if user_value is None else str(user_value),
        warnings=tuple(warnings),
    )


def get_confirmation_reasons(request: ImageRequest) -> list[str]:
    reasons: list[str] = []
    if request.n > 1:
        reasons.append(f"generating {request.n} images")
    if request.quality in {"high", "xhigh", "max"}:
        reasons.append(f"{request.quality} quality output")
    if request.reference_objects:
        reasons.append(f"{len(request.reference_objects)} reference image(s)")
    if request.parsed_size.get("pixels", 0) > 3_686_400:
        reasons.append("large/experimental output size")
    return reasons


def get_human_request_summary(request: ImageRequest) -> str:
    action = "Edit/reference-generate" if request.endpoint == "edits" else "Generate"
    noun = "image" if request.n == 1 else "images"
    parts = [
        f"{action} {request.n} {noun}",
        f"model={request.model}",
        f"size={request.size}",
        f"quality={request.quality}",
        f"format={request.output_format}",
    ]
    if request.reference_objects:
        parts.append(f"references={len(request.reference_objects)}")
    if request.mask_object:
        parts.append("mask=yes")
    return ", ".join(parts)


def choose_transport(request: ImageRequest) -> str:
    if request.endpoint != "edits":
        return "json"
    if request.transport != "auto":
        return request.transport
    refs_local = all(ref.kind == "local" for ref in request.reference_objects)
    mask_local = request.mask_object is None or request.mask_object.kind == "local"
    return "multipart" if refs_local and mask_local else "json"


def _request_parameters(request: ImageRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": request.model, "prompt": request.prompt}
    values = {
        "n": request.n,
        "size": request.size,
        "quality": request.quality,
        "output_format": request.output_format,
        "output_compression": request.output_compression,
        "background": request.background,
        "moderation": request.moderation,
        "user": request.user,
    }
    payload.update({key: value for key, value in values.items() if value is not None})
    return payload


def _ensure_local_file(reference: ImageReference, label: str) -> Path:
    path = Path(reference.value)
    if not path.is_file():
        raise OpenAIImageCLIError(f"{label} file not found or unreadable: {path}")
    return path


def _mime_from_path(path: Path) -> str:
    known = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    return (
        known.get(path.suffix.lower())
        or mimetypes.guess_type(path.name)[0]
        or "application/octet-stream"
    )


def _local_file_to_data_url(reference: ImageReference, label: str) -> str:
    path = _ensure_local_file(reference, label)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{_mime_from_path(path)};base64,{encoded}"


def _json_body(request: ImageRequest) -> bytes:
    payload = _request_parameters(request)
    if request.endpoint == "edits":
        images: list[dict[str, str]] = []
        for index, ref in enumerate(request.reference_objects, 1):
            if ref.kind == "file_id":
                images.append({"file_id": ref.value})
            elif ref.kind in {"url", "data_url"}:
                images.append({"image_url": ref.value})
            else:
                images.append(
                    {
                        "image_url": _local_file_to_data_url(
                            ref, f"reference image {index}"
                        )
                    }
                )
        payload["images"] = images
        if request.mask_object:
            ref = request.mask_object
            if ref.kind == "file_id":
                payload["mask"] = {"file_id": ref.value}
            elif ref.kind in {"url", "data_url"}:
                payload["mask"] = {"image_url": ref.value}
            else:
                payload["mask"] = {"image_url": _local_file_to_data_url(ref, "mask")}
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _multipart_body(request: ImageRequest) -> tuple[bytes, str]:
    if request.endpoint != "edits":
        raise OpenAIImageCLIError("Multipart transport is only used for edits.")
    if any(ref.kind != "local" for ref in request.reference_objects):
        raise OpenAIImageCLIError(
            "Multipart edits require local reference image paths only."
        )
    if request.mask_object and request.mask_object.kind != "local":
        raise OpenAIImageCLIError("Multipart edits require a local mask path.")

    boundary = f"----openai-image-{uuid.uuid4().hex}"
    body = bytearray()

    def line(value: str) -> None:
        body.extend(value.encode("utf-8") + b"\r\n")

    def add_field(name: str, value: Any) -> None:
        line(f"--{boundary}")
        line(f'Content-Disposition: form-data; name="{name}"')
        line("")
        line(str(value))

    def add_file(name: str, ref: ImageReference, label: str) -> None:
        path = _ensure_local_file(ref, label)
        filename = path.name.replace('"', "_")
        line(f"--{boundary}")
        line(f'Content-Disposition: form-data; name="{name}"; filename="{filename}"')
        line(f"Content-Type: {_mime_from_path(path)}")
        line("")
        body.extend(path.read_bytes())
        body.extend(b"\r\n")

    for key, value in _request_parameters(request).items():
        add_field(key, value)
    for index, ref in enumerate(request.reference_objects, 1):
        add_file("image[]", ref, f"reference image {index}")
    if request.mask_object:
        add_file("mask", request.mask_object, "mask")
    line(f"--{boundary}--")
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _request_id(headers: Any) -> str | None:
    return headers.get("x-request-id") or headers.get("openai-request-id")


def _decode_json(data: bytes) -> dict[str, Any]:
    if not data:
        return {}
    try:
        value = json.loads(data.decode("utf-8"))
        return value if isinstance(value, dict) else {"data": value}
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"raw": data.decode("utf-8", errors="replace")}


def _call_openai(
    request: ImageRequest,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
) -> tuple[dict[str, Any], str | None, str, str]:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise OpenAIImageCLIError(
            "OPENAI_API_KEY is not set. Export it before generating images."
        )
    root = (base_url or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    endpoint_path = (
        "/images/edits" if request.endpoint == "edits" else "/images/generations"
    )
    url = f"{root}{endpoint_path}"
    transport = choose_transport(request)
    headers = {
        "Authorization": f"Bearer {key}",
        "User-Agent": "openai-image/0.1.0",
    }
    organization = os.getenv("OPENAI_ORGANIZATION")
    project = os.getenv("OPENAI_PROJECT")
    if organization:
        headers["OpenAI-Organization"] = organization
    if project:
        headers["OpenAI-Project"] = project
    if transport == "multipart":
        body, content_type = _multipart_body(request)
        headers["Content-Type"] = content_type
    else:
        body = _json_body(request)
        headers["Content-Type"] = "application/json"

    http_request = urllib.request.Request(
        url, data=body, headers=headers, method="POST"
    )
    request_timeout = timeout or float(os.getenv("OPENAI_IMAGE_TIMEOUT", "600"))
    try:
        with urllib.request.urlopen(http_request, timeout=request_timeout) as response:
            response_data = _decode_json(response.read())
            return response_data, _request_id(response.headers), transport, url
    except urllib.error.HTTPError as exc:
        response_data = _decode_json(exc.read())
        error = response_data.get("error", response_data)
        if not isinstance(error, Mapping):
            error = {"message": str(error)}
        message = (
            error.get("message") or f"OpenAI image request failed with HTTP {exc.code}."
        )
        raise OpenAIAPIError(
            str(message),
            status=exc.code,
            request_id=_request_id(exc.headers),
            code=error.get("code"),
            details={"type": error.get("type"), "response": response_data},
        ) from exc
    except urllib.error.URLError as exc:
        raise OpenAIImageCLIError(f"OpenAI image request failed: {exc.reason}") from exc


def _bytes_from_image_item(
    item: Mapping[str, Any], *, timeout: float | None = None
) -> bytes:
    encoded = item.get("b64_json")
    if encoded:
        try:
            return base64.b64decode(encoded)
        except (ValueError, TypeError) as exc:
            raise OpenAIImageCLIError(
                "OpenAI response contained invalid base64 image data."
            ) from exc
    url = item.get("url")
    if url:
        try:
            with urllib.request.urlopen(str(url), timeout=timeout or 600) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            raise OpenAIImageCLIError(
                f"Failed to download image URL: HTTP {exc.code}"
            ) from exc
        except urllib.error.URLError as exc:
            raise OpenAIImageCLIError(
                f"Failed to download image URL: {exc.reason}"
            ) from exc
    raise OpenAIImageCLIError(
        "OpenAI response did not contain b64_json or url for an image."
    )


def _unique_path(path: Path) -> Path:
    candidate = path
    counter = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}-{counter}{path.suffix}")
        counter += 1
    return candidate


def _reference_summary(reference: ImageReference) -> dict[str, str]:
    return {
        "kind": reference.kind,
        "value": "data-url" if reference.kind == "data_url" else reference.value,
    }


def _strip_response_images(response: Mapping[str, Any]) -> dict[str, Any]:
    stripped = dict(response)
    if isinstance(response.get("data"), list):
        items: list[Any] = []
        for item in response["data"]:
            if isinstance(item, Mapping):
                clean = {
                    key: value
                    for key, value in item.items()
                    if key not in {"b64_json", "url"}
                }
                clean["has_b64_json"] = bool(item.get("b64_json"))
                clean["has_url"] = bool(item.get("url"))
                items.append(clean)
            else:
                items.append(item)
        stripped["data"] = items
    return stripped


def create_image(
    input_data: Mapping[str, Any] | None = None,
    *,
    cwd: str | Path | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    request = normalize_image_request(input_data, cwd=cwd)
    reasons = get_confirmation_reasons(request)
    summary = get_human_request_summary(request)
    references = [_reference_summary(ref) for ref in request.reference_objects]
    mask = _reference_summary(request.mask_object) if request.mask_object else None

    if request.dry_run:
        result: dict[str, Any] = {
            "ok": True,
            "dry_run": True,
            "summary": summary,
            "endpoint": request.endpoint,
            "transport": choose_transport(request),
            "model": request.model,
            "prompt": request.prompt,
            "n_requested": request.n,
            "size": request.size,
            "quality": request.quality,
            "output_format": request.output_format,
            "output_dir": str(request.output_dir),
            "output_prefix": request.output_prefix,
            "reference_images": references,
            "warnings": list(request.warnings),
            "confirmation_reasons": reasons,
        }
        if mask:
            result["mask"] = mask
        return result

    response, request_id, transport, url = _call_openai(
        request, api_key=api_key, base_url=base_url, timeout=timeout
    )
    images = response.get("data")
    if not isinstance(images, list) or not images:
        raise OpenAIImageCLIError("OpenAI response did not contain any images.")

    request.output_dir.mkdir(parents=True, exist_ok=True)
    response_format = str(response.get("output_format") or request.output_format)
    extension = "jpg" if response_format == "jpeg" else response_format
    files: list[dict[str, Any]] = []
    for index, item in enumerate(images, 1):
        if not isinstance(item, Mapping):
            raise OpenAIImageCLIError(
                "OpenAI response contained an invalid image item."
            )
        image_bytes = _bytes_from_image_item(item, timeout=timeout)
        suffix = f"_{index:02d}" if len(images) > 1 else ""
        target = _unique_path(
            request.output_dir / f"{request.output_prefix}{suffix}.{extension}"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(image_bytes)
        files.append(
            {
                "index": index,
                "path": str(target),
                "bytes": len(image_bytes),
                "format": response_format,
            }
        )

    metadata: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
        "request_id": request_id,
        "endpoint": request.endpoint,
        "transport": transport,
        "url": url,
        "model": request.model,
        "prompt": request.prompt,
        "n_requested": request.n,
        "n_returned": len(images),
        "size": response.get("size") or request.size,
        "quality": response.get("quality") or request.quality,
        "output_format": response_format,
        "background": response.get("background") or request.background,
        "reference_images": references,
        "files": files,
        "usage": response.get("usage"),
        "warnings": list(request.warnings),
        "response": _strip_response_images(response),
    }
    optional_metadata = {
        "output_compression": request.output_compression,
        "moderation": request.moderation,
        "mask": mask,
    }
    metadata.update(
        {key: value for key, value in optional_metadata.items() if value is not None}
    )

    metadata_path: Path | None = None
    if request.save_metadata:
        metadata_path = _unique_path(
            request.output_dir / f"{request.output_prefix}.json"
        )
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    result: dict[str, Any] = {
        "ok": True,
        "dry_run": False,
        "summary": summary,
        "endpoint": request.endpoint,
        "transport": transport,
        "model": request.model,
        "prompt": request.prompt,
        "n_requested": request.n,
        "n_returned": len(images),
        "size": response.get("size") or request.size,
        "quality": response.get("quality") or request.quality,
        "output_format": response_format,
        "files": files,
        "warnings": list(request.warnings),
    }
    if request_id is not None:
        result["request_id"] = request_id
    if metadata_path is not None:
        result["metadata_path"] = str(metadata_path)
    if response.get("usage") is not None:
        result["usage"] = response["usage"]
    return result


generate_image = create_image


def format_result_summary(result: Mapping[str, Any]) -> str:
    if result.get("dry_run"):
        lines = [
            f"Dry run: {result['summary']}",
            f"Output directory: {result['output_dir']}",
        ]
        if result.get("reference_images"):
            lines.append(f"Reference images: {len(result['reference_images'])}")
        if result.get("mask"):
            lines.append("Mask: yes")
        if result.get("warnings"):
            lines.append(f"Warnings: {' '.join(result['warnings'])}")
        if result.get("confirmation_reasons"):
            lines.append(f"Would confirm: {', '.join(result['confirmation_reasons'])}")
        return "\n".join(lines)

    returned = result["n_returned"]
    lines = [
        f"Generated {returned} image{'s' if returned != 1 else ''} ({result['summary']})."
    ]
    for file in result["files"]:
        lines.append(f"- {file['path']} ({round(file['bytes'] / 1024)} KiB)")
    if result.get("metadata_path"):
        lines.append(f"Metadata: {result['metadata_path']}")
    if result.get("request_id"):
        lines.append(f"OpenAI request id: {result['request_id']}")
    if result.get("warnings"):
        lines.append(f"Warnings: {' '.join(result['warnings'])}")
    return "\n".join(lines)
