"""Validation utilities for moondream-mcp."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

from .models import CaptionLength, SpatialRef

ImagePathsInput = Union[str, List[str]]
SpatialRefsInput = Union[str, List[SpatialRef], None]
MessagesInput = Union[str, List[Dict[str, Any]]]


class ValidationError(Exception):
    """Validation error with a stable machine-readable code."""

    def __init__(self, message: str, error_code: str = "VALIDATION_ERROR") -> None:
        self.message = message
        self.error_code = error_code
        super().__init__(message)


def validate_image_path(image_path: str) -> str:
    if not image_path or not image_path.strip():
        raise ValidationError("Image path cannot be empty", "EMPTY_PATH")

    image_path = image_path.strip()
    if _is_http_url(image_path):
        return image_path
    if "://" in image_path:
        raise ValidationError(
            "Only http:// and https:// image URLs are supported",
            "UNSUPPORTED_URL_SCHEME",
        )

    try:
        return str(Path(image_path).expanduser())
    except Exception as exc:
        raise ValidationError(f"Invalid file path: {exc}", "INVALID_PATH") from exc


def validate_question(question: str) -> str:
    if not question or not question.strip():
        raise ValidationError("Question cannot be empty", "EMPTY_QUESTION")

    question = sanitize_string(question, max_length=1000)
    if not question:
        raise ValidationError("Question cannot be empty", "EMPTY_QUESTION")
    return question


def validate_object_name(object_name: str) -> str:
    if not object_name or not object_name.strip():
        raise ValidationError("Object name cannot be empty", "EMPTY_OBJECT_NAME")

    object_name = sanitize_string(object_name, max_length=100)
    if not object_name:
        raise ValidationError("Object name cannot be empty", "EMPTY_OBJECT_NAME")
    return object_name


def validate_caption_length(length: str) -> CaptionLength:
    try:
        return CaptionLength(length.strip().lower())
    except (AttributeError, ValueError) as exc:
        valid_lengths = [item.value for item in CaptionLength]
        raise ValidationError(
            f"Invalid caption length '{length}'. Valid options: {valid_lengths}",
            "INVALID_LENGTH",
        ) from exc


def validate_operation(operation: str) -> str:
    valid_operations = ["caption", "query", "detect", "point", "segment"]
    normalized = operation.strip().lower() if isinstance(operation, str) else ""
    if normalized not in valid_operations:
        raise ValidationError(
            f"Invalid operation '{operation}'. Valid operations: {valid_operations}",
            "INVALID_OPERATION",
        )
    return normalized


def validate_image_paths_list(paths: ImagePathsInput) -> List[str]:
    if isinstance(paths, str):
        if not paths.strip():
            raise ValidationError("Image paths cannot be empty", "EMPTY_PATHS")
        try:
            paths_data = json.loads(paths.strip())
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"Invalid JSON format: {exc}", "INVALID_JSON"
            ) from exc
    else:
        paths_data = paths

    if not isinstance(paths_data, list):
        raise ValidationError("Image paths must be an array", "INVALID_PATHS_TYPE")
    if not paths_data:
        raise ValidationError("Image paths array cannot be empty", "EMPTY_PATHS_ARRAY")

    validated_paths: List[str] = []
    for index, path in enumerate(paths_data):
        if not isinstance(path, str):
            raise ValidationError(
                f"Path at index {index} must be a string, "
                f"got {type(path).__name__}",
                "INVALID_PATH_TYPE",
            )
        try:
            validated_paths.append(validate_image_path(path))
        except ValidationError as exc:
            raise ValidationError(
                f"Path at index {index}: {exc.message}",
                exc.error_code,
            ) from exc
    return validated_paths


def validate_json_parameters(params_json: str) -> Dict[str, Any]:
    if not params_json or not params_json.strip():
        return {}

    try:
        params_data = json.loads(params_json.strip())
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON format: {exc}", "INVALID_JSON") from exc

    if not isinstance(params_data, dict):
        raise ValidationError("Parameters must be a JSON object", "INVALID_PARAMS_TYPE")
    return params_data


def validate_spatial_refs_json(spatial_refs: SpatialRefsInput) -> List[SpatialRef]:
    """Parse normalized point or bounding-box references.

    Stringified JSON remains accepted for compatibility, while modern MCP clients
    can pass an array directly.
    """
    if spatial_refs is None:
        return []
    if isinstance(spatial_refs, str):
        if not spatial_refs.strip():
            return []
        try:
            raw_refs = json.loads(spatial_refs)
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"Invalid spatial_refs JSON: {exc}",
                "INVALID_SPATIAL_REFS_JSON",
            ) from exc
    else:
        raw_refs = spatial_refs

    if not isinstance(raw_refs, list):
        raise ValidationError(
            "spatial_refs must be an array",
            "INVALID_SPATIAL_REFS_TYPE",
        )

    validated: List[SpatialRef] = []
    for index, ref in enumerate(raw_refs):
        if not isinstance(ref, list) or len(ref) not in (2, 4):
            raise ValidationError(
                f"spatial_refs[{index}] must contain 2 or 4 coordinates",
                "INVALID_SPATIAL_REF",
            )

        coordinates: SpatialRef = []
        for coordinate in ref:
            if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
                raise ValidationError(
                    f"spatial_refs[{index}] contains a non-numeric coordinate",
                    "INVALID_SPATIAL_REF",
                )
            normalized = float(coordinate)
            if not 0.0 <= normalized <= 1.0:
                raise ValidationError(
                    f"spatial_refs[{index}] coordinates must be between 0 and 1",
                    "INVALID_SPATIAL_REF",
                )
            coordinates.append(normalized)

        if len(coordinates) == 4 and (
            coordinates[2] < coordinates[0] or coordinates[3] < coordinates[1]
        ):
            raise ValidationError(
                f"spatial_refs[{index}] maxima must be greater than minima",
                "INVALID_SPATIAL_REF",
            )
        validated.append(coordinates)
    return validated


def validate_messages_json(messages_input: MessagesInput) -> List[Dict[str, Any]]:
    """Parse an SDK-compatible OpenAI-style chat message list."""
    if isinstance(messages_input, str):
        if not messages_input.strip():
            raise ValidationError("Messages cannot be empty", "EMPTY_MESSAGES")
        try:
            messages = json.loads(messages_input)
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"Invalid messages JSON: {exc}",
                "INVALID_MESSAGES_JSON",
            ) from exc
    else:
        messages = messages_input

    if not isinstance(messages, list) or not messages:
        raise ValidationError(
            "Messages must be a non-empty array",
            "INVALID_MESSAGES_TYPE",
        )

    validated: List[Dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValidationError(
                f"Message at index {index} must be an object",
                "INVALID_MESSAGE",
            )

        role = message.get("role")
        content = message.get("content")
        if role not in ("system", "user", "assistant"):
            raise ValidationError(
                f"Message at index {index} has an invalid role",
                "INVALID_MESSAGE_ROLE",
            )
        if isinstance(content, str):
            normalized_content: Any = sanitize_string(content)
            if not normalized_content:
                raise ValidationError(
                    f"Message at index {index} has empty content",
                    "INVALID_MESSAGE_CONTENT",
                )
        elif isinstance(content, list) and content:
            normalized_content = content
            for part_index, part in enumerate(content):
                if not isinstance(part, dict) or not isinstance(part.get("type"), str):
                    raise ValidationError(
                        f"Message {index} content part {part_index} must be an object "
                        "with a string type",
                        "INVALID_MESSAGE_CONTENT",
                    )
        else:
            raise ValidationError(
                f"Message at index {index} requires non-empty string or list content",
                "INVALID_MESSAGE_CONTENT",
            )

        normalized = dict(message)
        normalized["content"] = normalized_content
        validated.append(normalized)
    return validated


def sanitize_string(
    value: Any,
    max_length: int = 10000,
    allowed_chars: Optional[str] = None,
) -> str:
    if not isinstance(value, str):
        raise ValidationError(
            f"Expected string, got {type(value).__name__}",
            "INVALID_TYPE",
        )

    sanitized = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", value)
    sanitized = sanitized.strip()
    if allowed_chars is not None:
        sanitized = "".join(char for char in sanitized if char in allowed_chars)
    if len(sanitized) > max_length:
        raise ValidationError(
            f"String too long: {len(sanitized)} characters (max {max_length})",
            "STRING_TOO_LONG",
        )
    return sanitized


def _is_http_url(path: str) -> bool:
    try:
        parsed = urlparse(path)
        return parsed.scheme.lower() in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False
