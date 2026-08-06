"""Validation utilities for moondream-mcp."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from .models import CaptionLength, SpatialRef


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
    if _is_url(image_path):
        return image_path

    try:
        return str(Path(image_path).expanduser())
    except Exception as exc:
        raise ValidationError(f"Invalid file path: {exc}", "INVALID_PATH") from exc


def validate_question(question: str) -> str:
    if not question or not question.strip():
        raise ValidationError("Question cannot be empty", "EMPTY_QUESTION")

    question = question.strip()
    if len(question) > 1000:
        raise ValidationError(
            f"Question too long: {len(question)} characters (max 1000)",
            "QUESTION_TOO_LONG",
        )
    return question


def validate_object_name(object_name: str) -> str:
    if not object_name or not object_name.strip():
        raise ValidationError("Object name cannot be empty", "EMPTY_OBJECT_NAME")

    object_name = object_name.strip()
    if len(object_name) > 100:
        raise ValidationError(
            f"Object name too long: {len(object_name)} characters (max 100)",
            "OBJECT_NAME_TOO_LONG",
        )

    dangerous_chars = ["<", ">", '"', "'"]
    if any(char in object_name for char in dangerous_chars):
        raise ValidationError(
            f"Object name contains invalid characters: {object_name}",
            "INVALID_OBJECT_NAME",
        )
    return object_name


def validate_caption_length(length: str) -> CaptionLength:
    try:
        return CaptionLength(length.lower())
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


def validate_image_paths_list(paths_json: str) -> List[str]:
    if not paths_json or not paths_json.strip():
        raise ValidationError("Image paths cannot be empty", "EMPTY_PATHS")

    try:
        paths_data = json.loads(paths_json.strip())
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON format: {exc}", "INVALID_JSON") from exc

    if not isinstance(paths_data, list):
        raise ValidationError("Image paths must be a JSON array", "INVALID_PATHS_TYPE")
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


def validate_spatial_refs_json(spatial_refs_json: str) -> List[SpatialRef]:
    """Parse normalized point or bounding-box references from JSON."""
    if not spatial_refs_json or not spatial_refs_json.strip():
        return []

    try:
        raw_refs = json.loads(spatial_refs_json)
    except json.JSONDecodeError as exc:
        raise ValidationError(
            f"Invalid spatial_refs JSON: {exc}",
            "INVALID_SPATIAL_REFS_JSON",
        ) from exc

    if not isinstance(raw_refs, list):
        raise ValidationError(
            "spatial_refs must be a JSON array",
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
            if isinstance(coordinate, bool) or not isinstance(
                coordinate, (int, float)
            ):
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


def validate_messages_json(messages_json: str) -> List[Dict[str, Any]]:
    """Parse an SDK-compatible chat message list."""
    if not messages_json or not messages_json.strip():
        raise ValidationError("Messages cannot be empty", "EMPTY_MESSAGES")

    try:
        messages = json.loads(messages_json)
    except json.JSONDecodeError as exc:
        raise ValidationError(
            f"Invalid messages JSON: {exc}",
            "INVALID_MESSAGES_JSON",
        ) from exc

    if not isinstance(messages, list) or not messages:
        raise ValidationError(
            "Messages must be a non-empty JSON array",
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
        if not isinstance(content, (str, list)):
            raise ValidationError(
                f"Message at index {index} requires string or list content",
                "INVALID_MESSAGE_CONTENT",
            )
        validated.append(dict(message))
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
    if len(sanitized) > max_length:
        raise ValidationError(
            f"String too long: {len(sanitized)} characters (max {max_length})",
            "STRING_TOO_LONG",
        )
    return sanitized


def _is_url(path: str) -> bool:
    try:
        parsed = urlparse(path)
        return bool(parsed.scheme and parsed.netloc)
    except Exception:
        return False
