"""Utility functions for vision analysis tools."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Union

from moondream_mcp.models import (
    CaptionLength,
    CaptionResult,
    ChatResult,
    DetectionResult,
    PointingResult,
    QueryResult,
    SegmentResult,
)

FormattedResult = Union[
    CaptionResult,
    QueryResult,
    DetectionResult,
    PointingResult,
    SegmentResult,
    ChatResult,
]


def create_error_response(
    error_code: str,
    error_message: str,
    operation: Optional[str] = None,
    image_path: Optional[str] = None,
) -> str:
    error_response: Dict[str, Any] = {
        "success": False,
        "error_code": error_code,
        "error_message": error_message,
        "timestamp": time.time(),
    }
    if operation:
        error_response["operation"] = operation
    if image_path:
        error_response["image_path"] = image_path
    return json.dumps(error_response, indent=2)


def validate_caption_length(length: str) -> CaptionLength:
    try:
        return CaptionLength(length.lower())
    except (AttributeError, ValueError) as exc:
        raise ValueError(
            f"Invalid length '{length}'. Must be 'short', 'normal', "
            "'long', or the compatibility alias 'detailed'"
        ) from exc


def validate_operation(operation: str) -> str:
    valid_operations = {
        "caption",
        "query",
        "detect",
        "point",
        "segment",
    }
    if operation not in valid_operations:
        raise ValueError(
            f"Invalid operation '{operation}'. Must be one of: "
            f"{', '.join(sorted(valid_operations))}"
        )
    return operation


def parse_json_parameters(parameters: str) -> Dict[str, Any]:
    try:
        result = json.loads(parameters)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON parameters: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError("Parameters must be a JSON object")
    return result


def parse_image_paths(image_paths_json: str) -> List[str]:
    try:
        paths = json.loads(image_paths_json)
    except json.JSONDecodeError as exc:
        raise ValueError("image_paths must be valid JSON array") from exc

    if not isinstance(paths, list):
        raise ValueError("image_paths must be a JSON array")
    if not paths:
        raise ValueError("image_paths cannot be empty")
    if len(paths) > 10:
        raise ValueError("Cannot process more than 10 images at once")
    if any(not isinstance(path, str) for path in paths):
        raise ValueError("Every image path must be a string")
    return paths


def format_result_as_json(result: FormattedResult) -> str:
    return result.model_dump_json(indent=2)


def create_batch_summary(
    results: List[Dict[str, Any]],
    operation: str,
    total_time_ms: float,
) -> Dict[str, Any]:
    successful = [result for result in results if result.get("success", False)]
    failed = [result for result in results if not result.get("success", False)]
    return {
        "operation": operation,
        "total_processed": len(results),
        "total_successful": len(successful),
        "total_failed": len(failed),
        "total_processing_time_ms": total_time_ms,
        "average_time_per_image_ms": (total_time_ms / len(results) if results else 0),
        "results": results,
    }


def sanitize_error_message(error: Exception) -> str:
    error_str = str(error)
    error_str = re.sub(r"/[^\s]*", "[PATH]", error_str)
    error_str = re.sub(r"C:\\[^\s]*", "[PATH]", error_str)
    error_str = re.sub(r"https?://[^\s]+", "[URL]", error_str)
    if len(error_str) > 200:
        return error_str[:197] + "..."
    return error_str


def validate_input_parameters(
    image_path: str,
    operation: Optional[str] = None,
    question: Optional[str] = None,
    object_name: Optional[str] = None,
) -> None:
    if not image_path or not image_path.strip():
        raise ValueError("image_path cannot be empty")
    if question is not None and not question.strip():
        raise ValueError("question cannot be empty")
    if object_name is not None and not object_name.strip():
        raise ValueError("object_name cannot be empty")
    if operation is not None:
        validate_operation(operation)


def get_error_code_for_exception(error: Exception) -> str:
    from moondream_mcp.moondream import (
        ImageProcessingError,
        InferenceError,
        ModelLoadError,
    )

    if isinstance(error, ModelLoadError):
        return "MODEL_LOAD_ERROR"
    if isinstance(error, ImageProcessingError):
        return "IMAGE_PROCESSING_ERROR"
    if isinstance(error, InferenceError):
        return "INFERENCE_ERROR"
    if isinstance(error, ValueError):
        return "INVALID_REQUEST"
    if isinstance(error, FileNotFoundError):
        return "FILE_NOT_FOUND"
    if isinstance(error, PermissionError):
        return "PERMISSION_DENIED"
    return "UNKNOWN_ERROR"


def measure_time_ms(start_time: float) -> float:
    return (time.time() - start_time) * 1000
