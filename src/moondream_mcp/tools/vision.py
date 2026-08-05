"""FastMCP tool registration for Moondream vision operations."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from moondream_mcp.models import (
    CaptionResult,
    DetectionResult,
    PointingResult,
    QueryResult,
)
from moondream_mcp.moondream import ImageProcessingError, ModelLoadError
from moondream_mcp.validation import (
    ValidationError,
    validate_caption_length,
    validate_image_path,
    validate_image_paths_list,
    validate_object_name,
    validate_operation,
    validate_question,
)

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from ..config import Config
    from ..moondream import MoondreamClient

SingleResult = Union[CaptionResult, QueryResult, DetectionResult, PointingResult]


async def _route_single_operation(
    client: "MoondreamClient",
    operation: str,
    image_path: str,
    params: Dict[str, Any],
) -> SingleResult:
    if operation == "caption":
        return await client.caption_image(
            image_path=image_path,
            length=validate_caption_length(params.get("length", "normal")),
            stream=bool(params.get("stream", False)),
        )

    if operation == "query":
        question = params.get("question")
        if not question:
            raise ValidationError(
                "question parameter is required for query operation",
                "MISSING_QUESTION",
            )
        return await client.query_image(
            image_path=image_path,
            question=validate_question(question),
            stream=bool(params.get("stream", False)),
        )

    if operation == "detect":
        object_name = params.get("object_name")
        if not object_name:
            raise ValidationError(
                "object_name parameter is required for detect operation",
                "MISSING_OBJECT_NAME",
            )
        return await client.detect_objects(
            image_path=image_path,
            object_name=validate_object_name(object_name),
        )

    if operation == "point":
        object_name = params.get("object_name")
        if not object_name:
            raise ValidationError(
                "object_name parameter is required for point operation",
                "MISSING_OBJECT_NAME",
            )
        return await client.point_objects(
            image_path=image_path,
            object_name=validate_object_name(object_name),
        )

    raise ValidationError(f"Unknown operation: {operation}", "INVALID_OPERATION")


def _create_error_response_dict(
    error: Exception,
    operation: str,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from .utils import get_error_code_for_exception

    if isinstance(error, ValidationError):
        error_code = error.error_code
        error_message = error.message
    elif isinstance(error, (ModelLoadError, ImageProcessingError)):
        error_code = getattr(error, "error_code", "PROCESSING_ERROR")
        error_message = str(error)
    elif isinstance(error, FileNotFoundError):
        error_code = "FILE_NOT_FOUND"
        error_message = f"Image file not found: {error}"
    elif isinstance(error, PermissionError):
        error_code = "PERMISSION_DENIED"
        error_message = f"Permission denied accessing image: {error}"
    else:
        error_code = get_error_code_for_exception(error)
        error_message = f"Unexpected error: {error}"

    return {
        "success": False,
        "error_message": error_message,
        "error_code": error_code,
        "error_context": context or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": operation,
    }


def _create_error_response(
    error: Exception,
    operation: str,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    return json.dumps(
        _create_error_response_dict(error, operation, context),
        indent=2,
    )


def _operation_params(
    operation: str,
    *,
    question: str,
    object_name: str,
    length: str,
    stream: bool,
) -> Dict[str, Any]:
    if operation == "caption":
        return {"length": length, "stream": stream}
    if operation == "query":
        if not question.strip():
            raise ValidationError(
                "question parameter is required for query operation",
                "MISSING_QUESTION",
            )
        return {"question": question, "stream": stream}
    if operation in ("detect", "point"):
        if not object_name.strip():
            raise ValidationError(
                f"object_name parameter is required for {operation} operation",
                "MISSING_OBJECT_NAME",
            )
        return {"object_name": object_name}
    raise ValidationError(f"Unknown operation: {operation}", "INVALID_OPERATION")


def register_vision_tools(
    mcp: "FastMCP",
    moondream_client: "MoondreamClient",
    config: Optional["Config"] = None,
) -> None:
    from ..config import Config

    if config is None:
        config = Config.from_env()

    @mcp.tool()
    async def caption_image(
        image_path: str,
        length: str = "normal",
        stream: bool = False,
    ) -> str:
        """Generate a caption.

        `length` accepts `short`, `normal`, or `long`. `detailed` remains an
        alias for `long` for compatibility with moondream-mcp 1.x.
        """
        try:
            result = await moondream_client.caption_image(
                image_path=validate_image_path(image_path),
                length=validate_caption_length(length),
                stream=stream,
            )
            return result.model_dump_json(indent=2)
        except Exception as error:
            return _create_error_response(
                error,
                "caption",
                {"image_path": image_path, "length": length, "stream": stream},
            )

    @mcp.tool()
    async def query_image(
        image_path: str,
        question: str,
        stream: bool = False,
    ) -> str:
        """Ask a natural-language question about an image."""
        try:
            result = await moondream_client.query_image(
                image_path=validate_image_path(image_path),
                question=validate_question(question),
                stream=stream,
            )
            return result.model_dump_json(indent=2)
        except Exception as error:
            return _create_error_response(
                error,
                "query",
                {
                    "image_path": image_path,
                    "question": question,
                    "stream": stream,
                },
            )

    @mcp.tool()
    async def detect_objects(image_path: str, object_name: str) -> str:
        """Detect objects and return normalized min/max bounding boxes."""
        try:
            result = await moondream_client.detect_objects(
                image_path=validate_image_path(image_path),
                object_name=validate_object_name(object_name),
            )
            return result.model_dump_json(indent=2)
        except Exception as error:
            return _create_error_response(
                error,
                "detect",
                {"image_path": image_path, "object_name": object_name},
            )

    @mcp.tool()
    async def point_objects(image_path: str, object_name: str) -> str:
        """Locate matching objects and return normalized x/y points."""
        try:
            result = await moondream_client.point_objects(
                image_path=validate_image_path(image_path),
                object_name=validate_object_name(object_name),
            )
            return result.model_dump_json(indent=2)
        except Exception as error:
            return _create_error_response(
                error,
                "point",
                {"image_path": image_path, "object_name": object_name},
            )

    @mcp.tool()
    async def analyze_image(
        image_path: str,
        operation: str,
        question: str = "",
        object_name: str = "",
        length: str = "normal",
        stream: bool = False,
    ) -> str:
        """Run one of caption, query, detect, or point on an image."""
        try:
            validated_operation = validate_operation(operation)
            result = await _route_single_operation(
                client=moondream_client,
                operation=validated_operation,
                image_path=validate_image_path(image_path),
                params=_operation_params(
                    validated_operation,
                    question=question,
                    object_name=object_name,
                    length=length,
                    stream=stream,
                ),
            )
            return result.model_dump_json(indent=2)
        except Exception as error:
            return _create_error_response(
                error,
                operation,
                {
                    "image_path": image_path,
                    "question": question,
                    "object_name": object_name,
                    "length": length,
                    "stream": stream,
                },
            )

    @mcp.tool()
    async def batch_analyze_images(
        image_paths: str,
        operation: str,
        question: str = "",
        object_name: str = "",
        length: str = "normal",
        stream: bool = False,
    ) -> str:
        """Run one operation on a JSON array of image paths."""
        started = time.perf_counter()
        try:
            validated_operation = validate_operation(operation)
            validated_paths = validate_image_paths_list(image_paths)
            if len(validated_paths) > config.max_batch_size:
                raise ValidationError(
                    f"Batch size {len(validated_paths)} exceeds maximum "
                    f"allowed {config.max_batch_size}",
                    "BATCH_SIZE_EXCEEDED",
                )

            params = _operation_params(
                validated_operation,
                question=question,
                object_name=object_name,
                length=length,
                stream=stream,
            )
            semaphore = asyncio.Semaphore(config.batch_concurrency)

            async def process(path: str) -> Dict[str, Any]:
                async with semaphore:
                    try:
                        result = await _route_single_operation(
                            moondream_client,
                            validated_operation,
                            path,
                            params,
                        )
                        return result.model_dump()
                    except Exception as error:
                        return _create_error_response_dict(
                            error,
                            validated_operation,
                            {"image_path": path},
                        )

            results = await asyncio.gather(*(process(path) for path in validated_paths))
            successful_count = sum(
                bool(result.get("success", False)) for result in results
            )
            individual_time = sum(
                float(result.get("processing_time_ms") or 0.0)
                for result in results
            )
            total_time = (time.perf_counter() - started) * 1000
            batch_result = {
                "success": True,
                "operation": validated_operation,
                "total_processed": len(results),
                "successful_count": successful_count,
                "failed_count": len(results) - successful_count,
                "results": results,
                "batch_processing_time_ms": total_time,
                "individual_processing_time_ms": individual_time,
                "average_time_per_image_ms": (
                    individual_time / len(results) if results else 0.0
                ),
                "metadata": {
                    "batch_size": len(results),
                    "concurrency": config.batch_concurrency,
                    "operation_params": params,
                },
            }
            return json.dumps(batch_result, indent=2)
        except Exception as error:
            return _create_error_response(
                error,
                f"batch_{operation}",
                {
                    "image_paths": image_paths,
                    "question": question,
                    "object_name": object_name,
                    "length": length,
                    "stream": stream,
                },
            )
