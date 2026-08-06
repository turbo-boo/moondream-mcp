"""Tests for Moondream 3.1 MCP tools."""

import json
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock

import pytest

from moondream_mcp.config import Config
from moondream_mcp.models import (
    BoundingBox,
    CaptionLength,
    CaptionResult,
    DetectedObject,
    DetectionResult,
    Point,
    PointedObject,
    PointingResult,
    QueryResult,
)
from moondream_mcp.moondream import ModelLoadError
from moondream_mcp.tools.vision import register_vision_tools


@pytest.fixture
def mock_mcp() -> MagicMock:
    mcp = MagicMock()
    mcp.tool.return_value = MagicMock()
    return mcp


@pytest.fixture
def mock_client() -> AsyncMock:
    return AsyncMock()


def registered_tool(mock_mcp: MagicMock, name: str) -> Callable[..., Any]:
    for call in mock_mcp.tool.return_value.call_args_list:
        if call.args and getattr(call.args[0], "__name__", None) == name:
            return call.args[0]
    raise AssertionError(f"Tool {name} was not registered")


def test_registers_all_tools(
    mock_mcp: MagicMock,
    mock_client: AsyncMock,
) -> None:
    register_vision_tools(mock_mcp, mock_client, Config())
    assert mock_mcp.tool.call_count == 6
    names = {
        call.args[0].__name__
        for call in mock_mcp.tool.return_value.call_args_list
        if call.args
    }
    assert names == {
        "caption_image",
        "query_image",
        "detect_objects",
        "point_objects",
        "analyze_image",
        "batch_analyze_images",
    }


@pytest.mark.asyncio
async def test_caption_accepts_detailed_alias(
    mock_mcp: MagicMock,
    mock_client: AsyncMock,
) -> None:
    mock_client.caption_image.return_value = CaptionResult(
        success=True,
        caption="A detailed caption",
        length=CaptionLength.DETAILED,
    )
    register_vision_tools(mock_mcp, mock_client, Config())

    result = json.loads(
        await registered_tool(mock_mcp, "caption_image")(
            "test.jpg",
            "detailed",
            False,
        )
    )

    assert result["success"] is True
    assert result["caption"] == "A detailed caption"
    mock_client.caption_image.assert_awaited_once_with(
        image_path="test.jpg",
        length=CaptionLength.DETAILED,
        stream=False,
    )


@pytest.mark.asyncio
async def test_query_passes_stream_flag(
    mock_mcp: MagicMock,
    mock_client: AsyncMock,
) -> None:
    mock_client.query_image.return_value = QueryResult(
        success=True,
        answer="Three people",
        question="How many people?",
    )
    register_vision_tools(mock_mcp, mock_client, Config())

    result = json.loads(
        await registered_tool(mock_mcp, "query_image")(
            "test.jpg",
            "How many people?",
            True,
        )
    )

    assert result["answer"] == "Three people"
    mock_client.query_image.assert_awaited_once_with(
        image_path="test.jpg",
        question="How many people?",
        stream=True,
    )


@pytest.mark.asyncio
async def test_detect_serializes_native_box_schema(
    mock_mcp: MagicMock,
    mock_client: AsyncMock,
) -> None:
    mock_client.detect_objects.return_value = DetectionResult(
        success=True,
        objects=[
            DetectedObject(
                name="person",
                confidence=None,
                bounding_box=BoundingBox(
                    x_min=0.1,
                    y_min=0.2,
                    x_max=0.4,
                    y_max=0.8,
                ),
            )
        ],
        object_name="person",
        total_found=1,
    )
    register_vision_tools(mock_mcp, mock_client, Config())

    result = json.loads(
        await registered_tool(mock_mcp, "detect_objects")(
            "test.jpg",
            "person",
        )
    )

    box = result["objects"][0]["bounding_box"]
    assert box == {
        "x_min": 0.1,
        "y_min": 0.2,
        "x_max": 0.4,
        "y_max": 0.8,
    }
    assert result["objects"][0]["confidence"] is None


@pytest.mark.asyncio
async def test_point_preserves_missing_confidence(
    mock_mcp: MagicMock,
    mock_client: AsyncMock,
) -> None:
    mock_client.point_objects.return_value = PointingResult(
        success=True,
        points=[
            PointedObject(
                name="car",
                confidence=None,
                point=Point(x=0.5, y=0.3),
            )
        ],
        object_name="car",
        total_found=1,
    )
    register_vision_tools(mock_mcp, mock_client, Config())

    result = json.loads(
        await registered_tool(mock_mcp, "point_objects")(
            "test.jpg",
            "car",
        )
    )

    assert result["points"][0]["point"] == {"x": 0.5, "y": 0.3}
    assert result["points"][0]["confidence"] is None


@pytest.mark.asyncio
async def test_analyze_routes_validated_caption_length(
    mock_mcp: MagicMock,
    mock_client: AsyncMock,
) -> None:
    mock_client.caption_image.return_value = CaptionResult(
        success=True,
        caption="Caption",
        length=CaptionLength.LONG,
    )
    register_vision_tools(mock_mcp, mock_client, Config())

    result = json.loads(
        await registered_tool(mock_mcp, "analyze_image")(
            image_path="test.jpg",
            operation="caption",
            length="long",
        )
    )

    assert result["caption"] == "Caption"
    mock_client.caption_image.assert_awaited_once_with(
        image_path="test.jpg",
        length=CaptionLength.LONG,
        stream=False,
    )


@pytest.mark.asyncio
async def test_model_load_error_keeps_specific_code(
    mock_mcp: MagicMock,
    mock_client: AsyncMock,
) -> None:
    mock_client.caption_image.side_effect = ModelLoadError("Model failed to load")
    register_vision_tools(mock_mcp, mock_client, Config())

    result = json.loads(
        await registered_tool(mock_mcp, "caption_image")(
            "test.jpg",
            "normal",
            False,
        )
    )

    assert result["success"] is False
    assert result["error_code"] == "MODEL_LOAD_ERROR"
    assert result["error_message"] == "Model failed to load"


@pytest.mark.asyncio
async def test_batch_isolates_errors(
    mock_mcp: MagicMock,
    mock_client: AsyncMock,
) -> None:
    async def caption_side_effect(
        image_path: str,
        length: CaptionLength,
        stream: bool,
    ) -> CaptionResult:
        if image_path == "bad.jpg":
            raise RuntimeError("bad image")
        return CaptionResult(
            success=True,
            caption=image_path,
            length=length,
            processing_time_ms=10.0,
        )

    mock_client.caption_image.side_effect = caption_side_effect
    config = Config(max_batch_size=3, batch_concurrency=2)
    register_vision_tools(mock_mcp, mock_client, config)

    result = json.loads(
        await registered_tool(mock_mcp, "batch_analyze_images")(
            image_paths=json.dumps(["one.jpg", "bad.jpg", "two.jpg"]),
            operation="caption",
        )
    )

    assert result["total_processed"] == 3
    assert result["successful_count"] == 2
    assert result["failed_count"] == 1
    assert result["results"][1]["success"] is False


@pytest.mark.asyncio
async def test_batch_limit(
    mock_mcp: MagicMock,
    mock_client: AsyncMock,
) -> None:
    register_vision_tools(
        mock_mcp,
        mock_client,
        Config(max_batch_size=1, batch_concurrency=1),
    )

    result = json.loads(
        await registered_tool(mock_mcp, "batch_analyze_images")(
            image_paths=json.dumps(["one.jpg", "two.jpg"]),
            operation="caption",
        )
    )

    assert result["success"] is False
    assert result["error_code"] == "BATCH_SIZE_EXCEEDED"
