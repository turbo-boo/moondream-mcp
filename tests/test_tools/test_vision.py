"""Tests for Moondream 2.0.1 MCP tools."""

import json
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock

import pytest

from moondream_mcp.config import Config
from moondream_mcp.models import (
    BoundingBox,
    CaptionLength,
    CaptionResult,
    ChatResult,
    DetectedObject,
    DetectionResult,
    Point,
    PointedObject,
    PointingResult,
    QueryResult,
    SegmentResult,
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


def registered_tool(
    mock_mcp: MagicMock,
    name: str,
) -> Callable[..., Any]:
    for call in mock_mcp.tool.return_value.call_args_list:
        if call.args and getattr(call.args[0], "__name__", None) == name:
            return call.args[0]
    raise AssertionError(f"Tool {name} was not registered")


def test_registers_all_tools(
    mock_mcp: MagicMock,
    mock_client: AsyncMock,
) -> None:
    register_vision_tools(mock_mcp, mock_client, Config())
    assert mock_mcp.tool.call_count == 8
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
        "segment_objects",
        "chat_messages",
        "analyze_image",
        "batch_analyze_images",
    }


@pytest.mark.asyncio
async def test_caption_returns_structured_result(
    mock_mcp: MagicMock,
    mock_client: AsyncMock,
) -> None:
    mock_client.caption_image.return_value = CaptionResult(
        success=True,
        caption="A detailed caption",
        length=CaptionLength.DETAILED,
    )
    register_vision_tools(mock_mcp, mock_client, Config())

    result = await registered_tool(mock_mcp, "caption_image")(
        "test.jpg",
        "detailed",
        False,
    )

    assert isinstance(result, dict)
    assert result["caption"] == "A detailed caption"
    mock_client.caption_image.assert_awaited_once_with(
        image_path="test.jpg",
        length=CaptionLength.DETAILED,
        stream=False,
    )


@pytest.mark.asyncio
async def test_query_accepts_native_spatial_refs(
    mock_mcp: MagicMock,
    mock_client: AsyncMock,
) -> None:
    mock_client.query_image.return_value = QueryResult(
        success=True,
        answer="Three people",
        question="How many people?",
        reasoning={"summary": "Counted the referenced region."},
    )
    register_vision_tools(mock_mcp, mock_client, Config())
    refs = [[0.1, 0.2, 0.8, 0.9]]

    result = await registered_tool(mock_mcp, "query_image")(
        "test.jpg",
        "How many people?",
        True,
        True,
        refs,
    )

    assert result["answer"] == "Three people"
    mock_client.query_image.assert_awaited_once_with(
        image_path="test.jpg",
        question="How many people?",
        stream=True,
        reasoning=True,
        spatial_refs=refs,
    )


@pytest.mark.asyncio
async def test_query_keeps_json_string_compatibility(
    mock_mcp: MagicMock,
    mock_client: AsyncMock,
) -> None:
    mock_client.query_image.return_value = QueryResult(
        success=True,
        answer="A car",
        question="What is this?",
    )
    register_vision_tools(mock_mcp, mock_client, Config())

    await registered_tool(mock_mcp, "query_image")(
        "test.jpg",
        "What is this?",
        False,
        False,
        "[[0.5, 0.5]]",
    )

    mock_client.query_image.assert_awaited_once_with(
        image_path="test.jpg",
        question="What is this?",
        stream=False,
        reasoning=False,
        spatial_refs=[[0.5, 0.5]],
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

    result = await registered_tool(mock_mcp, "detect_objects")(
        "test.jpg",
        "person",
    )

    assert result["objects"][0]["bounding_box"] == {
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

    result = await registered_tool(mock_mcp, "point_objects")(
        "test.jpg",
        "car",
    )

    assert result["points"][0]["point"] == {"x": 0.5, "y": 0.3}
    assert result["points"][0]["confidence"] is None


@pytest.mark.asyncio
async def test_segment_tool_accepts_native_refs(
    mock_mcp: MagicMock,
    mock_client: AsyncMock,
) -> None:
    mock_client.segment_objects.return_value = SegmentResult(
        success=True,
        object_name="person",
        path="M 0 0 L 1 1",
        bounding_box=BoundingBox(
            x_min=0.1,
            y_min=0.2,
            x_max=0.8,
            y_max=0.9,
        ),
    )
    register_vision_tools(mock_mcp, mock_client, Config())

    result = await registered_tool(mock_mcp, "segment_objects")(
        "test.jpg",
        "person",
        [[0.5, 0.5]],
        True,
    )

    assert result["path"] == "M 0 0 L 1 1"
    mock_client.segment_objects.assert_awaited_once_with(
        image_path="test.jpg",
        object_name="person",
        spatial_refs=[[0.5, 0.5]],
        stream=True,
    )


@pytest.mark.asyncio
async def test_chat_tool_accepts_native_messages(
    mock_mcp: MagicMock,
    mock_client: AsyncMock,
) -> None:
    mock_client.chat_messages.return_value = ChatResult(
        success=True,
        message={
            "role": "assistant",
            "content": "Hello.",
        },
    )
    register_vision_tools(mock_mcp, mock_client, Config())
    messages = [{"role": "user", "content": "Hello?"}]

    result = await registered_tool(mock_mcp, "chat_messages")(
        messages,
        False,
        True,
    )

    assert result["message"]["content"] == "Hello."
    mock_client.chat_messages.assert_awaited_once_with(
        messages=messages,
        stream=False,
        reasoning=True,
    )


@pytest.mark.asyncio
async def test_chat_keeps_json_string_compatibility(
    mock_mcp: MagicMock,
    mock_client: AsyncMock,
) -> None:
    mock_client.chat_messages.return_value = ChatResult(
        success=True,
        message={"role": "assistant", "content": "Hello."},
    )
    register_vision_tools(mock_mcp, mock_client, Config())
    messages = [{"role": "user", "content": "Hello?"}]

    await registered_tool(mock_mcp, "chat_messages")(json.dumps(messages))

    mock_client.chat_messages.assert_awaited_once_with(
        messages=messages,
        stream=False,
        reasoning=None,
    )


@pytest.mark.asyncio
async def test_analyze_routes_segment(
    mock_mcp: MagicMock,
    mock_client: AsyncMock,
) -> None:
    mock_client.segment_objects.return_value = SegmentResult(
        success=True,
        object_name="car",
        path="M 0 0",
    )
    register_vision_tools(mock_mcp, mock_client, Config())

    result = await registered_tool(mock_mcp, "analyze_image")(
        image_path="test.jpg",
        operation="segment",
        object_name="car",
        spatial_refs=[[0.5, 0.5]],
    )

    assert result["path"] == "M 0 0"
    mock_client.segment_objects.assert_awaited_once_with(
        image_path="test.jpg",
        object_name="car",
        spatial_refs=[[0.5, 0.5]],
        stream=False,
    )


@pytest.mark.asyncio
async def test_model_load_error_keeps_specific_code(
    mock_mcp: MagicMock,
    mock_client: AsyncMock,
) -> None:
    mock_client.caption_image.side_effect = ModelLoadError("Model failed to load")
    register_vision_tools(mock_mcp, mock_client, Config())

    result = await registered_tool(mock_mcp, "caption_image")(
        "test.jpg",
        "normal",
        False,
    )

    assert result["success"] is False
    assert result["error_code"] == "MODEL_LOAD_ERROR"
    assert result["error_message"] == "Model failed to load"


@pytest.mark.asyncio
async def test_invalid_spatial_refs_return_validation_error(
    mock_mcp: MagicMock,
    mock_client: AsyncMock,
) -> None:
    register_vision_tools(mock_mcp, mock_client, Config())

    result = await registered_tool(mock_mcp, "query_image")(
        "test.jpg",
        "What is this?",
        False,
        False,
        [[2, 0.5]],
    )

    assert result["success"] is False
    assert result["error_code"] == "INVALID_SPATIAL_REF"


@pytest.mark.asyncio
async def test_batch_accepts_native_paths_and_reports_partial_failure(
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

    result = await registered_tool(mock_mcp, "batch_analyze_images")(
        image_paths=["one.jpg", "bad.jpg", "two.jpg"],
        operation="caption",
    )

    assert result["success"] is False
    assert result["partial_success"] is True
    assert result["total_processed"] == 3
    assert result["successful_count"] == 2
    assert result["failed_count"] == 1
    assert result["results"][1]["success"] is False


@pytest.mark.asyncio
async def test_batch_keeps_json_string_compatibility(
    mock_mcp: MagicMock,
    mock_client: AsyncMock,
) -> None:
    mock_client.caption_image.return_value = CaptionResult(
        success=True,
        caption="ok",
        length=CaptionLength.NORMAL,
    )
    register_vision_tools(mock_mcp, mock_client, Config())

    result = await registered_tool(mock_mcp, "batch_analyze_images")(
        image_paths=json.dumps(["one.jpg"]),
        operation="caption",
    )

    assert result["success"] is True
    assert result["partial_success"] is False


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

    result = await registered_tool(mock_mcp, "batch_analyze_images")(
        image_paths=["one.jpg", "two.jpg"],
        operation="caption",
    )

    assert result["success"] is False
    assert result["error_code"] == "BATCH_SIZE_EXCEEDED"
