"""Concurrency and batch integration tests."""

import asyncio
import json
import time
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock

import pytest

from moondream_mcp.config import Config
from moondream_mcp.models import CaptionLength, CaptionResult
from moondream_mcp.tools.vision import register_vision_tools


def registered_tool(
    mock_mcp: MagicMock,
    name: str,
) -> Callable[..., Any]:
    for call in mock_mcp.tool.return_value.call_args_list:
        if call.args and getattr(call.args[0], "__name__", None) == name:
            return call.args[0]
    raise AssertionError(f"Tool {name} was not registered")


@pytest.fixture
def mock_mcp() -> MagicMock:
    mcp = MagicMock()
    mcp.tool.return_value = MagicMock()
    return mcp


@pytest.fixture
def mock_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def performance_config() -> Config:
    return Config(
        max_batch_size=50,
        batch_concurrency=5,
        max_concurrent_requests=8,
    )


@pytest.mark.asyncio
async def test_batch_processing_metrics(
    mock_mcp: MagicMock,
    mock_client: AsyncMock,
    performance_config: Config,
) -> None:
    mock_client.caption_image.return_value = CaptionResult(
        success=True,
        caption="Test caption",
        length=CaptionLength.NORMAL,
        processing_time_ms=50.0,
    )
    register_vision_tools(mock_mcp, mock_client, performance_config)

    result = json.loads(
        await registered_tool(mock_mcp, "batch_analyze_images")(
            image_paths=json.dumps([f"image-{index}.jpg" for index in range(10)]),
            operation="caption",
        )
    )

    assert result["total_processed"] == 10
    assert result["successful_count"] == 10
    assert result["failed_count"] == 0
    assert result["individual_processing_time_ms"] == 500.0
    assert result["average_time_per_image_ms"] == 50.0
    assert mock_client.caption_image.await_count == 10


@pytest.mark.asyncio
async def test_batch_concurrency_limit(
    mock_mcp: MagicMock,
    mock_client: AsyncMock,
) -> None:
    active = 0
    peak = 0

    async def caption_side_effect(
        image_path: str,
        length: CaptionLength,
        stream: bool,
    ) -> CaptionResult:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return CaptionResult(
            success=True,
            caption=image_path,
            length=length,
            processing_time_ms=10.0,
        )

    mock_client.caption_image.side_effect = caption_side_effect
    config = Config(
        max_batch_size=12,
        batch_concurrency=3,
        max_concurrent_requests=5,
    )
    register_vision_tools(mock_mcp, mock_client, config)

    result = json.loads(
        await registered_tool(mock_mcp, "batch_analyze_images")(
            image_paths=json.dumps([f"image-{index}.jpg" for index in range(12)]),
            operation="caption",
        )
    )

    assert result["successful_count"] == 12
    assert peak == 3


@pytest.mark.asyncio
async def test_direct_concurrent_tool_calls(
    mock_mcp: MagicMock,
    mock_client: AsyncMock,
    performance_config: Config,
) -> None:
    mock_client.caption_image.return_value = CaptionResult(
        success=True,
        caption="Concurrent caption",
        length=CaptionLength.NORMAL,
    )
    register_vision_tools(mock_mcp, mock_client, performance_config)
    caption = registered_tool(mock_mcp, "caption_image")

    started = time.perf_counter()
    results = await asyncio.gather(
        *(caption(f"image-{index}.jpg") for index in range(20))
    )
    elapsed = time.perf_counter() - started

    assert len(results) == 20
    assert all(json.loads(result)["success"] for result in results)
    assert elapsed < 2.0


def test_latest_tool_set_is_registered(
    mock_mcp: MagicMock,
    mock_client: AsyncMock,
    performance_config: Config,
) -> None:
    register_vision_tools(mock_mcp, mock_client, performance_config)
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
