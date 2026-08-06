"""Regression tests for image safety and runtime timeout handling."""

import asyncio
import io
import time

import pytest
from PIL import Image

from moondream_mcp.config import Config
from moondream_mcp.moondream import (
    ImageProcessingError,
    InferenceError,
    MoondreamClient,
)


def image_bytes(image: Image.Image, image_format: str) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_private_network_urls_are_blocked_by_default() -> None:
    client = MoondreamClient(Config())

    with pytest.raises(ImageProcessingError, match="Private-network"):
        await client._validate_remote_url("http://127.0.0.1/image.png")


@pytest.mark.asyncio
async def test_private_network_urls_can_be_explicitly_enabled() -> None:
    client = MoondreamClient(Config(allow_private_network_urls=True))

    await client._validate_remote_url("http://127.0.0.1/image.png")


def test_webp_format_matching_is_case_insensitive() -> None:
    client = MoondreamClient(Config())
    data = image_bytes(Image.new("RGB", (8, 8), "red"), "WEBP")

    decoded = client._decode_image(data, "test.webp")

    assert decoded.mode == "RGB"
    assert decoded.size == (8, 8)


def test_decoded_pixel_limit_blocks_decompression_bombs() -> None:
    client = MoondreamClient(Config(max_image_pixels=64))
    data = image_bytes(Image.new("RGB", (9, 9), "red"), "PNG")

    with pytest.raises(ImageProcessingError, match="81 pixels > 64"):
        client._decode_image(data, "large.png")


def test_transparency_is_composited_on_white() -> None:
    client = MoondreamClient(Config())
    transparent = Image.new("RGBA", (1, 1), (255, 0, 0, 0))

    processed = client._preprocess_image(transparent)

    assert processed.mode == "RGB"
    assert processed.getpixel((0, 0)) == (255, 255, 255)


@pytest.mark.asyncio
async def test_sync_inference_timeout_is_returned_as_inference_error() -> None:
    config = Config()
    config.timeout_seconds = 0.01  # type: ignore[assignment]
    client = MoondreamClient(config)

    def slow_operation() -> str:
        time.sleep(0.05)
        return "done"

    with pytest.raises(InferenceError, match="timed out"):
        await client._run_sync("Test operation", slow_operation)

    await asyncio.sleep(0.06)


@pytest.mark.asyncio
async def test_cleanup_supports_async_sdk_close() -> None:
    class AsyncModel:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    client = MoondreamClient(Config())
    model = AsyncModel()
    client._model = model

    await client.cleanup()

    assert model.closed is True
    assert client._model is None
