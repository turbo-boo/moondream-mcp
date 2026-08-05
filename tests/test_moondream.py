"""Tests for the Moondream 3.1 SDK adapter."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from moondream_mcp.config import Config
from moondream_mcp.models import BoundingBox, CaptionLength
from moondream_mcp.moondream import (
    ImageProcessingError,
    ModelLoadError,
    MoondreamClient,
)


class TestMoondreamClient:
    @pytest.fixture
    def config(self) -> Config:
        return Config(
            backend="photon",
            device="cuda",
            device_auto_detect=False,
            max_image_size=(512, 512),
            max_file_size_mb=10,
        )

    @pytest.fixture
    def client(self, config: Config) -> MoondreamClient:
        return MoondreamClient(config)

    @pytest.fixture
    def sample_image(self) -> Image.Image:
        return Image.new("RGB", (100, 100), color="red")

    def test_client_initialization(self, client: MoondreamClient) -> None:
        assert client._model is None
        assert client._tokenizer is None
        assert client._device is None
        assert client._session is None

    def test_bounding_box_accepts_native_and_legacy(self) -> None:
        native = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.8, y_max=0.9)
        assert native.width == pytest.approx(0.7)
        legacy = BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4)
        assert legacy.x_max == pytest.approx(0.4)
        assert legacy.y_max == pytest.approx(0.6)

    def test_detailed_maps_to_long(self) -> None:
        assert CaptionLength.DETAILED.sdk_value == "long"
        assert CaptionLength.LONG.sdk_value == "long"

    @pytest.mark.asyncio
    async def test_context_manager(self, client: MoondreamClient) -> None:
        async with client:
            assert client._session is not None

    @pytest.mark.asyncio
    async def test_cleanup_closes_sdk_model(
        self, client: MoondreamClient
    ) -> None:
        mock_session = AsyncMock()
        mock_model = MagicMock()
        client._session = mock_session
        client._model = mock_model

        await client.cleanup()

        mock_session.close.assert_awaited_once()
        mock_model.close.assert_called_once()
        assert client._model is None
        assert client._session is None

    def test_is_url(self, client: MoondreamClient) -> None:
        assert client._is_url("https://example.com/image.jpg")
        assert client._is_url("http://example.com/image.jpg")
        assert not client._is_url("ftp://example.com/image.jpg")
        assert not client._is_url("/path/to/image.jpg")

    def test_preprocess_image(
        self, client: MoondreamClient, sample_image: Image.Image
    ) -> None:
        grayscale = Image.new("L", (100, 100), color=128)
        assert client._preprocess_image(grayscale).mode == "RGB"

        large = Image.new("RGB", (1000, 1000), color="blue")
        processed = client._preprocess_image(large)
        assert processed.width <= 512
        assert processed.height <= 512

    @pytest.mark.asyncio
    async def test_load_image_from_file_not_found(
        self, client: MoondreamClient
    ) -> None:
        with pytest.raises(ImageProcessingError, match="Image file not found"):
            await client._load_image_from_file("/nonexistent/path.jpg")

    @pytest.mark.asyncio
    async def test_load_image_from_file_success(
        self,
        client: MoondreamClient,
        sample_image: Image.Image,
        tmp_path: Path,
    ) -> None:
        image_path = tmp_path / "test.jpg"
        sample_image.save(image_path)
        result = await client._load_image_from_file(str(image_path))
        assert result.mode == "RGB"

    @pytest.mark.asyncio
    async def test_load_photon_model(self, client: MoondreamClient) -> None:
        sdk_model = MagicMock()
        with patch("moondream_mcp.moondream.md.photon", return_value=sdk_model):
            await client._load_model()
        assert client._model is sdk_model

    @pytest.mark.asyncio
    async def test_load_cloud_model(self) -> None:
        config = Config(
            backend="cloud",
            api_key="test-key",
            device="cpu",
        )
        client = MoondreamClient(config)
        sdk_model = MagicMock()
        with patch("moondream_mcp.moondream.md.vl", return_value=sdk_model) as factory:
            await client._load_model()
        factory.assert_called_once_with(
            api_key="test-key",
            model="moondream3.1-9B-A2B",
        )

    @pytest.mark.asyncio
    async def test_model_loading_error(self, client: MoondreamClient) -> None:
        with patch(
            "moondream_mcp.moondream.md.photon",
            side_effect=RuntimeError("Model not found"),
        ):
            with pytest.raises(ModelLoadError, match="Model not found"):
                await client._load_model()

    @pytest.mark.asyncio
    async def test_caption_uses_long_sdk_value(
        self,
        client: MoondreamClient,
        sample_image: Image.Image,
    ) -> None:
        model = MagicMock()
        model.caption.return_value = {"caption": "A red square"}
        client._model = model

        with patch.object(client, "_load_image", return_value=sample_image):
            result = await client.caption_image(
                "test.jpg",
                CaptionLength.DETAILED,
            )

        assert result.success
        assert result.caption == "A red square"
        model.caption.assert_called_once_with(
            sample_image,
            length="long",
            stream=False,
        )

    @pytest.mark.asyncio
    async def test_streaming_caption(
        self,
        client: MoondreamClient,
        sample_image: Image.Image,
    ) -> None:
        model = MagicMock()
        model.caption.return_value = {"caption": iter(["A ", "red ", "square"])}
        client._model = model

        with patch.object(client, "_load_image", return_value=sample_image):
            result = await client.caption_image("test.jpg", stream=True)

        assert result.caption == "A red square"

    @pytest.mark.asyncio
    async def test_query_streaming(
        self,
        client: MoondreamClient,
        sample_image: Image.Image,
    ) -> None:
        model = MagicMock()
        model.query.return_value = {"answer": iter(["Yes", "."])}
        client._model = model

        with patch.object(client, "_load_image", return_value=sample_image):
            result = await client.query_image(
                "test.jpg",
                "Is it red?",
                stream=True,
            )

        assert result.answer == "Yes."

    @pytest.mark.asyncio
    async def test_native_detection_schema(
        self,
        client: MoondreamClient,
        sample_image: Image.Image,
    ) -> None:
        model = MagicMock()
        model.detect.return_value = {
            "objects": [
                {
                    "x_min": 0.1,
                    "y_min": 0.2,
                    "x_max": 0.8,
                    "y_max": 0.9,
                }
            ]
        }
        client._model = model

        with patch.object(client, "_load_image", return_value=sample_image):
            result = await client.detect_objects("test.jpg", "square")

        assert result.success
        assert result.total_found == 1
        assert result.objects[0].confidence is None
        box = result.objects[0].bounding_box
        assert box.x_min == 0.1
        assert box.x_max == 0.8

    @pytest.mark.asyncio
    async def test_point_schema(
        self,
        client: MoondreamClient,
        sample_image: Image.Image,
    ) -> None:
        model = MagicMock()
        model.point.return_value = {"points": [{"x": 0.5, "y": 0.6}]}
        client._model = model

        with patch.object(client, "_load_image", return_value=sample_image):
            result = await client.point_objects("test.jpg", "center")

        assert result.success
        assert result.points[0].confidence is None
        assert result.points[0].point.x == 0.5
        assert result.points[0].point.y == 0.6

    @pytest.mark.asyncio
    async def test_semaphore_concurrency_control(
        self,
        client: MoondreamClient,
        sample_image: Image.Image,
    ) -> None:
        client._semaphore = asyncio.Semaphore(1)
        model = MagicMock()
        model.caption.return_value = {"caption": "Test"}
        client._model = model

        with patch.object(client, "_load_image", return_value=sample_image):
            results = await asyncio.gather(
                client.caption_image("test1.jpg"),
                client.caption_image("test2.jpg"),
            )

        assert all(result.success for result in results)
