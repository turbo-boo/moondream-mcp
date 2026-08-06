"""Async wrapper around the Moondream 3.1 Python SDK."""

from __future__ import annotations

import asyncio
import io
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import aiofiles
import aiohttp
import moondream as md
import torch
from PIL import Image

from .config import Config
from .models import (
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
    SpatialRef,
)


class MoondreamError(Exception):
    def __init__(self, message: str, error_code: str = "MOONDREAM_ERROR") -> None:
        self.message = message
        self.error_code = error_code
        super().__init__(message)


class ModelLoadError(MoondreamError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "MODEL_LOAD_ERROR")


class ImageProcessingError(MoondreamError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "IMAGE_PROCESSING_ERROR")


class InferenceError(MoondreamError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "INFERENCE_ERROR")


class MoondreamClient:
    """Thread-safe async facade for Photon or the Moondream cloud client."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._device: Optional[torch.device] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self._model_lock = asyncio.Lock()

    async def __aenter__(self) -> "MoondreamClient":
        await self._ensure_session()
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        await self.cleanup()

    async def _ensure_session(self) -> None:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(
                total=self.config.request_timeout_seconds
            )
            connector = aiohttp.TCPConnector(
                limit=max(10, self.config.max_concurrent_requests),
            )
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={"User-Agent": self.config.user_agent},
            )

    async def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return
        async with self._model_lock:
            if self._model is None:
                await self._load_model()

    async def _load_model(self) -> None:
        print(
            f"Loading Moondream backend={self.config.backend} "
            f"model={self.config.model_name}",
            file=sys.stderr,
        )

        self._device = torch.device(self.config.device)
        loop = asyncio.get_running_loop()

        def load_sync() -> Any:
            if self.config.backend == "photon":
                if self.config.device == "cpu":
                    raise RuntimeError(
                        "Photon local inference currently requires an NVIDIA "
                        "Ampere-or-newer GPU or Apple Silicon. Use "
                        "MOONDREAM_BACKEND=cloud for CPU-only hosts."
                    )
                return md.photon(self.config.model_name)

            if not self.config.api_key:
                raise RuntimeError(
                    "MOONDREAM_API_KEY is required for the cloud backend"
                )
            return md.vl(
                api_key=self.config.api_key,
                model=self.config.model_name,
            )

        try:
            self._model = await loop.run_in_executor(None, load_sync)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load Moondream model: {exc}"
            ) from exc

        print("Moondream model loaded", file=sys.stderr)

    async def _load_image(self, image_path: str) -> Image.Image:
        try:
            if self._is_url(image_path):
                return await self._load_image_from_url(image_path)
            return await self._load_image_from_file(image_path)
        except ImageProcessingError:
            raise
        except Exception as exc:
            raise ImageProcessingError(
                f"Failed to load image from {image_path}: {exc}"
            ) from exc

    @staticmethod
    def _is_url(path: str) -> bool:
        try:
            parsed = urlparse(path)
            return parsed.scheme in ("http", "https") and bool(parsed.netloc)
        except Exception:
            return False

    async def _load_image_from_url(self, url: str) -> Image.Image:
        await self._ensure_session()
        if self._session is None:
            raise RuntimeError("HTTP session was not initialized")

        try:
            async with self._session.get(
                url,
                allow_redirects=True,
                max_redirects=self.config.max_redirects,
            ) as response:
                if response.status != 200:
                    raise ImageProcessingError(
                        f"Failed to download image: HTTP {response.status}"
                    )

                content_type = response.headers.get("content-type", "")
                if not content_type.lower().startswith("image/"):
                    raise ImageProcessingError(
                        f"URL does not point to an image: {content_type}"
                    )

                max_bytes = self.config.max_file_size_mb * 1024 * 1024
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > max_bytes:
                    size_mb = int(content_length) / (1024 * 1024)
                    raise ImageProcessingError(
                        f"Image too large: {size_mb:.1f}MB > "
                        f"{self.config.max_file_size_mb}MB"
                    )

                chunks = bytearray()
                async for chunk in response.content.iter_chunked(64 * 1024):
                    chunks.extend(chunk)
                    if len(chunks) > max_bytes:
                        raise ImageProcessingError(
                            "Image exceeded the configured size limit while "
                            "downloading"
                        )

                image = Image.open(io.BytesIO(chunks))
                image.load()
                return self._preprocess_image(image)
        except aiohttp.ClientError as exc:
            raise ImageProcessingError(
                f"Network error loading image: {exc}"
            ) from exc

    async def _load_image_from_file(self, file_path: str) -> Image.Image:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise ImageProcessingError(f"Image file not found: {file_path}")
        if not path.is_file():
            raise ImageProcessingError(
                f"Image path is not a file: {file_path}"
            )

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self.config.max_file_size_mb:
            raise ImageProcessingError(
                f"Image too large: {size_mb:.1f}MB > "
                f"{self.config.max_file_size_mb}MB"
            )

        try:
            async with aiofiles.open(path, "rb") as file:
                image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            image.load()
            return self._preprocess_image(image)
        except ImageProcessingError:
            raise
        except Exception as exc:
            raise ImageProcessingError(
                f"Error reading image file: {exc}"
            ) from exc

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")

            max_width, max_height = self.config.max_image_size
            if image.width > max_width or image.height > max_height:
                image.thumbnail(
                    (max_width, max_height),
                    Image.Resampling.LANCZOS,
                )
            return image
        except Exception as exc:
            raise ImageProcessingError(
                f"Error preprocessing image: {exc}"
            ) from exc

    @staticmethod
    def _join_stream(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, Iterable):
            return "".join(str(part) for part in value)
        return str(value)

    def _metadata(
        self,
        image_path: str,
        image: Image.Image,
    ) -> Dict[str, Any]:
        return {
            "image_path": image_path,
            "image_size": f"{image.width}x{image.height}",
            "backend": self.config.backend,
            "model": self.config.model_name,
            "device": self.config.device,
        }

    async def caption_image(
        self,
        image_path: str,
        length: CaptionLength = CaptionLength.NORMAL,
        stream: bool = False,
    ) -> CaptionResult:
        async with self._semaphore:
            started = time.perf_counter()
            try:
                await self._ensure_model_loaded()
                image = await self._load_image(image_path)
                loop = asyncio.get_running_loop()

                def infer() -> str:
                    if self._model is None:
                        raise RuntimeError("Model not initialized")
                    use_stream = stream and self.config.enable_streaming
                    result = self._model.caption(
                        image,
                        length=length.sdk_value,
                        stream=use_stream,
                    )
                    return self._join_stream(result["caption"])

                caption = await loop.run_in_executor(None, infer)
                return CaptionResult(
                    success=True,
                    caption=caption,
                    length=length,
                    processing_time_ms=(time.perf_counter() - started) * 1000,
                    metadata=self._metadata(image_path, image),
                )
            except (ModelLoadError, ImageProcessingError):
                raise
            except Exception as exc:
                return CaptionResult(
                    success=False,
                    caption=None,
                    length=length,
                    error_message=str(exc),
                    error_code="INFERENCE_ERROR",
                    processing_time_ms=(time.perf_counter() - started) * 1000,
                    metadata={"image_path": image_path},
                )

    async def query_image(
        self,
        image_path: str,
        question: str,
        stream: bool = False,
        reasoning: bool = False,
        spatial_refs: Optional[List[SpatialRef]] = None,
    ) -> QueryResult:
        async with self._semaphore:
            started = time.perf_counter()
            try:
                await self._ensure_model_loaded()
                image = await self._load_image(image_path)
                loop = asyncio.get_running_loop()

                def infer() -> Dict[str, Any]:
                    if self._model is None:
                        raise RuntimeError("Model not initialized")
                    use_stream = stream and self.config.enable_streaming
                    kwargs: Dict[str, Any] = {
                        "stream": use_stream,
                        "reasoning": reasoning,
                    }
                    if spatial_refs:
                        kwargs["spatial_refs"] = spatial_refs
                    result = self._model.query(
                        image,
                        question,
                        **kwargs,
                    )
                    return {
                        "answer": self._join_stream(result["answer"]),
                        "reasoning": result.get("reasoning"),
                    }

                result = await loop.run_in_executor(None, infer)
                return QueryResult(
                    success=True,
                    answer=result["answer"],
                    question=question,
                    reasoning=result["reasoning"],
                    processing_time_ms=(time.perf_counter() - started) * 1000,
                    metadata=self._metadata(image_path, image),
                )
            except (ModelLoadError, ImageProcessingError):
                raise
            except Exception as exc:
                return QueryResult(
                    success=False,
                    answer=None,
                    question=question,
                    reasoning=None,
                    error_message=str(exc),
                    error_code="INFERENCE_ERROR",
                    processing_time_ms=(time.perf_counter() - started) * 1000,
                    metadata={"image_path": image_path},
                )

    async def detect_objects(
        self,
        image_path: str,
        object_name: str,
    ) -> DetectionResult:
        async with self._semaphore:
            started = time.perf_counter()
            try:
                await self._ensure_model_loaded()
                image = await self._load_image(image_path)
                loop = asyncio.get_running_loop()

                def infer() -> Dict[str, Any]:
                    if self._model is None:
                        raise RuntimeError("Model not initialized")
                    return self._model.detect(image, object_name)

                result = await loop.run_in_executor(None, infer)
                detected = [
                    DetectedObject(
                        name=object_name,
                        confidence=_optional_float(obj.get("confidence")),
                        bounding_box=_parse_bounding_box(obj),
                    )
                    for obj in result.get("objects", [])
                ]
                return DetectionResult(
                    success=True,
                    objects=detected,
                    object_name=object_name,
                    total_found=len(detected),
                    processing_time_ms=(time.perf_counter() - started) * 1000,
                    metadata=self._metadata(image_path, image),
                )
            except (ModelLoadError, ImageProcessingError):
                raise
            except Exception as exc:
                return DetectionResult(
                    success=False,
                    object_name=object_name,
                    error_message=str(exc),
                    error_code="INFERENCE_ERROR",
                    processing_time_ms=(time.perf_counter() - started) * 1000,
                    metadata={"image_path": image_path},
                )

    async def point_objects(
        self,
        image_path: str,
        object_name: str,
    ) -> PointingResult:
        async with self._semaphore:
            started = time.perf_counter()
            try:
                await self._ensure_model_loaded()
                image = await self._load_image(image_path)
                loop = asyncio.get_running_loop()

                def infer() -> Dict[str, Any]:
                    if self._model is None:
                        raise RuntimeError("Model not initialized")
                    return self._model.point(image, object_name)

                result = await loop.run_in_executor(None, infer)
                points = [
                    PointedObject(
                        name=object_name,
                        confidence=_optional_float(item.get("confidence")),
                        point=Point(
                            x=float(item["x"]),
                            y=float(item["y"]),
                        ),
                    )
                    for item in result.get("points", [])
                ]
                return PointingResult(
                    success=True,
                    points=points,
                    object_name=object_name,
                    total_found=len(points),
                    processing_time_ms=(time.perf_counter() - started) * 1000,
                    metadata=self._metadata(image_path, image),
                )
            except (ModelLoadError, ImageProcessingError):
                raise
            except Exception as exc:
                return PointingResult(
                    success=False,
                    object_name=object_name,
                    error_message=str(exc),
                    error_code="INFERENCE_ERROR",
                    processing_time_ms=(time.perf_counter() - started) * 1000,
                    metadata={"image_path": image_path},
                )

    async def segment_objects(
        self,
        image_path: str,
        object_name: str,
        spatial_refs: Optional[List[SpatialRef]] = None,
        stream: bool = False,
    ) -> SegmentResult:
        async with self._semaphore:
            started = time.perf_counter()
            try:
                await self._ensure_model_loaded()
                image = await self._load_image(image_path)
                loop = asyncio.get_running_loop()

                def infer() -> Dict[str, Any]:
                    if self._model is None:
                        raise RuntimeError("Model not initialized")
                    kwargs: Dict[str, Any] = {
                        "stream": stream and self.config.enable_streaming,
                    }
                    if spatial_refs:
                        kwargs["spatial_refs"] = spatial_refs
                    raw = self._model.segment(
                        image,
                        object_name,
                        **kwargs,
                    )
                    return _consume_segment_result(raw)

                result = await loop.run_in_executor(None, infer)
                raw_bbox = result.get("bbox")
                return SegmentResult(
                    success=True,
                    object_name=object_name,
                    path=result.get("path"),
                    bounding_box=(
                        _parse_bounding_box(raw_bbox)
                        if isinstance(raw_bbox, dict)
                        else None
                    ),
                    processing_time_ms=(time.perf_counter() - started) * 1000,
                    metadata=self._metadata(image_path, image),
                )
            except (ModelLoadError, ImageProcessingError):
                raise
            except Exception as exc:
                return SegmentResult(
                    success=False,
                    object_name=object_name,
                    error_message=str(exc),
                    error_code="INFERENCE_ERROR",
                    processing_time_ms=(time.perf_counter() - started) * 1000,
                    metadata={"image_path": image_path},
                )

    async def chat_messages(
        self,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        reasoning: Optional[bool] = None,
    ) -> ChatResult:
        async with self._semaphore:
            started = time.perf_counter()
            try:
                await self._ensure_model_loaded()
                loop = asyncio.get_running_loop()

                def infer() -> Dict[str, Any]:
                    if self._model is None:
                        raise RuntimeError("Model not initialized")
                    kwargs: Dict[str, Any] = {
                        "stream": stream and self.config.enable_streaming,
                    }
                    if reasoning is not None:
                        kwargs["reasoning"] = reasoning
                    raw = self._model.chat(messages, **kwargs)
                    return _consume_chat_result(raw)

                message = await loop.run_in_executor(None, infer)
                return ChatResult(
                    success=True,
                    message=message,
                    processing_time_ms=(time.perf_counter() - started) * 1000,
                    metadata={
                        "backend": self.config.backend,
                        "model": self.config.model_name,
                        "device": self.config.device,
                    },
                )
            except ModelLoadError:
                raise
            except Exception as exc:
                return ChatResult(
                    success=False,
                    message=None,
                    error_message=str(exc),
                    error_code="INFERENCE_ERROR",
                    processing_time_ms=(time.perf_counter() - started) * 1000,
                )

    async def cleanup(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

        model = self._model
        self._model = None
        if model is not None:
            close = getattr(model, "close", None)
            if callable(close):
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, close)

        self._tokenizer = None
        self._device = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Moondream client cleaned up", file=sys.stderr)


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _parse_bounding_box(obj: Dict[str, Any]) -> BoundingBox:
    if all(key in obj for key in ("x_min", "y_min", "x_max", "y_max")):
        return BoundingBox(
            x_min=float(obj["x_min"]),
            y_min=float(obj["y_min"]),
            x_max=float(obj["x_max"]),
            y_max=float(obj["y_max"]),
        )

    if all(key in obj for key in ("x", "y", "width", "height")):
        x = float(obj["x"])
        y = float(obj["y"])
        return BoundingBox(
            x_min=x,
            y_min=y,
            x_max=min(1.0, x + float(obj["width"])),
            y_max=min(1.0, y + float(obj["height"])),
        )

    raise InferenceError(
        "Detection result did not contain a recognized bounding-box schema"
    )


def _consume_segment_result(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return {
            "path": raw.get("path"),
            "bbox": raw.get("bbox") or raw.get("bounding_box"),
        }

    if not isinstance(raw, Iterable):
        raise InferenceError("Segment result was not iterable")

    path: Optional[str] = None
    bbox: Optional[Dict[str, Any]] = None
    for update in raw:
        if not isinstance(update, dict):
            continue
        if update.get("path") is not None:
            path = str(update["path"])
        raw_bbox = update.get("bbox") or update.get("bounding_box")
        if isinstance(raw_bbox, dict):
            bbox = raw_bbox
    return {"path": path, "bbox": bbox}


def _consume_chat_result(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        message = raw.get("message")
        if isinstance(message, dict):
            normalized = dict(message)
            normalized["content"] = _join_chat_content(
                normalized.get("content")
            )
            return normalized
        content = raw.get("content")
        if content is not None:
            return {
                "role": "assistant",
                "content": _join_chat_content(content),
            }

    if isinstance(raw, str):
        return {"role": "assistant", "content": raw}

    if not isinstance(raw, Iterable):
        raise InferenceError("Chat result was not iterable")

    parts: List[str] = []
    for chunk in raw:
        content = _extract_chat_chunk(chunk)
        if content:
            parts.append(content)
    return {"role": "assistant", "content": "".join(parts)}


def _join_chat_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Iterable):
        return "".join(_extract_chat_chunk(item) for item in content)
    return "" if content is None else str(content)


def _extract_chat_chunk(chunk: Any) -> str:
    if isinstance(chunk, str):
        return chunk
    if not isinstance(chunk, dict):
        return str(chunk)

    direct = chunk.get("content")
    if isinstance(direct, str):
        return direct

    for key in ("message", "delta"):
        nested = chunk.get(key)
        if isinstance(nested, dict):
            nested_content = nested.get("content")
            if isinstance(nested_content, str):
                return nested_content
    return ""
