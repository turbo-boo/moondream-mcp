"""Async wrapper around the Moondream 3.1 Python SDK."""

from __future__ import annotations

import asyncio
import inspect
import io
import ipaddress
import socket
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar
from urllib.parse import urljoin, urlparse

import aiofiles
import aiohttp
import moondream as md
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

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

T = TypeVar("T")


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
            timeout = aiohttp.ClientTimeout(total=self.config.request_timeout_seconds)
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
            raise ModelLoadError(f"Failed to load Moondream model: {exc}") from exc

        print("Moondream model loaded", file=sys.stderr)

    async def _run_sync(self, operation: str, function: Callable[[], T]) -> T:
        """Run blocking SDK work without blocking the MCP event loop."""
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, function)
        try:
            return await asyncio.wait_for(
                future,
                timeout=self.config.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise InferenceError(
                f"{operation} timed out after {self.config.timeout_seconds} seconds"
            ) from exc

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
            return parsed.scheme.lower() in ("http", "https") and bool(parsed.netloc)
        except Exception:
            return False

    async def _validate_remote_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
            raise ImageProcessingError("Only http:// and https:// image URLs are supported")
        if self.config.allow_private_network_urls:
            return

        hostname = parsed.hostname.rstrip(".").lower()
        if hostname == "localhost" or hostname.endswith(".localhost"):
            raise ImageProcessingError("Private-network image URLs are disabled")

        addresses: List[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
        try:
            addresses.append(ipaddress.ip_address(hostname.split("%", 1)[0]))
        except ValueError:
            try:
                loop = asyncio.get_running_loop()
                resolved = await loop.getaddrinfo(
                    hostname,
                    parsed.port or (443 if parsed.scheme.lower() == "https" else 80),
                    type=socket.SOCK_STREAM,
                )
            except socket.gaierror as exc:
                raise ImageProcessingError(
                    f"Could not resolve image URL host: {hostname}"
                ) from exc
            for result in resolved:
                address = result[4][0].split("%", 1)[0]
                addresses.append(ipaddress.ip_address(address))

        if not addresses or any(not address.is_global for address in addresses):
            raise ImageProcessingError("Private-network image URLs are disabled")

    async def _load_image_from_url(self, url: str) -> Image.Image:
        await self._ensure_session()
        if self._session is None:
            raise RuntimeError("HTTP session was not initialized")

        current_url = url
        try:
            for redirect_count in range(self.config.max_redirects + 1):
                await self._validate_remote_url(current_url)
                async with self._session.get(
                    current_url,
                    allow_redirects=False,
                ) as response:
                    if response.status in (301, 302, 303, 307, 308):
                        location = response.headers.get("location")
                        if not location:
                            raise ImageProcessingError(
                                "Image URL redirect did not include a location"
                            )
                        if redirect_count >= self.config.max_redirects:
                            raise ImageProcessingError(
                                f"Image URL exceeded {self.config.max_redirects} redirects"
                            )
                        current_url = urljoin(current_url, location)
                        continue

                    if not 200 <= response.status < 300:
                        raise ImageProcessingError(
                            f"Failed to download image: HTTP {response.status}"
                        )

                    image_data = await self._read_image_response(response)
                    return self._decode_image(image_data, current_url)
        except aiohttp.ClientError as exc:
            raise ImageProcessingError(f"Network error loading image: {exc}") from exc

        raise ImageProcessingError("Failed to resolve image URL redirect")

    async def _read_image_response(self, response: aiohttp.ClientResponse) -> bytes:
        content_type = response.headers.get("content-type", "")
        if not content_type.lower().startswith("image/"):
            raise ImageProcessingError(
                f"URL does not point to an image: {content_type or 'unknown content type'}"
            )

        max_bytes = self.config.max_file_size_mb * 1024 * 1024
        content_length = response.headers.get("content-length")
        if content_length:
            try:
                declared_length = int(content_length)
            except ValueError:
                declared_length = 0
            if declared_length > max_bytes:
                size_mb = declared_length / (1024 * 1024)
                raise ImageProcessingError(
                    f"Image too large: {size_mb:.1f}MB > "
                    f"{self.config.max_file_size_mb}MB"
                )

        chunks = bytearray()
        async for chunk in response.content.iter_chunked(64 * 1024):
            chunks.extend(chunk)
            if len(chunks) > max_bytes:
                raise ImageProcessingError(
                    "Image exceeded the configured size limit while downloading"
                )
        return bytes(chunks)

    async def _load_image_from_file(self, file_path: str) -> Image.Image:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise ImageProcessingError(f"Image file not found: {file_path}")
        if not path.is_file():
            raise ImageProcessingError(f"Image path is not a file: {file_path}")

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self.config.max_file_size_mb:
            raise ImageProcessingError(
                f"Image too large: {size_mb:.1f}MB > "
                f"{self.config.max_file_size_mb}MB"
            )

        try:
            async with aiofiles.open(path, "rb") as file:
                image_data = await file.read()
            return self._decode_image(image_data, file_path)
        except ImageProcessingError:
            raise
        except Exception as exc:
            raise ImageProcessingError(f"Error reading image file: {exc}") from exc

    def _decode_image(self, image_data: bytes, source: str) -> Image.Image:
        try:
            with Image.open(io.BytesIO(image_data)) as opened:
                image_format = opened.format
                if image_format not in self.config.supported_formats:
                    supported = ", ".join(self.config.supported_formats)
                    raise ImageProcessingError(
                        f"Unsupported image format {image_format or 'unknown'}; "
                        f"supported formats: {supported}"
                    )

                pixel_count = opened.width * opened.height
                if pixel_count > self.config.max_image_pixels:
                    raise ImageProcessingError(
                        f"Image dimensions are too large: {opened.width}x{opened.height} "
                        f"({pixel_count} pixels > {self.config.max_image_pixels})"
                    )

                opened.load()
                image = opened.copy()
            return self._preprocess_image(image)
        except ImageProcessingError:
            raise
        except (UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise ImageProcessingError(f"Invalid or unsafe image from {source}") from exc
        except Exception as exc:
            raise ImageProcessingError(f"Could not decode image from {source}: {exc}") from exc

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        try:
            image = ImageOps.exif_transpose(image)

            if image.mode in ("RGBA", "LA") or (
                image.mode == "P" and "transparency" in image.info
            ):
                rgba = image.convert("RGBA")
                background = Image.new("RGB", rgba.size, "white")
                background.paste(rgba, mask=rgba.getchannel("A"))
                image = background
            elif image.mode != "RGB":
                image = image.convert("RGB")

            max_width, max_height = self.config.max_image_size
            if image.width > max_width or image.height > max_height:
                image.thumbnail(
                    (max_width, max_height),
                    Image.Resampling.LANCZOS,
                )
            return image
        except Exception as exc:
            raise ImageProcessingError(f"Error preprocessing image: {exc}") from exc

    @staticmethod
    def _join_stream(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            raise InferenceError("Text result unexpectedly contained an object")
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
            "device": self.config.device if self.config.backend == "photon" else "remote",
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

                def infer() -> str:
                    if self._model is None:
                        raise RuntimeError("Model not initialized")
                    result = self._model.caption(
                        image,
                        length=length.sdk_value,
                        stream=stream and self.config.enable_streaming,
                    )
                    return self._join_stream(result["caption"])

                caption = await self._run_sync("Caption generation", infer)
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
                    error_code=getattr(exc, "error_code", "INFERENCE_ERROR"),
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

                def infer() -> Dict[str, Any]:
                    if self._model is None:
                        raise RuntimeError("Model not initialized")
                    kwargs: Dict[str, Any] = {
                        "stream": stream and self.config.enable_streaming,
                        "reasoning": reasoning,
                    }
                    if spatial_refs:
                        kwargs["spatial_refs"] = spatial_refs
                    result = self._model.query(image, question, **kwargs)
                    raw_reasoning = result.get("reasoning")
                    normalized_reasoning: Any
                    if raw_reasoning is None or isinstance(raw_reasoning, dict):
                        normalized_reasoning = raw_reasoning
                    else:
                        normalized_reasoning = self._join_stream(raw_reasoning)
                    return {
                        "answer": self._join_stream(result["answer"]),
                        "reasoning": normalized_reasoning,
                    }

                result = await self._run_sync("Image query", infer)
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
                    error_code=getattr(exc, "error_code", "INFERENCE_ERROR"),
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

                def infer() -> Dict[str, Any]:
                    if self._model is None:
                        raise RuntimeError("Model not initialized")
                    return self._model.detect(image, object_name)

                result = await self._run_sync("Object detection", infer)
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
                    error_code=getattr(exc, "error_code", "INFERENCE_ERROR"),
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

                def infer() -> Dict[str, Any]:
                    if self._model is None:
                        raise RuntimeError("Model not initialized")
                    return self._model.point(image, object_name)

                result = await self._run_sync("Object pointing", infer)
                points = [
                    PointedObject(
                        name=object_name,
                        confidence=_optional_float(item.get("confidence")),
                        point=Point(x=float(item["x"]), y=float(item["y"])),
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
                    error_code=getattr(exc, "error_code", "INFERENCE_ERROR"),
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

                def infer() -> Dict[str, Any]:
                    if self._model is None:
                        raise RuntimeError("Model not initialized")
                    kwargs: Dict[str, Any] = {
                        "stream": stream and self.config.enable_streaming,
                    }
                    if spatial_refs:
                        kwargs["spatial_refs"] = spatial_refs
                    raw = self._model.segment(image, object_name, **kwargs)
                    return _consume_segment_result(raw)

                result = await self._run_sync("Object segmentation", infer)
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
                    error_code=getattr(exc, "error_code", "INFERENCE_ERROR"),
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

                message = await self._run_sync("Chat generation", infer)
                return ChatResult(
                    success=True,
                    message=message,
                    processing_time_ms=(time.perf_counter() - started) * 1000,
                    metadata={
                        "backend": self.config.backend,
                        "model": self.config.model_name,
                        "device": (
                            self.config.device
                            if self.config.backend == "photon"
                            else "remote"
                        ),
                    },
                )
            except ModelLoadError:
                raise
            except Exception as exc:
                return ChatResult(
                    success=False,
                    message=None,
                    error_message=str(exc),
                    error_code=getattr(exc, "error_code", "INFERENCE_ERROR"),
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
                if inspect.iscoroutinefunction(close):
                    await close()
                else:
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(None, close)
                    if inspect.isawaitable(result):
                        await result

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
            normalized["content"] = _join_chat_content(normalized.get("content"))
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
    if isinstance(content, dict):
        return _extract_chat_chunk(content)
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
