"""Configuration for the Moondream MCP server."""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch

DeviceType = Literal["cpu", "cuda", "mps"]
BackendType = Literal["photon", "cloud"]


@dataclass
class Config:
    """Runtime configuration.

    ``model_revision`` and ``trust_remote_code`` are retained only so existing
    deployments do not fail when upgrading from the Transformers-based backend.
    Moondream 3.1 Photon does not use either value.
    """

    model_name: str = "moondream3.1-9B-A2B"
    backend: BackendType = "photon"
    api_key: Optional[str] = None

    # Legacy compatibility settings; ignored by the Photon and cloud SDKs.
    model_revision: str = "2025-01-09"
    trust_remote_code: bool = True

    # Photon selects its own execution device. These fields remain useful for
    # diagnostics and compatibility with existing environment configurations.
    device: DeviceType = "cpu"
    device_auto_detect: bool = True

    max_image_size: Tuple[int, int] = (2048, 2048)
    supported_formats: Tuple[str, ...] = ("JPEG", "PNG", "WebP", "BMP", "TIFF")
    max_file_size_mb: int = 50

    timeout_seconds: int = 120
    max_concurrent_requests: int = 5
    enable_streaming: bool = True
    max_batch_size: int = 10
    batch_concurrency: int = 3
    enable_batch_progress: bool = True

    request_timeout_seconds: int = 30
    max_redirects: int = 5
    user_agent: str = "Moondream-MCP/2.0.0"

    @classmethod
    def from_env(cls) -> "Config":
        config = cls()

        config.model_name = os.getenv("MOONDREAM_MODEL_NAME", config.model_name)
        config.backend = _parse_backend(os.getenv("MOONDREAM_BACKEND", config.backend))
        config.api_key = os.getenv("MOONDREAM_API_KEY") or None

        # Accepted for a smooth upgrade from 1.x, but no longer passed to a
        # Transformers loader.
        config.model_revision = os.getenv(
            "MOONDREAM_MODEL_REVISION", config.model_revision
        )
        config.trust_remote_code = _parse_bool(
            os.getenv("MOONDREAM_TRUST_REMOTE_CODE", "true")
        )

        device_env = os.getenv("MOONDREAM_DEVICE")
        if device_env:
            normalized_device = device_env.lower()
            if normalized_device == "auto":
                config.device = config._detect_best_device()
                config.device_auto_detect = True
            elif normalized_device in ("cpu", "cuda", "mps"):
                config.device = normalized_device  # type: ignore[assignment]
                config.device_auto_detect = False
            else:
                raise ValueError(
                    f"Invalid MOONDREAM_DEVICE: {device_env}. "
                    "Must be one of: auto, cpu, cuda, mps"
                )
        else:
            config.device = config._detect_best_device()

        max_size_env = os.getenv("MOONDREAM_MAX_IMAGE_SIZE")
        if max_size_env:
            try:
                if "x" in max_size_env.lower():
                    width, height = map(int, max_size_env.lower().split("x", 1))
                    config.max_image_size = (width, height)
                else:
                    size = int(max_size_env)
                    config.max_image_size = (size, size)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid MOONDREAM_MAX_IMAGE_SIZE: {max_size_env}. "
                    "Use format: '2048' or '2048x1536'"
                ) from exc

        config.max_file_size_mb = _env_int(
            "MOONDREAM_MAX_FILE_SIZE_MB", config.max_file_size_mb
        )
        config.timeout_seconds = _env_int(
            "MOONDREAM_TIMEOUT_SECONDS", config.timeout_seconds
        )
        config.max_concurrent_requests = _env_int(
            "MOONDREAM_MAX_CONCURRENT_REQUESTS",
            config.max_concurrent_requests,
        )
        config.enable_streaming = _parse_bool(
            os.getenv("MOONDREAM_ENABLE_STREAMING", "true")
        )
        config.max_batch_size = _env_int(
            "MOONDREAM_MAX_BATCH_SIZE", config.max_batch_size
        )
        config.batch_concurrency = _env_int(
            "MOONDREAM_BATCH_CONCURRENCY", config.batch_concurrency
        )
        config.enable_batch_progress = _parse_bool(
            os.getenv("MOONDREAM_ENABLE_BATCH_PROGRESS", "true")
        )

        config.request_timeout_seconds = _env_int(
            "MOONDREAM_REQUEST_TIMEOUT_SECONDS",
            config.request_timeout_seconds,
        )
        config.max_redirects = _env_int("MOONDREAM_MAX_REDIRECTS", config.max_redirects)
        config.user_agent = os.getenv("MOONDREAM_USER_AGENT", config.user_agent)

        config._validate()
        return config

    def _detect_best_device(self) -> DeviceType:
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _validate(self) -> None:
        if self.backend not in ("photon", "cloud"):
            raise ValueError("backend must be 'photon' or 'cloud'")
        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty")
        if self.backend == "cloud" and not self.api_key:
            raise ValueError(
                "MOONDREAM_API_KEY is required when MOONDREAM_BACKEND=cloud"
            )

        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be at least 1")

        max_width, max_height = self.max_image_size
        if max_width < 1 or max_height < 1:
            raise ValueError("max_image_size dimensions must be at least 1")
        if max_width > 4096 or max_height > 4096:
            raise ValueError("max_image_size dimensions cannot exceed 4096")

        if self.max_file_size_mb < 1:
            raise ValueError("max_file_size_mb must be at least 1")
        if self.max_file_size_mb > 500:
            raise ValueError("max_file_size_mb cannot exceed 500")

        if self.max_concurrent_requests < 1:
            raise ValueError("max_concurrent_requests must be at least 1")
        if self.max_concurrent_requests > 50:
            raise ValueError("max_concurrent_requests cannot exceed 50")

        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        if self.max_batch_size > 100:
            raise ValueError("max_batch_size cannot exceed 100")

        if self.batch_concurrency < 1:
            raise ValueError("batch_concurrency must be at least 1")
        if self.batch_concurrency > self.max_concurrent_requests:
            raise ValueError("batch_concurrency cannot exceed max_concurrent_requests")

        if self.request_timeout_seconds < 1:
            raise ValueError("request_timeout_seconds must be at least 1")
        if self.max_redirects < 0:
            raise ValueError("max_redirects cannot be negative")

    def validate_dependencies(self) -> None:
        try:
            import aiohttp  # noqa: F401
            import moondream as md
            from PIL import Image  # noqa: F401
        except ImportError as exc:
            raise ValueError(
                f"Missing required dependency: {exc.name}. "
                "Install the project with: pip install -e ."
            ) from exc

        required_factory = "photon" if self.backend == "photon" else "vl"
        if not hasattr(md, required_factory):
            raise ValueError(
                "The installed moondream package is too old for this backend. "
                "Install moondream==2.0.1."
            )

        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "CUDA was requested but is unavailable. Install CUDA-enabled "
                "PyTorch or set MOONDREAM_DEVICE=auto."
            )
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise ValueError(
                "MPS was requested but is unavailable. MPS requires Apple "
                "Silicon and a supported macOS release."
            )

    def get_device_info(self) -> str:
        if self.device == "cuda":
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                return f"CUDA ({device_name}, {memory_gb:.1f}GB)"
            return "CUDA (not available)"
        if self.device == "mps":
            if torch.backends.mps.is_available():
                return "MPS (Apple Silicon)"
            return "MPS (not available)"
        return "CPU"

    def __str__(self) -> str:
        return (
            "Config("
            f"backend={self.backend}, "
            f"model={self.model_name}, "
            f"device={self.get_device_info()}, "
            f"max_size={self.max_image_size[0]}x{self.max_image_size[1]}, "
            f"timeout={self.timeout_seconds}s, "
            f"max_concurrent={self.max_concurrent_requests}"
            ")"
        )


def _parse_bool(value: str) -> bool:
    return value.lower() in ("true", "1", "yes", "on")


def _parse_backend(value: str) -> BackendType:
    normalized = value.strip().lower()
    if normalized not in ("photon", "cloud"):
        raise ValueError(
            f"Invalid MOONDREAM_BACKEND: {value}. Must be one of: photon, cloud"
        )
    return normalized  # type: ignore[return-value]


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: {value}. Expected an integer.") from exc
