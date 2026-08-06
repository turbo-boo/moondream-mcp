"""FastMCP server for Moondream 3.1 vision-language models."""

from typing import Any

__version__ = "2.0.1"
__author__ = "Moondream MCP Contributors"
__email__ = "contributors@moondream-mcp.dev"
__description__ = "FastMCP server for Moondream 3.1"

from .config import Config
from .models import (
    AnalysisResult,
    CaptionRequest,
    CaptionResult,
    ChatRequest,
    ChatResult,
    DetectionRequest,
    DetectionResult,
    ImageAnalysisRequest,
    PointingRequest,
    PointingResult,
    QueryRequest,
    QueryResult,
    SegmentRequest,
    SegmentResult,
)
from .moondream import MoondreamClient, MoondreamError


def create_server() -> Any:
    """Create the FastMCP server without eager FastMCP imports."""
    from .server import create_server as _create_server

    return _create_server()


def main() -> None:
    """Run the MCP server."""
    from .server import main as _main

    _main()


__all__ = [
    "Config",
    "MoondreamClient",
    "MoondreamError",
    "create_server",
    "main",
    "ImageAnalysisRequest",
    "CaptionRequest",
    "QueryRequest",
    "DetectionRequest",
    "PointingRequest",
    "SegmentRequest",
    "ChatRequest",
    "AnalysisResult",
    "CaptionResult",
    "QueryResult",
    "DetectionResult",
    "PointingResult",
    "SegmentResult",
    "ChatResult",
]
