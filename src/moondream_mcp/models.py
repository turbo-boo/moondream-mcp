"""Pydantic request and response models for Moondream MCP."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class CaptionLength(str, Enum):
    """Caption lengths accepted by the MCP API.

    ``detailed`` is retained as a compatibility alias and is translated to the
    Moondream 3.1 SDK's ``long`` value before inference.
    """

    SHORT = "short"
    NORMAL = "normal"
    LONG = "long"
    DETAILED = "detailed"

    @property
    def sdk_value(self) -> str:
        return "long" if self is CaptionLength.DETAILED else self.value


class ImageAnalysisRequest(BaseModel):
    image_path: str = Field(
        ..., description="Path to image file (local path or URL)", min_length=1
    )

    @field_validator("image_path")
    @classmethod
    def validate_image_path(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Image path cannot be empty")
        return value


class CaptionRequest(ImageAnalysisRequest):
    length: CaptionLength = Field(
        default=CaptionLength.NORMAL, description="Caption length"
    )
    stream: bool = Field(default=False, description="Stream text generation")


class QueryRequest(ImageAnalysisRequest):
    question: str = Field(..., min_length=1, max_length=1000)
    stream: bool = Field(default=False, description="Stream text generation")

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be empty")
        return value


class DetectionRequest(ImageAnalysisRequest):
    object_name: str = Field(..., min_length=1, max_length=100)

    @field_validator("object_name")
    @classmethod
    def validate_object_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Object name cannot be empty")
        return value


class PointingRequest(ImageAnalysisRequest):
    object_name: str = Field(..., min_length=1, max_length=100)

    @field_validator("object_name")
    @classmethod
    def validate_object_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Object name cannot be empty")
        return value


class BatchAnalysisRequest(BaseModel):
    image_paths: List[str] = Field(..., min_length=1, max_length=100)
    operation: str = Field(..., pattern="^(caption|query|detect|point)$")
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("image_paths")
    @classmethod
    def validate_image_paths(cls, values: List[str]) -> List[str]:
        validated: List[str] = []
        for value in values:
            value = value.strip()
            if not value:
                raise ValueError("Image path cannot be empty")
            validated.append(value)
        return validated


class StandardError(BaseModel):
    success: bool = False
    error_message: str
    error_code: str
    error_context: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[str] = None


class AnalysisResult(BaseModel):
    success: bool
    processing_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create_error(
        cls,
        error_message: str,
        error_code: str = "ANALYSIS_ERROR",
        metadata: Optional[Dict[str, Any]] = None,
        processing_time_ms: Optional[float] = None,
    ) -> "AnalysisResult":
        return cls(
            success=False,
            error_message=error_message,
            error_code=error_code,
            metadata=metadata or {},
            processing_time_ms=processing_time_ms,
        )


class CaptionResult(AnalysisResult):
    caption: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    length: Optional[CaptionLength] = None


class QueryResult(AnalysisResult):
    answer: Optional[str] = None
    question: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class BoundingBox(BaseModel):
    """Normalized Moondream bounding box.

    Moondream 3.1 natively returns ``x_min``, ``y_min``, ``x_max`` and
    ``y_max``. Legacy ``x``, ``y``, ``width`` and ``height`` input remains
    accepted to avoid breaking callers that construct this model directly.
    """

    x_min: float = Field(..., ge=0.0, le=1.0)
    y_min: float = Field(..., ge=0.0, le=1.0)
    x_max: float = Field(..., ge=0.0, le=1.0)
    y_max: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def convert_legacy_box(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "x_min" in data:
            return data
        if all(key in data for key in ("x", "y", "width", "height")):
            x = float(data["x"])
            y = float(data["y"])
            return {
                "x_min": x,
                "y_min": y,
                "x_max": min(1.0, x + float(data["width"])),
                "y_max": min(1.0, y + float(data["height"])),
            }
        return data

    @model_validator(mode="after")
    def validate_order(self) -> "BoundingBox":
        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise ValueError("Bounding box maxima must be greater than minima")
        return self

    @property
    def x(self) -> float:
        return self.x_min

    @property
    def y(self) -> float:
        return self.y_min

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min


class DetectedObject(BaseModel):
    name: str
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    bounding_box: BoundingBox


class DetectionResult(AnalysisResult):
    objects: List[DetectedObject] = Field(default_factory=list)
    object_name: Optional[str] = None
    total_found: int = Field(default=0, ge=0)


class Point(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)


class PointedObject(BaseModel):
    name: str
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    point: Point


class PointingResult(AnalysisResult):
    points: List[PointedObject] = Field(default_factory=list)
    object_name: Optional[str] = None
    total_found: int = Field(default=0, ge=0)


class BatchAnalysisResult(BaseModel):
    results: List[AnalysisResult]
    total_processed: int = Field(..., ge=0)
    total_successful: int = Field(..., ge=0)
    total_failed: int = Field(..., ge=0)
    total_processing_time_ms: float = Field(..., ge=0.0)


AnalysisRequestType = Union[
    CaptionRequest,
    QueryRequest,
    DetectionRequest,
    PointingRequest,
    BatchAnalysisRequest,
]

AnalysisResultType = Union[
    CaptionResult,
    QueryResult,
    DetectionResult,
    PointingResult,
    BatchAnalysisResult,
]
