"""Tests for current Moondream MCP input validation."""

import json
from pathlib import Path

import pytest

from moondream_mcp.models import CaptionLength
from moondream_mcp.validation import (
    ValidationError,
    sanitize_string,
    validate_caption_length,
    validate_image_path,
    validate_image_paths_list,
    validate_json_parameters,
    validate_messages_json,
    validate_object_name,
    validate_operation,
    validate_question,
    validate_spatial_refs_json,
)


def test_validation_error_fields() -> None:
    error = ValidationError("message", "CODE")
    assert error.message == "message"
    assert error.error_code == "CODE"
    assert str(error) == "message"


def test_validate_image_path() -> None:
    assert validate_image_path("test.jpg") == str(Path("test.jpg"))
    assert validate_image_path("https://example.com/image.jpg") == (
        "https://example.com/image.jpg"
    )
    with pytest.raises(ValidationError, match="cannot be empty"):
        validate_image_path(" ")
    with pytest.raises(ValidationError) as exc_info:
        validate_image_path("ftp://example.com/image.jpg")
    assert exc_info.value.error_code == "UNSUPPORTED_URL_SCHEME"


def test_validate_question() -> None:
    assert validate_question("  What is this?  ") == "What is this?"
    assert validate_question("What\x00 is this?") == "What is this?"
    with pytest.raises(ValidationError) as exc_info:
        validate_question("")
    assert exc_info.value.error_code == "EMPTY_QUESTION"
    with pytest.raises(ValidationError) as exc_info:
        validate_question("x" * 1001)
    assert exc_info.value.error_code == "STRING_TOO_LONG"


def test_validate_object_name() -> None:
    assert validate_object_name("  person's face  ") == "person's face"
    assert validate_object_name('sign saying "open"') == 'sign saying "open"'
    with pytest.raises(ValidationError) as exc_info:
        validate_object_name("x" * 101)
    assert exc_info.value.error_code == "STRING_TOO_LONG"


def test_caption_lengths_include_long_and_detailed_alias() -> None:
    assert validate_caption_length(" SHORT ") is CaptionLength.SHORT
    assert validate_caption_length("normal") is CaptionLength.NORMAL
    assert validate_caption_length("long") is CaptionLength.LONG
    assert validate_caption_length("detailed") is CaptionLength.DETAILED
    with pytest.raises(ValidationError) as exc_info:
        validate_caption_length("medium")
    assert exc_info.value.error_code == "INVALID_LENGTH"


def test_operations_include_segment() -> None:
    for operation in ("caption", "query", "detect", "point", "segment"):
        assert validate_operation(operation.upper()) == operation
    with pytest.raises(ValidationError) as exc_info:
        validate_operation("chat")
    assert exc_info.value.error_code == "INVALID_OPERATION"


def test_validate_image_paths_list_accepts_json_and_native_arrays() -> None:
    assert validate_image_paths_list('["one.jpg", "two.jpg"]') == [
        "one.jpg",
        "two.jpg",
    ]
    assert validate_image_paths_list(["one.jpg", "two.jpg"]) == [
        "one.jpg",
        "two.jpg",
    ]

    with pytest.raises(ValidationError) as exc_info:
        validate_image_paths_list("not json")
    assert exc_info.value.error_code == "INVALID_JSON"

    with pytest.raises(ValidationError) as exc_info:
        validate_image_paths_list([])
    assert exc_info.value.error_code == "EMPTY_PATHS_ARRAY"


def test_validate_json_parameters() -> None:
    assert validate_json_parameters("") == {}
    assert validate_json_parameters('{"value": 1}') == {"value": 1}
    with pytest.raises(ValidationError) as exc_info:
        validate_json_parameters("[]")
    assert exc_info.value.error_code == "INVALID_PARAMS_TYPE"


def test_spatial_refs_accept_points_boxes_and_native_arrays() -> None:
    expected = [[0.5, 0.4], [0.1, 0.2, 0.8, 0.9]]
    assert validate_spatial_refs_json(json.dumps(expected)) == expected
    assert validate_spatial_refs_json(expected) == expected
    assert validate_spatial_refs_json(None) == []


@pytest.mark.parametrize(
    "value,error_code",
    [
        ('{"x": 1}', "INVALID_SPATIAL_REFS_TYPE"),
        ("[[0.5]]", "INVALID_SPATIAL_REF"),
        ("[[1.2, 0.5]]", "INVALID_SPATIAL_REF"),
        ("[[0.8, 0.2, 0.1, 0.9]]", "INVALID_SPATIAL_REF"),
        ([[True, 0.5]], "INVALID_SPATIAL_REF"),
    ],
)
def test_spatial_refs_reject_invalid_values(
    value: object,
    error_code: str,
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        validate_spatial_refs_json(value)  # type: ignore[arg-type]
    assert exc_info.value.error_code == error_code


def test_validate_chat_messages_accepts_json_and_native_arrays() -> None:
    raw_messages = [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "Describe the image."},
    ]
    assert validate_messages_json(json.dumps(raw_messages)) == raw_messages
    assert validate_messages_json(raw_messages) == raw_messages

    multimodal = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this."},
                {"type": "image_url", "image_url": {"url": "test.jpg"}},
            ],
        }
    ]
    assert validate_messages_json(multimodal) == multimodal


@pytest.mark.parametrize(
    "messages,error_code",
    [
        ("", "EMPTY_MESSAGES"),
        ("{}", "INVALID_MESSAGES_TYPE"),
        ('[{"role":"tool","content":"x"}]', "INVALID_MESSAGE_ROLE"),
        ('[{"role":"user","content":4}]', "INVALID_MESSAGE_CONTENT"),
        ('[{"role":"user","content":""}]', "INVALID_MESSAGE_CONTENT"),
        ('[{"role":"user","content":[]}]', "INVALID_MESSAGE_CONTENT"),
    ],
)
def test_validate_chat_messages_rejects_invalid_values(
    messages: str,
    error_code: str,
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        validate_messages_json(messages)
    assert exc_info.value.error_code == error_code


def test_sanitize_string() -> None:
    assert sanitize_string("  text\x00  ") == "text"
    assert sanitize_string("a-b-c", allowed_chars="abc") == "abc"
    with pytest.raises(ValidationError) as exc_info:
        sanitize_string(1)
    assert exc_info.value.error_code == "INVALID_TYPE"
