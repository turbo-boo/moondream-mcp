# Changelog

All notable changes to this project are documented here. The project follows
Semantic Versioning.

## [2.0.0] - 2026-08-06

### Changed

- Replaced the Transformers remote-code loader with `moondream==2.0.1`.
- Updated the default local model to `moondream3.1-9B-A2B` through Photon.
- Updated FastMCP to `3.4.4` and simplified the stdio server lifecycle.
- Changed object-detection boxes to the native normalized
  `x_min`, `y_min`, `x_max`, `y_max` schema.
- Stopped generating placeholder confidence scores when Moondream omits them.
- Updated packaging metadata for PEP 639 and Python 3.10 through 3.14.

### Added

- Local Photon and Moondream Cloud backends.
- `segment_objects` with optional spatial references and streaming support.
- `chat_messages` with streaming and optional reasoning.
- Query reasoning and normalized point/box spatial references.
- Query streaming in addition to caption streaming.
- Streamed remote-image byte-limit enforcement.
- Tests for the Moondream 2.0.1 SDK surface and all eight MCP tools.

### Compatibility

- Existing MCP tool names remain available.
- `detailed` remains an alias for the new `long` caption length.
- Legacy `x`, `y`, `width`, and `height` boxes remain accepted by the Python
  response model.
- `MOONDREAM_MODEL_REVISION` and `MOONDREAM_TRUST_REMOTE_CODE` remain accepted
  as ignored upgrade-compatibility settings.

## [1.0.2] - 2025-07-02

### Changed

- Updated the documented package version.

## [1.0.1] - 2025-07-01

### Fixed

- Replaced runtime assertions with explicit errors.

## [1.0.0] - 2024-12-19

### Added

- Initial FastMCP server for Moondream2.
- Caption, query, detection, pointing, combined analysis, and batch tools.
- Local and remote image loading, validation, concurrency controls, and tests.
