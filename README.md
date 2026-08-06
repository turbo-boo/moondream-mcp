# Moondream MCP Server

A FastMCP server for Moondream 3.1. It exposes captioning, visual question
answering, reasoning, spatial references, object detection, visual pointing,
segmentation, and chat through the Model Context Protocol.

The default model is `moondream3.1-9B-A2B`, loaded locally with the Photon
runtime from `moondream==2.0.1`. A Moondream Cloud backend is also available.

## What changed in 2.0

- Replaced the legacy Transformers `AutoModelForCausalLM` loader with the
  official Moondream SDK.
- Updated the local backend to `md.photon("moondream3.1-9B-A2B")`.
- Added an optional cloud backend through `md.vl(...)`.
- Added `segment_objects` and `chat_messages` MCP tools.
- Added query reasoning and spatial-reference inputs.
- Updated caption lengths to `short`, `normal`, and `long`.
- Corrected detection output to the native normalized
  `x_min`, `y_min`, `x_max`, `y_max` schema.
- Removed fabricated confidence values when the SDK does not return one.
- Updated FastMCP to `3.4.4` and Moondream to `2.0.1`.
- Hardened remote image downloads by enforcing the byte limit while streaming.

`detailed` remains accepted as an alias for `long`. Legacy
`x`/`y`/`width`/`height` boxes can still be parsed by the Python response model.

## Requirements

- Python 3.10 through 3.14
- Local Photon backend:
  - NVIDIA Ampere-or-newer GPU with a compatible PyTorch installation, or
  - Apple Silicon on a supported macOS release
- Cloud backend:
  - A Moondream API key

Model weights are downloaded automatically on the first local run.

## Installation

### uvx

```bash
uvx moondream-mcp
```

### pip

```bash
pip install moondream-mcp
moondream-mcp
```

### Source

```bash
git clone https://github.com/turbo-boo/moondream-mcp.git
cd moondream-mcp
pip install -e .
moondream-mcp
```

## MCP host configuration

### Local Photon backend

```json
{
  "mcpServers": {
    "moondream": {
      "command": "uvx",
      "args": ["moondream-mcp"],
      "env": {
        "MOONDREAM_BACKEND": "photon",
        "MOONDREAM_MODEL_NAME": "moondream3.1-9B-A2B",
        "MOONDREAM_DEVICE": "auto"
      }
    }
  }
}
```

### Moondream Cloud backend

```json
{
  "mcpServers": {
    "moondream": {
      "command": "uvx",
      "args": ["moondream-mcp"],
      "env": {
        "MOONDREAM_BACKEND": "cloud",
        "MOONDREAM_MODEL_NAME": "moondream3.1-9B-A2B",
        "MOONDREAM_API_KEY": "your-api-key"
      }
    }
  }
}
```

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `MOONDREAM_BACKEND` | `photon` | `photon` for local inference or `cloud` |
| `MOONDREAM_MODEL_NAME` | `moondream3.1-9B-A2B` | Model identifier |
| `MOONDREAM_API_KEY` | unset | Required by the cloud backend |
| `MOONDREAM_DEVICE` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |
| `MOONDREAM_MAX_IMAGE_SIZE` | `2048x2048` | Maximum preprocessed image dimensions |
| `MOONDREAM_MAX_FILE_SIZE_MB` | `50` | Maximum local or remote image size |
| `MOONDREAM_TIMEOUT_SECONDS` | `120` | Inference timeout setting |
| `MOONDREAM_MAX_CONCURRENT_REQUESTS` | `5` | Global inference concurrency |
| `MOONDREAM_ENABLE_STREAMING` | `true` | Allow SDK streaming modes |
| `MOONDREAM_MAX_BATCH_SIZE` | `10` | Maximum images in one batch |
| `MOONDREAM_BATCH_CONCURRENCY` | `3` | Per-batch concurrency |
| `MOONDREAM_REQUEST_TIMEOUT_SECONDS` | `30` | Remote image request timeout |
| `MOONDREAM_MAX_REDIRECTS` | `5` | Remote image redirect limit |

`MOONDREAM_MODEL_REVISION` and `MOONDREAM_TRUST_REMOTE_CODE` are accepted only
for upgrade compatibility. Photon does not use Transformers remote code or a
Transformers revision.

## MCP tools

### `caption_image`

```json
{
  "image_path": "/path/to/image.jpg",
  "length": "long",
  "stream": false
}
```

Accepted lengths are `short`, `normal`, and `long`. `detailed` is a compatibility
alias for `long`.

### `query_image`

```json
{
  "image_path": "/path/to/image.jpg",
  "question": "What color is the highlighted object?",
  "stream": false,
  "reasoning": true,
  "spatial_refs": "[[0.10, 0.20, 0.80, 0.90]]"
}
```

`spatial_refs` is a JSON string containing normalized points `[x, y]` or boxes
`[x_min, y_min, x_max, y_max]`.

### `detect_objects`

```json
{
  "image_path": "/path/to/image.jpg",
  "object_name": "person"
}
```

Each object contains a normalized bounding box:

```json
{
  "x_min": 0.10,
  "y_min": 0.15,
  "x_max": 0.42,
  "y_max": 0.88
}
```

### `point_objects`

```json
{
  "image_path": "/path/to/image.jpg",
  "object_name": "red car"
}
```

Each result contains normalized `x` and `y` coordinates.

### `segment_objects`

```json
{
  "image_path": "/path/to/image.jpg",
  "object_name": "person",
  "spatial_refs": "[[0.50, 0.50]]",
  "stream": false
}
```

Returns the segmentation `path` produced by the SDK and its normalized bounding
box when available.

### `chat_messages`

```json
{
  "messages": "[{\"role\":\"system\",\"content\":\"Be precise.\"},{\"role\":\"user\",\"content\":\"Describe the scene.\"}]",
  "stream": false,
  "reasoning": true
}
```

`messages` is a JSON string containing SDK-compatible message objects. Roles may
be `system`, `user`, or `assistant`; content may be a string or a structured
content list.

### `analyze_image`

Runs one of `caption`, `query`, `detect`, `point`, or `segment` through a single
typed tool. It accepts the corresponding `question`, `object_name`, `length`,
`stream`, `reasoning`, and `spatial_refs` fields.

### `batch_analyze_images`

Runs one image operation over a JSON array of local paths or image URLs while
respecting the configured batch limit and concurrency.

## Output conventions

- Coordinates are normalized from `0.0` to `1.0`.
- Missing confidence values remain `null`; the server does not invent scores.
- Errors contain a stable `error_code`, readable `error_message`, operation, and
  non-sensitive context.
- Model inference runs in an executor so the MCP event loop remains responsive.

## Development

```bash
pip install -e ".[dev]"
pytest
black --check src tests
isort --check-only src tests
mypy src/moondream_mcp
python -m build
```

Unit tests mock the model SDK. A real model download is not required for the
normal test suite.

## Security notes

Remote URLs must return an image content type and stay within the configured
byte limit. The limit is enforced while streaming the response even when the
server omits `Content-Length`.

Do not place `MOONDREAM_API_KEY` in MCP arguments, logs, or checked-in files.
Use the environment configuration of the MCP host.

## License

The MCP server code is MIT licensed. The Moondream model is distributed under
its own model license; review that license before redistribution or commercial
deployment.
