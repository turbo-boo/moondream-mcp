# Moondream MCP Server

A FastMCP server for Moondream 3.1. It exposes captioning, visual question
answering, reasoning, spatial references, object detection, visual pointing,
segmentation, and chat through the Model Context Protocol.

The default model is `moondream3.1-9B-A2B`, loaded locally with the Photon
runtime from `moondream==2.0.1`. A Moondream Cloud backend is also available.

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
| `MOONDREAM_MAX_IMAGE_SIZE` | `2048x2048` | Maximum preprocessed dimensions |
| `MOONDREAM_MAX_IMAGE_PIXELS` | `40000000` | Maximum decoded pixel count |
| `MOONDREAM_MAX_FILE_SIZE_MB` | `50` | Maximum local or remote image size |
| `MOONDREAM_TIMEOUT_SECONDS` | `120` | Time before an inference call returns a timeout error |
| `MOONDREAM_MAX_CONCURRENT_REQUESTS` | `5` | Global request concurrency |
| `MOONDREAM_ENABLE_STREAMING` | `true` | Allow SDK streaming modes |
| `MOONDREAM_MAX_BATCH_SIZE` | `10` | Maximum images in one batch |
| `MOONDREAM_BATCH_CONCURRENCY` | `3` | Per-batch concurrency |
| `MOONDREAM_REQUEST_TIMEOUT_SECONDS` | `30` | Remote image request timeout |
| `MOONDREAM_MAX_REDIRECTS` | `5` | Remote image redirect limit |
| `MOONDREAM_ALLOW_PRIVATE_NETWORK_URLS` | `false` | Permit loopback or private image hosts |
| `MOONDREAM_USER_AGENT` | `Moondream-MCP/2.0.1` | Remote image request user agent |

Boolean variables accept `true`, `false`, `1`, `0`, `yes`, `no`, `on`, or
`off`. Invalid spellings fail at startup instead of silently changing behavior.

`MOONDREAM_MODEL_REVISION` and `MOONDREAM_TRUST_REMOTE_CODE` are accepted only
for upgrade compatibility. Photon does not use Transformers remote code or a
Transformers revision.

## MCP tools

Tools return structured MCP content. Clients receive objects directly rather
than JSON text that must be parsed a second time.

For compatibility, `spatial_refs`, `messages`, and batch `image_paths` still
accept their former JSON-string representation. Native arrays are preferred.

### `caption_image`

```json
{
  "image_path": "/path/to/image.jpg",
  "length": "long",
  "stream": false
}
```

Accepted lengths are `short`, `normal`, and `long`. `detailed` remains a
compatibility alias for `long`.

### `query_image`

```json
{
  "image_path": "/path/to/image.jpg",
  "question": "What color is the highlighted object?",
  "stream": false,
  "reasoning": true,
  "spatial_refs": [[0.10, 0.20, 0.80, 0.90]]
}
```

Each spatial reference is a normalized point `[x, y]` or box
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
  "spatial_refs": [[0.50, 0.50]],
  "stream": false
}
```

Returns the segmentation `path` produced by the SDK and its normalized bounding
box when available.

### `chat_messages`

```json
{
  "messages": [
    {"role": "system", "content": "Be precise."},
    {"role": "user", "content": "Describe the scene."}
  ],
  "stream": false,
  "reasoning": true
}
```

Roles may be `system`, `user`, or `assistant`. Content may be a non-empty string
or an SDK-compatible structured content list.

### `analyze_image`

Runs one of `caption`, `query`, `detect`, `point`, or `segment` through one tool.
It accepts the corresponding `question`, `object_name`, `length`, `stream`,
`reasoning`, and `spatial_refs` fields.

### `batch_analyze_images`

```json
{
  "image_paths": ["one.jpg", "two.jpg"],
  "operation": "caption",
  "length": "normal"
}
```

The response includes `success`, `partial_success`, `successful_count`, and
`failed_count`. One failed image no longer makes the batch appear fully
successful.

## Output conventions

- Coordinates are normalized from `0.0` to `1.0`.
- Missing confidence values remain `null`; the server does not invent scores.
- Errors contain a stable `error_code`, readable `error_message`, operation, and
  non-sensitive context.
- Blocking SDK work runs outside the MCP event loop.
- The inference timeout bounds how long the MCP caller waits. A native SDK call
  already running in a worker thread may finish afterward because Python cannot
  forcibly stop that thread safely.

## Image handling

- EXIF orientation is applied before inference.
- Transparent images are composited onto white before RGB conversion.
- Encoded byte size, decoded pixel count, supported format, and target dimensions
  are checked separately.
- Supported formats are JPEG, PNG, WebP, BMP, and TIFF.
- Remote redirects are followed manually so every destination is validated.
- Private, loopback, link-local, multicast, reserved, and otherwise non-public
  remote addresses are blocked by default.

Set `MOONDREAM_ALLOW_PRIVATE_NETWORK_URLS=true` only when the MCP server must
read from a trusted local image service. This weakens SSRF protection and should
not be enabled for untrusted callers.

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

Do not place `MOONDREAM_API_KEY` in MCP arguments, logs, or checked-in files.
Use the environment configuration of the MCP host.

DNS and network policy can differ between deployments. For internet-facing
setups, combine the application checks with an outbound firewall or proxy that
also blocks private and metadata networks.

## License

The MCP server code is MIT licensed. The Moondream model is distributed under
its own model license; review that license before redistribution or commercial
deployment.
