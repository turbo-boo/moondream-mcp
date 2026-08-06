"""Moondream MCP server entry point."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from typing import Optional

from fastmcp import FastMCP

from .config import Config
from .moondream import MoondreamClient
from .tools import register_vision_tools


def create_server() -> tuple[FastMCP, MoondreamClient]:
    try:
        config = Config.from_env()
        print(f"Configuration loaded: {config}", file=sys.stderr)
        config.validate_dependencies()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise

    mcp: FastMCP = FastMCP(name="moondream-mcp")
    moondream_client = MoondreamClient(config)
    register_vision_tools(mcp, moondream_client, config)

    print(
        "Registered tools: caption_image, query_image, detect_objects, "
        "point_objects, segment_objects, chat_messages, analyze_image, "
        "batch_analyze_images",
        file=sys.stderr,
    )
    return mcp, moondream_client


async def run_server_async() -> None:
    mcp, moondream_client = create_server()
    shutdown_event = asyncio.Event()
    force_shutdown_count = 0

    def signal_handler(signum: int, frame: Optional[object]) -> None:
        nonlocal force_shutdown_count
        force_shutdown_count += 1
        if force_shutdown_count == 1:
            print(
                f"Received signal {signum}; shutting down",
                file=sys.stderr,
            )
            shutdown_event.set()
        elif force_shutdown_count >= 3:
            os._exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    async with moondream_client:
        print(
            f"Starting stdio MCP server with backend="
            f"{moondream_client.config.backend}",
            file=sys.stderr,
        )
        server_task = asyncio.create_task(mcp.run_async(transport="stdio"))
        shutdown_task = asyncio.create_task(shutdown_event.wait())

        done, pending = await asyncio.wait(
            (server_task, shutdown_task),
            return_when=asyncio.FIRST_COMPLETED,
        )

        if shutdown_task in done:
            server_task.cancel()

        for task in pending:
            task.cancel()

        await asyncio.gather(*pending, return_exceptions=True)

        if server_task in done:
            await server_task


def main() -> None:
    if sys.version_info < (3, 10):
        print("Python 3.10 or higher is required", file=sys.stderr)
        raise SystemExit(1)

    try:
        asyncio.run(run_server_async())
    except KeyboardInterrupt:
        raise SystemExit(0)
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
