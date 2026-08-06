#!/usr/bin/env python3
"""Configure Claude Desktop to launch the Moondream MCP server."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CommandSpec = Tuple[str, List[str]]


def get_claude_desktop_config_path() -> Path:
    """Return the Claude Desktop configuration path for this platform."""
    system = platform.system()

    if system == "Darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "Claude"
            / "claude_desktop_config.json"
        )
    if system == "Windows":
        app_data = os.environ.get("APPDATA")
        if app_data:
            return Path(app_data) / "Claude" / "claude_desktop_config.json"
        return (
            Path.home()
            / "AppData"
            / "Roaming"
            / "Claude"
            / "claude_desktop_config.json"
        )
    if system == "Linux":
        xdg_config = os.environ.get("XDG_CONFIG_HOME")
        base = Path(xdg_config) if xdg_config else Path.home() / ".config"
        return base / "Claude" / "claude_desktop_config.json"
    raise RuntimeError(f"Unsupported platform: {system}")


def find_moondream_mcp_command() -> Optional[CommandSpec]:
    """Find an executable command and argument list for the server."""
    executable = shutil.which("moondream-mcp")
    if executable:
        return executable, []

    venv_path = os.environ.get("VIRTUAL_ENV")
    if venv_path:
        if platform.system() == "Windows":
            candidate = Path(venv_path) / "Scripts" / "moondream-mcp.exe"
        else:
            candidate = Path(venv_path) / "bin" / "moondream-mcp"
        if candidate.is_file():
            return str(candidate), []

    try:
        import moondream_mcp  # noqa: F401
    except ImportError:
        return None
    return sys.executable, ["-m", "moondream_mcp.server"]


def find_moondream_mcp_executable() -> Optional[str]:
    """Compatibility helper returning only the executable portion."""
    command = find_moondream_mcp_command()
    return command[0] if command else None


def create_moondream_mcp_config(
    executable_path: str,
    environment_vars: Optional[Dict[str, str]] = None,
    args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create one Claude Desktop MCP server entry."""
    env_dict = dict(environment_vars or {})
    env_dict.setdefault("MOONDREAM_DEVICE", "auto")
    env_dict.setdefault("MOONDREAM_BACKEND", "photon")

    return {
        "command": executable_path,
        "args": list(args or []),
        "env": env_dict,
    }


def load_existing_config(config_path: Path) -> Dict[str, Any]:
    """Load an existing JSON object, or return an empty configuration."""
    if not config_path.exists():
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as file:
            result = json.load(file)
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Could not read existing config: {exc}") from exc

    if not isinstance(result, dict):
        raise RuntimeError("Claude Desktop config must contain a JSON object")
    return result


def save_config(config_path: Path, config: Dict[str, Any]) -> Optional[Path]:
    """Atomically save the configuration and return a backup path if created."""
    config_path.parent.mkdir(parents=True, exist_ok=True)

    backup_path: Optional[Path] = None
    if config_path.exists():
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = config_path.with_name(f"{config_path.name}.{timestamp}.backup")
        shutil.copy2(config_path, backup_path)

    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=config_path.parent,
            prefix=f".{config_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            json.dump(config, temporary, indent=2, ensure_ascii=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        temporary_path.replace(config_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return backup_path


def setup_claude_desktop(
    force: bool = False,
    environment_vars: Optional[Dict[str, str]] = None,
    config_path: Optional[Path] = None,
) -> bool:
    """Add or replace the Moondream server entry in Claude Desktop."""
    try:
        resolved_config_path = config_path or get_claude_desktop_config_path()
        print(f"Claude Desktop config path: {resolved_config_path}")

        command = find_moondream_mcp_command()
        if command is None:
            print("Error: Could not find the moondream-mcp installation.")
            print("Install it with: pip install moondream-mcp")
            return False

        executable, args = command
        print(f"Using command: {executable} {' '.join(args)}".rstrip())

        config = load_existing_config(resolved_config_path)
        existing_servers = config.get("mcpServers")
        if existing_servers is None:
            servers: Dict[str, Any] = {}
            config["mcpServers"] = servers
        elif isinstance(existing_servers, dict):
            servers = existing_servers
        else:
            raise RuntimeError("mcpServers must be a JSON object")

        if "moondream-mcp" in servers and not force:
            print("Moondream MCP is already configured; use --force to replace it.")
            return True

        servers["moondream-mcp"] = create_moondream_mcp_config(
            executable,
            environment_vars,
            args,
        )
        backup_path = save_config(resolved_config_path, config)

        if backup_path is not None:
            print(f"Created backup: {backup_path}")
        print(f"Configuration saved to: {resolved_config_path}")
        print("Restart Claude Desktop to load the Moondream tools.")
        return True
    except Exception as exc:
        print(f"Error setting up Claude Desktop integration: {exc}")
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Set up Claude Desktop integration for Moondream MCP"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing moondream-mcp entry",
    )
    parser.add_argument(
        "--backend",
        choices=["photon", "cloud"],
        default="photon",
        help="Moondream backend to configure",
    )
    parser.add_argument(
        "--model",
        default="moondream3.1-9B-A2B",
        help="Moondream model identifier",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Local inference device preference",
    )
    parser.add_argument(
        "--max-image-size",
        help="Maximum dimensions, such as 2048 or 2048x1536",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Inference timeout in seconds",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()

    env_vars = {
        "MOONDREAM_BACKEND": args.backend,
        "MOONDREAM_MODEL_NAME": args.model,
        "MOONDREAM_DEVICE": args.device,
    }
    if args.max_image_size:
        env_vars["MOONDREAM_MAX_IMAGE_SIZE"] = args.max_image_size
    if args.timeout is not None:
        if args.timeout < 1:
            raise SystemExit("--timeout must be at least 1")
        env_vars["MOONDREAM_TIMEOUT_SECONDS"] = str(args.timeout)

    if args.backend == "cloud":
        api_key = os.environ.get("MOONDREAM_API_KEY")
        if not api_key:
            raise SystemExit(
                "MOONDREAM_API_KEY must be set in the environment for --backend cloud"
            )
        env_vars["MOONDREAM_API_KEY"] = api_key

    success = setup_claude_desktop(
        force=args.force,
        environment_vars=env_vars,
    )
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
