"""Tests for the Claude Desktop setup helper."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

from moondream_mcp.scripts.setup_claude_desktop import (
    create_moondream_mcp_config,
    find_moondream_mcp_command,
    save_config,
    setup_claude_desktop,
)


def test_module_fallback_separates_command_and_args() -> None:
    with patch(
        "moondream_mcp.scripts.setup_claude_desktop.shutil.which",
        return_value=None,
    ), patch.dict("os.environ", {}, clear=True):
        command = find_moondream_mcp_command()

    assert command == (sys.executable, ["-m", "moondream_mcp.server"])


def test_create_config_does_not_mutate_caller_environment() -> None:
    environment = {"MOONDREAM_MODEL_NAME": "custom-model"}

    result = create_moondream_mcp_config(
        "/usr/bin/python",
        environment,
        ["-m", "moondream_mcp.server"],
    )

    assert environment == {"MOONDREAM_MODEL_NAME": "custom-model"}
    assert result["command"] == "/usr/bin/python"
    assert result["args"] == ["-m", "moondream_mcp.server"]
    assert result["env"]["MOONDREAM_BACKEND"] == "photon"
    assert result["env"]["MOONDREAM_DEVICE"] == "auto"


def test_save_config_is_atomic_and_creates_unique_backup(tmp_path: Path) -> None:
    config_path = tmp_path / "claude_desktop_config.json"
    config_path.write_text('{"existing": true}\n', encoding="utf-8")

    backup_path = save_config(config_path, {"mcpServers": {}})

    assert backup_path is not None
    assert backup_path.exists()
    assert json.loads(backup_path.read_text(encoding="utf-8")) == {"existing": True}
    assert json.loads(config_path.read_text(encoding="utf-8")) == {"mcpServers": {}}


def test_setup_preserves_other_servers(tmp_path: Path) -> None:
    config_path = tmp_path / "claude_desktop_config.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "other": {"command": "other-server", "args": []},
                }
            }
        ),
        encoding="utf-8",
    )

    with patch(
        "moondream_mcp.scripts.setup_claude_desktop.find_moondream_mcp_command",
        return_value=(sys.executable, ["-m", "moondream_mcp.server"]),
    ):
        success = setup_claude_desktop(
            environment_vars={"MOONDREAM_BACKEND": "cloud"},
            config_path=config_path,
        )

    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert success is True
    assert saved["mcpServers"]["other"]["command"] == "other-server"
    assert saved["mcpServers"]["moondream-mcp"]["command"] == sys.executable
    assert saved["mcpServers"]["moondream-mcp"]["args"] == [
        "-m",
        "moondream_mcp.server",
    ]
