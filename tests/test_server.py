"""Tests for the FastMCP server lifecycle."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from moondream_mcp.config import Config
from moondream_mcp.server import create_server, main, run_server_async


class TestServer:
    @patch("moondream_mcp.server.Config.from_env")
    @patch("moondream_mcp.server.FastMCP")
    @patch("moondream_mcp.server.MoondreamClient")
    @patch("moondream_mcp.server.register_vision_tools")
    def test_create_server_success(
        self,
        mock_register_tools: MagicMock,
        mock_client_class: MagicMock,
        mock_fastmcp_class: MagicMock,
        mock_config_from_env: MagicMock,
    ) -> None:
        mock_config = MagicMock(spec=Config)
        mock_config_from_env.return_value = mock_config
        mock_mcp = MagicMock()
        mock_fastmcp_class.return_value = mock_mcp
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mcp, client = create_server()

        mock_config.validate_dependencies.assert_called_once()
        mock_fastmcp_class.assert_called_once_with(name="moondream-mcp")
        mock_client_class.assert_called_once_with(mock_config)
        mock_register_tools.assert_called_once_with(
            mock_mcp,
            mock_client,
            mock_config,
        )
        assert mcp is mock_mcp
        assert client is mock_client

    @patch("moondream_mcp.server.Config.from_env")
    def test_create_server_config_error(
        self,
        mock_config_from_env: MagicMock,
    ) -> None:
        mock_config_from_env.side_effect = ValueError("Invalid configuration")
        with pytest.raises(ValueError, match="Invalid configuration"):
            create_server()

    @patch("moondream_mcp.server.Config.from_env")
    def test_create_server_dependency_error(
        self,
        mock_config_from_env: MagicMock,
    ) -> None:
        mock_config = MagicMock(spec=Config)
        mock_config.validate_dependencies.side_effect = ValueError("Missing dependency")
        mock_config_from_env.return_value = mock_config

        with pytest.raises(ValueError, match="Missing dependency"):
            create_server()

    @pytest.mark.asyncio
    @patch("moondream_mcp.server.create_server")
    @patch("signal.signal")
    async def test_run_server_async_success(
        self,
        mock_signal: MagicMock,
        mock_create_server: MagicMock,
    ) -> None:
        mock_mcp = AsyncMock()
        mock_client = AsyncMock()
        mock_client.config.backend = "photon"
        mock_create_server.return_value = (mock_mcp, mock_client)
        mock_mcp.run_async = AsyncMock(return_value=None)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        await run_server_async()

        mock_mcp.run_async.assert_awaited_once_with(transport="stdio")
        mock_client.__aenter__.assert_awaited_once()
        mock_client.__aexit__.assert_awaited_once()
        assert mock_signal.call_count == 2

    @pytest.mark.asyncio
    @patch("moondream_mcp.server.create_server")
    @patch("signal.signal")
    async def test_run_server_async_server_error(
        self,
        mock_signal: MagicMock,
        mock_create_server: MagicMock,
    ) -> None:
        mock_mcp = AsyncMock()
        mock_client = AsyncMock()
        mock_client.config.backend = "photon"
        mock_create_server.return_value = (mock_mcp, mock_client)
        mock_mcp.run_async = AsyncMock(side_effect=RuntimeError("Server error"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with pytest.raises(RuntimeError, match="Server error"):
            await run_server_async()

    @patch("sys.version_info", (3, 9))
    @patch("moondream_mcp.server.asyncio.run")
    def test_main_python_version_check(
        self,
        mock_asyncio_run: MagicMock,
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        mock_asyncio_run.assert_not_called()

    @patch("sys.version_info", (3, 10))
    @patch("moondream_mcp.server.asyncio.run")
    def test_main_success(self, mock_asyncio_run: MagicMock) -> None:
        main()
        mock_asyncio_run.assert_called_once()

    @patch("sys.version_info", (3, 10))
    @patch("moondream_mcp.server.asyncio.run")
    def test_main_keyboard_interrupt(self, mock_asyncio_run: MagicMock) -> None:
        mock_asyncio_run.side_effect = KeyboardInterrupt()
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    @patch("sys.version_info", (3, 10))
    @patch("moondream_mcp.server.asyncio.run")
    def test_main_exception(self, mock_asyncio_run: MagicMock) -> None:
        mock_asyncio_run.side_effect = RuntimeError("Fatal error")
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
