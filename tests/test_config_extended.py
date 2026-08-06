"""Extended tests for configuration management."""

import os
from unittest.mock import patch

import pytest

from moondream_mcp.config import Config


class TestConfigExtended:
    def test_config_from_env_with_all_variables(self) -> None:
        env_vars = {
            "MOONDREAM_MODEL_NAME": "custom-model",
            "MOONDREAM_DEVICE": "cuda",
            "MOONDREAM_MAX_IMAGE_SIZE": "2048",
            "MOONDREAM_MAX_IMAGE_PIXELS": "12000000",
            "MOONDREAM_REQUEST_TIMEOUT_SECONDS": "45",
            "MOONDREAM_MAX_CONCURRENT_REQUESTS": "8",
            "MOONDREAM_ENABLE_STREAMING": "true",
            "MOONDREAM_MAX_BATCH_SIZE": "15",
            "MOONDREAM_BATCH_CONCURRENCY": "5",
            "MOONDREAM_ALLOW_PRIVATE_NETWORK_URLS": "true",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = Config.from_env()

        assert config.model_name == "custom-model"
        assert config.device == "cuda"
        assert config.max_image_size == (2048, 2048)
        assert config.max_image_pixels == 12_000_000
        assert config.request_timeout_seconds == 45
        assert config.max_concurrent_requests == 8
        assert config.enable_streaming is True
        assert config.max_batch_size == 15
        assert config.batch_concurrency == 5
        assert config.allow_private_network_urls is True

    @pytest.mark.parametrize(
        ("env_value", "expected"),
        [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("0", False),
            ("no", False),
            ("off", False),
        ],
    )
    def test_config_from_env_boolean_parsing(
        self,
        env_value: str,
        expected: bool,
    ) -> None:
        with patch.dict(
            os.environ,
            {"MOONDREAM_ENABLE_STREAMING": env_value},
            clear=False,
        ):
            config = Config.from_env()
        assert config.enable_streaming is expected

    @pytest.mark.parametrize("invalid_value", ["", "invalid", "tru"])
    def test_invalid_boolean_is_rejected(self, invalid_value: str) -> None:
        with patch.dict(
            os.environ,
            {"MOONDREAM_ENABLE_STREAMING": invalid_value},
            clear=False,
        ):
            with pytest.raises(ValueError, match="MOONDREAM_ENABLE_STREAMING"):
                Config.from_env()

    def test_config_from_env_integer_parsing_invalid(self) -> None:
        with patch.dict(
            os.environ,
            {"MOONDREAM_MAX_IMAGE_PIXELS": "not-an-integer"},
            clear=False,
        ):
            with pytest.raises(ValueError, match="MOONDREAM_MAX_IMAGE_PIXELS"):
                Config.from_env()

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("1024", (1024, 1024)),
            ("1920x1080", (1920, 1080)),
            ("1920X1080", (1920, 1080)),
        ],
    )
    def test_config_max_image_size_formats(
        self,
        value: str,
        expected: tuple[int, int],
    ) -> None:
        with patch.dict(
            os.environ,
            {"MOONDREAM_MAX_IMAGE_SIZE": value},
            clear=False,
        ):
            config = Config.from_env()
        assert config.max_image_size == expected

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"max_image_size": (0, 100)}, "dimensions must be at least 1"),
            ({"max_image_size": (5000, 100)}, "dimensions cannot exceed 4096"),
            ({"max_image_pixels": 0}, "max_image_pixels must be at least 1"),
            ({"timeout_seconds": 0}, "timeout_seconds must be at least 1"),
            (
                {"max_concurrent_requests": 0, "batch_concurrency": 1},
                "max_concurrent_requests must be at least 1",
            ),
            (
                {"max_concurrent_requests": 51},
                "max_concurrent_requests cannot exceed 50",
            ),
            ({"max_batch_size": 0}, "max_batch_size must be at least 1"),
            ({"max_batch_size": 101}, "max_batch_size cannot exceed 100"),
            ({"batch_concurrency": 0}, "batch_concurrency must be at least 1"),
            (
                {"batch_concurrency": 10, "max_concurrent_requests": 5},
                "batch_concurrency cannot exceed max_concurrent_requests",
            ),
            ({"max_file_size_mb": 0}, "max_file_size_mb must be at least 1"),
            ({"max_file_size_mb": 501}, "max_file_size_mb cannot exceed 500"),
            (
                {"request_timeout_seconds": 0},
                "request_timeout_seconds must be at least 1",
            ),
            ({"max_redirects": -1}, "max_redirects cannot be negative"),
            ({"max_redirects": 21}, "max_redirects cannot exceed 20"),
        ],
    )
    def test_config_validation_bounds(
        self,
        kwargs: dict[str, object],
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            Config(**kwargs)  # type: ignore[arg-type]

    def test_config_device_validation(self) -> None:
        for device in ("cpu", "cuda", "mps"):
            assert Config(device=device).device == device

        with patch.dict(
            os.environ,
            {"MOONDREAM_DEVICE": "invalid_device"},
            clear=False,
        ):
            with pytest.raises(ValueError, match="Invalid MOONDREAM_DEVICE"):
                Config.from_env()

    def test_config_repr_and_str(self) -> None:
        config = Config()
        assert "model_name" in repr(config)
        assert "backend=photon" in str(config)
        assert "model=moondream3.1-9B-A2B" in str(config)

    def test_config_equality(self) -> None:
        assert Config() == Config()
        assert Config() != Config(max_image_size=(1024, 1024))

    def test_config_device_info(self) -> None:
        assert Config(device="cpu").get_device_info() == "CPU"
        assert "CUDA" in Config(device="cuda").get_device_info()
        assert "MPS" in Config(device="mps").get_device_info()
        assert (
            Config(
                backend="cloud",
                api_key="test-key",
            ).get_device_info()
            == "remote"
        )

    def test_config_env_var_precedence(self) -> None:
        with patch.dict(
            os.environ,
            {"MOONDREAM_MAX_IMAGE_SIZE": "512"},
            clear=False,
        ):
            assert Config.from_env().max_image_size == (512, 512)

        with patch.dict(os.environ, {}, clear=True):
            assert Config.from_env().max_image_size == (2048, 2048)

    def test_config_network_settings(self) -> None:
        config = Config(
            request_timeout_seconds=60,
            max_redirects=10,
            allow_private_network_urls=True,
            user_agent="Custom-Agent/1.0",
        )
        assert config.request_timeout_seconds == 60
        assert config.max_redirects == 10
        assert config.allow_private_network_urls is True
        assert config.user_agent == "Custom-Agent/1.0"

    def test_config_supported_formats(self) -> None:
        assert Config().supported_formats == ("JPEG", "PNG", "WEBP", "BMP", "TIFF")

    def test_legacy_upgrade_variables_remain_accepted(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MOONDREAM_MODEL_REVISION": "custom-rev",
                "MOONDREAM_TRUST_REMOTE_CODE": "false",
            },
            clear=False,
        ):
            config = Config.from_env()
        assert config.model_revision == "custom-rev"
        assert config.trust_remote_code is False
