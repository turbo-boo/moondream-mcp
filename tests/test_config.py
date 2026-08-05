"""Tests for Moondream 3.1 configuration."""

import os

import pytest

from moondream_mcp.config import Config


class TestConfig:
    def test_default_config(self) -> None:
        config = Config()
        assert config.model_name == "moondream3.1-9B-A2B"
        assert config.backend == "photon"
        assert config.api_key is None
        assert config.max_image_size == (2048, 2048)
        assert config.timeout_seconds == 120
        assert config.max_concurrent_requests == 5

    def test_from_env_with_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in list(os.environ):
            if key.startswith("MOONDREAM_"):
                monkeypatch.delenv(key, raising=False)

        config = Config.from_env()
        assert config.model_name == "moondream3.1-9B-A2B"
        assert config.backend == "photon"
        assert config.device in ("cpu", "cuda", "mps")

    def test_cloud_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MOONDREAM_BACKEND", "cloud")
        monkeypatch.setenv("MOONDREAM_API_KEY", "test-key")
        config = Config.from_env()
        assert config.backend == "cloud"
        assert config.api_key == "test-key"

    def test_cloud_backend_requires_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MOONDREAM_BACKEND", "cloud")
        monkeypatch.delenv("MOONDREAM_API_KEY", raising=False)
        with pytest.raises(ValueError, match="MOONDREAM_API_KEY"):
            Config.from_env()

    def test_invalid_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MOONDREAM_BACKEND", "legacy")
        with pytest.raises(ValueError, match="Invalid MOONDREAM_BACKEND"):
            Config.from_env()

    def test_from_env_with_custom_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MOONDREAM_MODEL_NAME", "custom/model")
        monkeypatch.setenv("MOONDREAM_DEVICE", "cpu")
        monkeypatch.setenv("MOONDREAM_MAX_IMAGE_SIZE", "1024x768")
        monkeypatch.setenv("MOONDREAM_TIMEOUT_SECONDS", "60")

        config = Config.from_env()
        assert config.model_name == "custom/model"
        assert config.device == "cpu"
        assert config.max_image_size == (1024, 768)
        assert config.timeout_seconds == 60

    def test_invalid_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MOONDREAM_DEVICE", "invalid")
        with pytest.raises(ValueError, match="Invalid MOONDREAM_DEVICE"):
            Config.from_env()

    def test_invalid_image_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MOONDREAM_MAX_IMAGE_SIZE", "invalid")
        with pytest.raises(ValueError, match="Invalid MOONDREAM_MAX_IMAGE_SIZE"):
            Config.from_env()

    def test_validation_errors(self) -> None:
        config = Config(timeout_seconds=0)
        with pytest.raises(ValueError, match="timeout_seconds must be at least 1"):
            config._validate()

        config = Config(max_image_size=(0, 0))
        with pytest.raises(
            ValueError, match="max_image_size dimensions must be at least 1"
        ):
            config._validate()

    def test_config_string_representation(self) -> None:
        config_str = str(Config())
        assert "Config(" in config_str
        assert "backend=photon" in config_str
        assert "moondream3.1-9B-A2B" in config_str
