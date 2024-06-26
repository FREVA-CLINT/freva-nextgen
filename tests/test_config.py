"""Unit tests for the configuration."""

import logging
import os
from pathlib import Path
from typing import List

import mock
from freva_rest.config import ServerConfig, defaults
from pytest import LogCaptureFixture


def test_valid_config() -> None:
    """Test the creation of a valid config file."""

    cfg = ServerConfig(defaults["API_CONFIG"], debug=True)
    assert cfg.log_level == 10
    assert cfg.solr_host == "localhost"
    defaults["DEBUG"] = False
    cfg.reload()
    assert cfg.solr_host == "localhost"


def test_invalid_config(caplog: LogCaptureFixture) -> None:
    """Test the behaviour for an invalid config file."""
    caplog.clear()
    _ = ServerConfig(Path("/foo/bar.toml"), debug=True)
    assert caplog.records
    records: List[logging.LogRecord] = caplog.records
    assert any([record.levelname == "CRITICAL" for record in records])
    assert any(["Failed to load" in record.message for record in records])


def test_keycloak_url() -> None:
    """Test if the correct keycload url."""
    env = os.environ.copy()
    env["KEYCLOAK_HOST"] = "localhost"
    with mock.patch.dict(os.environ, env, clear=True):
        assert ServerConfig.get_keycloak_url().startswith("https://")
    env["KEYCLOAK_HOST"] = "http://localhost"
    with mock.patch.dict(os.environ, env, clear=True):
        assert ServerConfig.get_keycloak_url().startswith("http://")
