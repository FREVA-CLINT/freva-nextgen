"""Test for the configuration utility."""

import sys
from pathlib import Path

import mock
import pytest
from freva_client import databrowser


def test_invalid_eval_config(invalid_eval_conf_file: Path) -> None:
    """Test if loading an invalid evaluation system config file fails."""
    assert invalid_eval_conf_file.is_file()
    with pytest.raises(ValueError):
        databrowser()
    db = databrowser(host="www.example.com:8080")
    assert db.url == "http://www.example.com:8080/api/databrowser"


def test_invalid_freva_config(invalid_freva_conf_file: Path) -> None:
    """Test if loading an invalid freva config file fails."""
    assert invalid_freva_conf_file.is_file()
    with pytest.raises(ValueError):
        databrowser()
    db = databrowser(host="https://www.example.com")
    assert db.url == "https://www.example.com/api/databrowser"


def test_valid_eval_config(valid_eval_conf_file: Path) -> None:
    """Test if we can load an evaluation system config file."""
    assert valid_eval_conf_file.is_file()
    db = databrowser()
    assert db.url == "https://www.eval.conf:8081/api/databrowser"
    valid_eval_conf_file.write_text(
        "[evaluation_system]\ndatabrowser.host = http://www.eval.conf/api\n"
    )
    db = databrowser()
    assert db.url == "http://www.eval.conf/api/databrowser"


def test_valid_freva_config(valid_freva_config: Path) -> None:
    """Test if we can load a freva config file."""
    assert valid_freva_config.is_dir()
    # Mock osx user
    with mock.patch.object(sys, "platform", "darwin"):
        with mock.patch("sysconfig.get_config_var", lambda x: x):
            with mock.patch(
                "sysconfig.get_path",
                lambda x, y="foo": str(valid_freva_config),
            ):
                db = databrowser()
                assert db.url == "https://www.freva.com:80/api/databrowser"
    config_file = valid_freva_config / "share" / "freva" / "freva.toml"
    assert config_file.is_file()
    config_file.write_text(config_file.read_text().replace(":80", ""))
    # Mock any user
    with mock.patch.object(sys, "platform", "linux"):
        with mock.patch(
            "sysconfig.get_path",
            lambda x, y="foo": str(valid_freva_config),
        ):
            db = databrowser()
            assert db.url == "https://www.freva.com/api/databrowser"
