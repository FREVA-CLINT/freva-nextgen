"""Test the command line interface cli."""

import json
from copy import deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from types import TracebackType

import mock
from pytest_mock import MockerFixture
from typer.testing import CliRunner

from freva_client import __version__
from freva_client.auth import Auth
from freva_client.cli import app as cli_app
from freva_rest.cli import cli, get_cert_file


class MockTempfile:
    """Mock the NamedTemporaryFile class."""

    temp_dir: str = "/tmp"

    def __init__(self, *args: str, **kwargs: str) -> None:
        pass

    def __enter__(self) -> "MockTempfile":
        return self

    def __exit__(
        self, exc_type: type, exc_value: Exception, traceback: TracebackType
    ) -> None:
        Path(self.name).unlink()

    @property
    def name(self) -> str:
        """Mock the Tempfile.name method"""
        return str(Path(self.temp_dir) / "foo.txt")


def test_get_cert_file() -> None:
    """Test getting the cerfiles."""
    out1, out2 = get_cert_file(None, None, None)
    assert out1 == out2 == ""
    out1, out2 = get_cert_file("foo", None, None)
    assert out1.startswith("foo")
    assert out2.startswith("foo")
    out1, out2 = get_cert_file("foo", None, "bar")
    assert out1.startswith("foo")
    assert out2.startswith("bar")
    out1, out2 = get_cert_file("foo", "bar", "baz")
    assert out1.startswith("bar")
    assert out2.startswith("baz")


def test_data_loader_cli(mocker: MockerFixture, loader_config: bytes) -> None:
    """Test the data-loader command line interface."""
    mock_run = mocker.patch("data_portal_worker.cli._main")
    mock_reload = mocker.patch("data_portal_worker.cli.run_process")
    with NamedTemporaryFile(suffix=".json") as temp_f:
        Path(temp_f.name).write_bytes(loader_config)
        from data_portal_worker.cli import run_data_loader

        run_data_loader(
            [
                "-c",
                temp_f.name,
                "-r",
                "redis://example.com:1234",
                "-p",
                "1234",
                "-e",
                "20",
                "-v",
                "--dev",
            ]
        )
        mock_reload.assert_called_once_with(
            Path.cwd(),
            target=mock_run,
            args=(Path(temp_f.name),),
            kwargs=dict(
                port=1234,
                exp=20,
                redis_host="redis://example.com:1234",
                dev=True,
            ),
        )
        run_data_loader(
            [
                "-c",
                temp_f.name,
                "-r",
                "redis://example.com:1234",
                "-p",
                "4321",
                "-e",
                "10",
            ]
        )
        mock_run.assert_called_once_with(
            Path(temp_f.name),
            port=4321,
            exp=10,
            redis_host="redis://example.com:1234",
            dev=False,
        )


def test_rest_cli(mocker: MockerFixture) -> None:
    """Test the rest api command line interface."""
    mock_run = mocker.patch("uvicorn.run")
    with TemporaryDirectory() as temp_dir:
        MockTempfile.temp_dir = temp_dir
        with mock.patch("freva_rest.cli.NamedTemporaryFile", MockTempfile):
            cli(["--dev"])
            mock_run.assert_called_once_with(
                "freva_rest.api:app",
                host="0.0.0.0",
                port=7777,
                reload=True,
                log_level=20,
                workers=None,
                loop="uvloop",
                env_file=str(Path(temp_dir) / "foo.txt"),
            )
            cli(["--debug"])
            mock_run.assert_called_with(
                "freva_rest.api:app",
                host="0.0.0.0",
                port=7777,
                reload=False,
                log_level=10,
                workers=8,
                loop="uvloop",
                env_file=str(Path(temp_dir) / "foo.txt"),
            )


def test_auth(
    test_server: str, cli_runner: CliRunner, auth_instance: Auth
) -> None:
    """Test authentication."""
    old_token = deepcopy(auth_instance._auth_token)
    try:
        res = cli_runner.invoke(
            cli_app, ["auth", "--host", test_server, "-u", "janedoe"]
        )
        assert res.exit_code == 0
        assert res.stdout
        assert "access_token" in json.loads(res.stdout)
    finally:
        auth_instance._auth_token = old_token


def test_main(cli_runner: CliRunner) -> None:
    """Test the functionality of the freva_client main app."""
    res = cli_runner.invoke(cli_app, ["-V"])
    assert res.exit_code == 0
    assert res.stdout
    assert __version__ in res.stdout
