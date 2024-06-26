"""Pytest configuration settings."""

import asyncio
import json
import os
import socket
import threading
import time
from base64 import b64encode
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Dict, Iterator

import mock
import pytest
import uvicorn
from data_portal_worker import run_data_loader
from databrowser_api.mock import read_data
from fastapi.testclient import TestClient
from freva_client.auth import Auth
from freva_client.utils import logger
from freva_rest.api import app
from freva_rest.config import ServerConfig, defaults
from freva_rest.utils import create_redis_connection
from typer.testing import CliRunner


def run_test_server(port: int) -> None:
    """Start a test server using uvcorn."""
    logger.setLevel(10)
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False, workers=None)


def find_free_port() -> int:
    """Get a free port where we can start the test server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def run_loader_process(port: int) -> None:
    """Start the data loader process."""
    redis_host = os.getenv("REDIS_HOST")
    with NamedTemporaryFile(suffix=".json") as temp_f:
        Path(temp_f.name).write_bytes(
            b64encode(
                json.dumps(
                    {
                        "user": os.getenv("REDIS_USER"),
                        "passwd": os.getenv("REDIS_PASS"),
                        "host": redis_host,
                        "ssl_cert": Path(os.getenv("REDIS_SSL_CERTFILE")).read_text(),
                        "ssl_key": Path(os.getenv("REDIS_SSL_KEYFILE")).read_text(),
                    }
                ).encode("utf-8")
            )
        )
        args = [
            "-p",
            str(port),
            "-v",
            "--dev",
            "-c",
            temp_f.name,
        ]
        run_data_loader(args)


async def flush_cache() -> None:
    """Flush the redis cache."""
    cache = await create_redis_connection()
    await cache.flushdb()


async def shutdown_data_loader() -> None:
    """Cancel the data loader process."""
    cache = await create_redis_connection()
    for i in range(5):
        await cache.publish(
            "data-portal", json.dumps({"shutdown": True}).encode("utf-8")
        )
        time.sleep(1)


def _prep_env(**config: str) -> Dict[str, str]:
    env = os.environ.copy()
    config = config or {}
    for key in ("FREVA_CONFIG", "EVALUATION_SYSTEM_CONFIG_FILE"):
        _ = env.pop(key, "")
    for key, value in config.items():
        env[key] = value
    return env


@pytest.fixture(scope="function")
def auth_instance() -> Iterator[Auth]:
    """Fixture to provide a fresh Auth instance for each test."""
    with mock.patch("freva_client.auth.getpass", lambda x: "janedoe123"):
        with mock.patch("freva_client.auth.getuser", lambda: "janedoe"):
            yield Auth()


@pytest.fixture(scope="function")
def cli_runner() -> Iterator[CliRunner]:
    """Set up a cli mock app."""
    yield CliRunner(mix_stderr=False)
    logger.reset_cli()


@pytest.fixture(scope="function")
def valid_freva_config() -> Iterator[Path]:
    """Mock a valid freva config path."""
    with mock.patch.dict(os.environ, _prep_env(), clear=True):
        with TemporaryDirectory() as temp_dir:
            freva_config = Path(temp_dir) / "share" / "freva" / "freva.toml"
            freva_config.parent.mkdir(exist_ok=True, parents=True)
            freva_config.write_text("[freva]\nhost = 'https://www.freva.com:80/api'")
            yield Path(temp_dir)


@pytest.fixture(scope="function")
def invalid_freva_conf_file() -> Iterator[Path]:
    """Mock a broken freva config."""
    with TemporaryDirectory() as temp_dir:
        freva_config = Path(temp_dir) / "share" / "freva" / "freva.toml"
        freva_config.parent.mkdir(parents=True)
        with mock.patch.dict(
            os.environ,
            _prep_env(FREVA_CONFIG=str(freva_config)),
            clear=True,
        ):
            freva_config.write_text("[freva]\nhost = https://freva_conf/api")
            yield freva_config


@pytest.fixture(scope="function")
def free_port() -> Iterator[int]:
    """Define a free port to run stuff on."""
    yield find_free_port()


@pytest.fixture(scope="function")
def valid_eval_conf_file() -> Iterator[Path]:
    """Mock a valid evaluation config file."""
    with TemporaryDirectory() as temp_dir:
        eval_file = Path(temp_dir) / "eval.conf"
        eval_file.write_text(
            "[evaluation_system]\n"
            "solr.host = https://www.eval.conf:8081/api\n"
            "databrowser.port = 8080"
        )
        with mock.patch.dict(
            os.environ,
            _prep_env(EVALUATION_SYSTEM_CONFIG_FILE=str(eval_file)),
            clear=True,
        ):
            with mock.patch("sysconfig.get_path", lambda x, y="foo": str(temp_dir)):
                yield eval_file


@pytest.fixture(scope="function")
def invalid_eval_conf_file() -> Iterator[Path]:
    """Mock an invalid evaluation config file."""
    with TemporaryDirectory() as temp_dir:
        eval_file = Path(temp_dir) / "eval.conf"
        eval_file.write_text(
            "[foo]\n" "solr.host = http://localhost\n" "databrowser.port = 8080"
        )
        with mock.patch.dict(
            os.environ,
            _prep_env(EVALUATION_SYSTEM_CONFIG_FILE=str(eval_file)),
            clear=True,
        ):
            with mock.patch("sysconfig.get_path", lambda x, y="foo": str(temp_dir)):
                yield eval_file


@pytest.fixture(scope="session", autouse=True)
def test_server() -> Iterator[str]:
    """Start the test server."""
    env = os.environ.copy()
    port = find_free_port()
    env["API_URL"] = f"http://127.0.0.1:{port}"
    with mock.patch.dict(os.environ, env, clear=True):
        asyncio.run(flush_cache())
        thread1 = threading.Thread(target=run_test_server, args=(port,))
        thread1.daemon = True
        thread1.start()
        time.sleep(1)
        thread2 = threading.Thread(target=run_loader_process, args=(find_free_port(),))
        thread2.daemon = True
        thread2.start()
        time.sleep(5)
        yield env["API_URL"]
        asyncio.run(shutdown_data_loader())
        asyncio.run(flush_cache())


@pytest.fixture(scope="module")
def cfg() -> Iterator[ServerConfig]:
    """Create a valid server config."""
    cfg = ServerConfig(defaults["API_CONFIG"], debug=True)
    for core in cfg.solr_cores:
        asyncio.run(read_data(core, cfg.solr_host, cfg.solr_port))
    yield cfg


@pytest.fixture(scope="module")
def client(cfg: ServerConfig) -> Iterator[TestClient]:
    """Setup the test client for the unit test."""
    with mock.patch("databrowser_api.endpoints.SolrSearch.batch_size", 3):
        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture(scope="function")
def client_no_mongo(cfg: ServerConfig) -> Iterator[TestClient]:
    """Setup a client with an invalid mongodb."""
    env = os.environ.copy()
    env["MONGO_HOST"] = "foo.bar.de"
    with mock.patch.dict(os.environ, env, clear=True):
        cfg = ServerConfig(defaults["API_CONFIG"], debug=True)
        for core in cfg.solr_cores:
            asyncio.run(read_data(core, cfg.solr_host, cfg.solr_port))
        with mock.patch("freva_rest.rest.server_config.mongo_collection", None):
            with TestClient(app) as test_client:
                yield test_client


@pytest.fixture(scope="function")
def client_no_solr(cfg: ServerConfig) -> Iterator[TestClient]:
    """Setup a client with an invalid mongodb."""
    env = os.environ.copy()
    env["SOLR_HOST"] = "foo.bar.de"
    with mock.patch.dict(os.environ, env, clear=True):
        ServerConfig(defaults["API_CONFIG"], debug=True)
        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture(scope="module")
def auth(client) -> Iterator[Dict[str, str]]:
    """Create a valid acccess token."""
    res = client.post(
        "/api/auth/v2/token",
        data={"username": "janedoe", "password": "janedoe123"},
    )
    yield res.json()
