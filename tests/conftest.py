"""Pytest configuration settings."""

import asyncio
import inspect
import json
import os
import shutil
import socket
import threading
import time
from base64 import b64encode
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, AsyncIterator, Dict, Iterator, Tuple

import mock
import pytest
import requests
import uvicorn
from typer.testing import CliRunner

from data_portal_worker.cli import _main as run_data_loader
from freva_client.auth import Auth
from freva_client.utils import logger
from freva_rest.api import app
from freva_rest.config import ServerConfig
from freva_rest.databrowser_api.mock import read_data
from freva_rest.utils import create_redis_connection


async def Run(*cmd: str, stdout: Any = None, **kwargs: Any) -> Any:
    """Mock the execution of a sub command."""

    class Writer:
        """Mock the return value of asyncio.create_subprocess_exec."""

        def __init__(self, returncode: int):
            self.stderr = None
            self.returncode = returncode

        @property
        async def stdout(self) -> AsyncIterator[bytes]:
            for out in ("foo", "bar"):
                yield out.encode("utf-8")
                yield "\n".encode("utf-8")

        async def wait(self) -> None:
            """This should have a wait method."""
            await asyncio.sleep(0.5)

        async def communicate(self) -> Tuple[bytes, bytes]:
            """Mock the communicate method."""
            return self.stdout, self.stderr

    return_code = int(os.getenv("MOCK_RETURN_CODE", "0"))
    if cmd[0] == "rm":
        shutil.rmtree(cmd[-1])
        if return_code != 0:
            raise RuntimeError("Mock command failed.")
    elif cmd[0] == "git":
        # We should execute this command.
        os.system(" ".join(cmd))
    if hasattr(stdout, "write") and inspect.iscoroutinefunction(stdout.write):
        await stdout.write(f"Running {' '.join(cmd)}\n")
        await asyncio.sleep(2)
    elif hasattr(stdout, "write"):
        stdout.write(f"Running {' '.join(cmd)}\n".encode("utf-8"))

    return Writer(return_code)


def run_test_server(port: int) -> None:
    """Start a test server using uvcorn."""
    logger.setLevel(10)
    with mock.patch("freva_rest.databrowser_api.endpoints.Solr.batch_size", 3):
        with mock.patch(
            "freva_rest.databrowser_api.endpoints.server_config.proxy",
            f"http://localhost:{port}",
        ):
            uvicorn.run(
                app,
                host="0.0.0.0",
                port=port,
                reload=False,
                workers=None,
                loop="uvloop",
            )


def find_free_port() -> int:
    """Get a free port where we can start the test server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def get_data_loader_config() -> bytes:
    """Create the config for the data-loader process."""
    return b64encode(
        json.dumps(
            {
                "user": os.getenv("API_REDIS_USER"),
                "passwd": os.getenv("API_REDIS_PASSWORD"),
                "host": os.getenv("API_REDIS_HOST"),
                "ssl_cert": Path(os.getenv("API_REDIS_SSL_CERTFILE")).read_text(),
                "ssl_key": Path(os.getenv("API_REDIS_SSL_KEYFILE")).read_text(),
            }
        ).encode("utf-8")
    )


def run_loader_process(port: int) -> None:
    """Start the data loader process."""
    with NamedTemporaryFile(suffix=".json") as temp_f:
        Path(temp_f.name).write_bytes(get_data_loader_config())
        run_data_loader(Path(temp_f.name), port=port, dev=True)


async def flush_cache() -> None:
    """Flush the redis cache."""
    cache = await create_redis_connection()
    await cache.flushdb()


async def shutdown_data_loader() -> None:
    """Cancel the data loader process."""
    cache = await create_redis_connection()
    for _ in range(5):
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


def setup_server() -> Iterator[str]:
    """Start the test server."""
    port = find_free_port()
    asyncio.run(flush_cache())
    thread1 = threading.Thread(target=run_test_server, args=(port,))
    thread1.daemon = True
    thread1.start()
    time.sleep(1)
    thread2 = threading.Thread(
        target=run_loader_process, args=(find_free_port(),)
    )
    thread2.daemon = True
    thread2.start()
    time.sleep(5)
    yield f"http://localhost:{port}/api/freva-nextgen"


async def drop_mongo_collections() -> None:
    """Wipe the mongoDB."""
    conf = ServerConfig()
    db = conf.mongo_client[conf.mongo_db]
    collections = await db.list_collection_names()
    for col in collections:
        await db.drop_collection(col)


@pytest.fixture(scope="session")
def loader_config() -> Iterator[bytes]:
    """Fixture to provide the data-loader config."""
    yield get_data_loader_config()


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
            freva_config.write_text(
                "[freva]\nhost = 'https://www.freva.com:80/api'"
            )
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
            with mock.patch(
                "sysconfig.get_path", lambda x, y="foo": str(temp_dir)
            ):
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
            with mock.patch(
                "sysconfig.get_path", lambda x, y="foo": str(temp_dir)
            ):
                yield eval_file


@pytest.fixture(scope="module")
def cfg() -> Iterator[ServerConfig]:
    """Create a valid server config."""
    conf = ServerConfig(debug=True)
    for core in conf.solr_cores:
        asyncio.run(read_data(core, conf.solr_url))
    yield conf


@pytest.fixture(scope="session", autouse=True)
def test_server() -> Iterator[str]:
    """Setup a new instance of a test server while mocking an environment."""
    env = os.environ.copy()
    for key in (
        "REDIS_HOST",
        "REDIS_USER",
        "REDIS_SSL_KEYFILE",
        "REDIS_SSL_CERTFILE",
    ):
        env[key] = os.getenv(f"API_{key}", "")
    env["REDIS_PASS"] = os.getenv("API_REDIS_PASSWORD")
    with mock.patch.dict(os.environ, env, clear=True):
        asyncio.run(drop_mongo_collections())
        yield from setup_server()
        asyncio.run(shutdown_data_loader())
        asyncio.run(flush_cache())
        asyncio.run(drop_mongo_collections())


@pytest.fixture(scope="module")
def auth(test_server: str) -> Iterator[Dict[str, str]]:
    """Create a valid acccess token."""
    res = requests.post(
        f"{test_server}/auth/v2/token",
        data={
            "username": "janedoe",
            "password": "janedoe123",
            "grant_type": "password",
        },
    )
    yield res.json()


@pytest.fixture(scope="function")
def user_data_payload_sample() -> Dict[str, list[str]]:
    """Create a user data payload."""
    return {
        "user_metadata": [
            {
                "variable": "ua",
                "time_frequency": "mon",
                "time": "[1979-01-16T12:00:00Z TO 1979-11-16T00:00:00Z]",
                "cmor_table": "mon",
                "version": "",
                "file": (
                    "freva-rest/src/freva_rest/databrowser_api/mock/data/model"
                    "/global/cmip6/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/amip"
                    "/r2i1p1f1/Amon/ua/gn/v20190815/"
                    "ua_mon_MPI-ESM1-2-LR_amip_r2i1p1f1_gn_197901-199812.nc"
                ),
            },
            {
                "variable": "ua",
                "time_frequency": "mon",
                "time": "[1979-01-16T12:00:00Z TO 1979-11-16T00:00:00Z]",
                "cmor_table": "mon",
                "version": "",
                "file": (
                    "freva-rest/src/freva_rest/databrowser_api/mock/data/model/"
                    "global/cmip6/CMIP6/CMIP/CSIRO-ARCCSS/ACCESS-CM2/"
                    "amip/r1i1p1f1/Amon/ua/gn/v20201108/"
                    "ua_Amon_ACCESS-CM2_amip_r1i1p1f1_gn_197901-201412.nc"
                ),
            },
            {
                "variable": "ua",
                "time_frequency": "mon",
                "time": "[1979-01-16T12:00:00Z TO 1979-11-16T00:00:00Z]",
                "cmor_table": "mon",
                "version": "",
                "file": (
                    "freva-rest/src/freva_rest/databrowser_api/mock/data/model/"
                    "global/cmip6/CMIP6/CMIP/CSIRO-ARCCSS/ACCESS-CM2/"
                    "amip/r1i1p1f1/Amon/ua/gn/v20191108/"
                    "ua_Amon_ACCESS-CM2_amip_r1i1p1f1_gn_197001-201512.nc"
                ),
            },
            {
                "variable": "pr",
                "time_frequency": "3h",
                "time": "[2007-01-02T01:30:00Z TO 2007-01-02T04:30:00Z]",
                "cmor_table": "3h",
                "version": "",
                "file": (
                    "freva-rest/src/freva_rest/databrowser_api/mock/data/model/"
                    "regional/cordex/output/EUR-11/GERICS/NCC-NorESM1-M/"
                    "rcp85/r1i1p1/GERICS-REMO2015/v1/3hr/pr/v20181212/"
                    "pr_EUR-11_NCC-NorESM1-M_rcp85_r1i1p1_GERICS-REMO2015"
                    "_v2_3hr_200701020130-200701020430.nc"
                ),
            },
            {
                "variable": "tas",
                "time_frequency": "day",
                "time": "[1949-12-01T12:00:00Z TO 1949-12-10T12:00:00Z]",
                "cmor_table": "day",
                "version": "",
                "file": (
                    "freva-rest/src/freva_rest/databrowser_api/mock/data/model/"
                    "regional/cordex/output/EUR-11/CLMcom/"
                    "MPI-M-MPI-ESM-LR/historical/r0i0p0/"
                    "CLMcom-CCLM4-8-17/v1/fx/orog/v20140515/"
                    "orog_EUR-11_MPI-M-MPI-ESM-LR_historical_r1i1p1_"
                    "CLMcom-CCLM4-8-17_v1_fx.nc"
                ),
            },
            {
                "variable": "tas",
                "time_frequency": "day",
                "time": "[1949-12-01T12:00:00Z TO 1949-12-10T12:00:00Z]",
                "cmor_table": "day",
                "version": "",
                "file": (
                    "freva-rest/src/freva_rest/databrowser_api/mock/data/model/"
                    "regional/cordex/output/EUR-11/CLMcom/"
                    "MPI-M-MPI-ESM-LR/historical/r1i1p1/"
                    "CLMcom-CCLM4-8-17/v1/daypt/tas/v20140515/"
                    "tas_EUR-11_MPI-M-MPI-ESM-LR_historical_r1i1p1_"
                    "CLMcom-CCLM4-8-17_v1_daypt_194912011200-194912101200.nc"
                ),
            },
        ],
        "facets": {"project": "cmip5", "experiment": "myFavExp"},
    }


@pytest.fixture(scope="function")
def user_data_payload_sample_partially_success() -> Dict[str, list[str]]:
    return {
        "user_metadata": [
            {
                "variable": "ua",
                "time_frequency": "mon",
                "time": "[1979-01-16T12:00:00Z TO 1979-11-16T00:00:00Z]",
                "cmor_table": "mon",
                "version": "",
                "file": (
                    "freva-rest/src/freva_rest/databrowser_api/mock/data/model/"
                    "global/cmip6/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/amip/"
                    "r2i1p1f1/Amon/ua/gn/v20190815/"
                    "ua_mon_MPI-ESM1-2-LR_amip_r2i1p1f1_gn_197901-199812.nc"
                ),
            },
            {
                "variable": "ua",
                "time_frequency": "mon",
                "time": "[1979-01-16T12:00:00Z TO 1979-11-16T00:00:00Z]",
                "cmor_table": "mon",
                "version": "",
                "file": "",
            },
            {
                "variable": "tas",
                "time_frequency": "day",
                "time": "fx",
                "file": (
                    "freva-rest/src/freva_rest/databrowser_api/mock/data/model/"
                    "regional/cordex/output/EUR-11/CLMcom/"
                    "MPI-M-MPI-ESM-LR/historical/r0i0p0/"
                    "CLMcom-CCLM4-8-17/v1/fx/orog/v20140515/"
                    "orog_EUR-11_MPI-M-MPI-ESM-LR_historical_r1i1p1_"
                    "CLMcom-CCLM4-8-17_v1_fx.nc"
                ),
            },
        ],
        "facets": {"project": "cmip5", "experiment": "myFavExp"},
    }


@pytest.fixture(scope="module")
def subporcess_mocker(
    test_server: str,
) -> Iterator[Tuple[str, Dict[str, Dict[str, str]]]]:
    """Patch the subporcess run process for adding tools."""
    env = os.environ.copy()
    env["QUEUE_MAX_SIZE"] = "1"
    with mock.patch(
        "freva_rest.tool_api.utils.asyncio.create_subprocess_exec", Run
    ):
        tokens: Dict[str, Dict[str, str]] = {}
        for username in ("janedoe", "johndoe", "alicebrown"):
            res = requests.post(
                f"{test_server}/auth/v2/token",
                data={
                    "username": username,
                    "password": f"{username}123",
                    "grant_type": "password",
                },
                timeout=5,
            )
            token = res.json()
            tokens[username] = {
                "Authorization": f'{token["token_type"]} {token["access_token"]}'
            }
        with mock.patch.dict(os.environ, env, clear=True):
            yield test_server, tokens
