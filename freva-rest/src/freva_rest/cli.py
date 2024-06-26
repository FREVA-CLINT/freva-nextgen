"""Command line interface (cli) for running the rest server.

Configuring
-----------

There are two fundamental different options to configure the service.

1. via the `config` ``.toml`` file.
2. via environment variables.

Note, that the order here is important. First, any configuration from the
config file is loaded, only if the configuration wasn't found in the config
file environment variables are evaluated. The following environment
variables can be set:

- ``DEBUG``: Start server in debug mode (1), (default: 0 -> no debug).
- ``API_PORT``: the port the rest service should be running on (default 8080).
- ``API_WORKER``: the number of multi-process work serving the API (default: 8).
- ``SOLR_HOST``: host name of the solr server, host name and port should be
                 separated by a ``:``, for example ``localhost:8983``
- ``SOLR_CORE`` : name of the solr core that contains datasets with multiple
                  versions
- ``MONGO_HOST``: host name of the mongodb server, where query statistics are
                 stored. Host name and port should separated by a ``:``, for
                 example ``localhost:27017``
- ``MONGO_USER``: user name for the mongodb.
- ``MONGO_PASSWORD``: password to log on to the mongodb.
- ``MONGO_DB``: database name of the mongodb instance.
- ``API_URL``: url of the machine that runs of the rest api
- ``API_CACHE_EXP``: expiry time in seconds of the cached data
- ``REDIS_HOST``: Host and port of the redis cache
                  Host name and port should separated by a ``:``, for
                  example ``localhost:5672``
- ``REDIS_PASS``: Password for the redis connection.
- ``REDIS_USER``: Username for the redis connection.
- ``REDIS_SSL_CERTFILE``: Path to the TSL certificate file used to encrypt
                          the redis connection.
- ``REDIS_SSL_KEYFILE``: Path to the TSL key file used to encrypt the redis
                         connection.
📝  You can override the path to the default config file using the
    ``API_CONFIG`` environment variable. The default location of this config
    file is ``/opt/databrowser/api_config.toml``.
"""

import asyncio
import logging
import os
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional, Tuple

import typer
import uvicorn

from .config import ServerConfig, defaults
from .logger import logger

cli = typer.Typer(name="freva-rest-server", help=__doc__, epilog=__doc__)


class Services(str, Enum):
    """Literal implementation for the cli."""

    zarr_stream: str = "zarr-stream"
    stream: str = "zarr-stream"
    data_portal: str = "zarr-stream"
    databrowser: str = "databrowser"
    search: str = "databrowser"


def get_cert_file(
    cert_dir: Optional[str],
    cert_file: Optional[str],
    key_file: Optional[str],
) -> Tuple[str, str]:
    """Get the certificate (key and cert) if they are configured."""
    default_cert = default_key = ""
    if cert_dir:
        default_cert = str(Path(cert_dir) / "client-cert.pem")
        default_key = str(Path(cert_dir) / "client-key.pem")
    return cert_file or default_cert, key_file or default_key


@cli.command(name="freva-rest-api")
def start(
    config_file: Optional[Path] = typer.Option(
        os.environ.get("API_CONFIG", defaults["API_CONFIG"]),
        "-c",
        "--config-file",
        help="Path to the server configuration file",
    ),
    port: int = typer.Option(
        os.environ.get("API_PORT", 8080),
        "-p",
        "--port",
        help="The port the api is running on",
    ),
    services: List[Services] = typer.Option(
        ("zarr-stream", "databrowser"),
        "-s",
        "--services",
        help="Set additional services this rest API should serve.",
    ),
    ssl_cert_dir: Optional[str] = typer.Option(
        None,
        "--cert-dir",
        help=(
            "Set the path to the directory contaning the tls cert and key files"
            " that are used to establish a secure connection, if you set the"
            " it will be assumed that cert file is saved as client-cert.pem"
            " and the key file client-key.pem. This flag can be used as a"
            " short cut instead of using the `--tls-cert` and `--tls-key` flats"
        ),
    ),
    ssl_cert: Optional[str] = typer.Option(
        os.environ.get("REDIS_SSL_CERTFILE"),
        "--tls-cert",
        help=(
            "Set the path to the tls certificate file that is used to establish"
            " a secure connection to the data portal cache."
        ),
    ),
    ssl_key: Optional[str] = typer.Option(
        os.environ.get("REDIS_SSL_KEYFILE"),
        "--tls-key",
        help=(
            "Set the path to the tls key file that is used to establish"
            " a secure connection to the data portal cache."
        ),
    ),
    dev: bool = typer.Option(False, help="Add test data to the dev solr."),
    debug: bool = typer.Option(
        bool(int(os.environ.get("DEBUG", 0))), help="Turn on debug mode."
    ),
) -> None:
    """Start the freva rest API."""
    if debug:
        logger.setLevel(logging.DEBUG)
    defaults["API_CONFIG"] = (config_file or defaults["API_CONFIG"]).absolute()
    defaults["DEBUG"] = debug
    defaults["API_CACHE_EXP"] = int(os.environ.get("API_CACHE_EXP") or "3600")
    defaults["REDIS_HOST"] = (
        os.environ.get("REDIS_HOST") or "redis://localhost:6379"
    )
    defaults["API_URL"] = (
        os.environ.get("API_URL") or f"http://localhost:{port}"
    )
    defaults["REDIS_SSL_CERTFILE"] = ssl_cert
    defaults["REDIS_SSL_KEYFILE"] = ssl_key
    if ssl_cert:
        ssl_cert = str(Path(ssl_cert).absolute())
    cfg = ServerConfig(defaults["API_CONFIG"], debug=debug)
    if dev:
        from databrowser_api.mock import read_data

        for core in cfg.solr_cores:
            asyncio.run(read_data(core, cfg.solr_host, cfg.solr_port))
    workers = {False: int(os.environ.get("API_WORKER", 8)), True: None}
    ssl_cert, ssl_key = get_cert_file(ssl_cert_dir, ssl_cert, ssl_key)
    keycloak_client_id = os.getenv("KEYCLOAK_CLIENT_ID", "freva")
    keycloak_client_secret = os.getenv("KEYCLOAK_CLIENT_SECRET", "")
    api_services = ",".join(services).replace("_", "-")
    with NamedTemporaryFile(suffix=".conf", prefix="env") as temp_f:
        Path(temp_f.name).write_text(
            (
                f"DEBUG={int(debug)}\n"
                f"API_CONFIG={defaults['API_CONFIG']}\n"
                f"API_PORT={port}\n"
                f"API_CACHE_EXP={defaults['API_CACHE_EXP']}\n"
                f"REDIS_HOST={defaults['REDIS_HOST']}\n"
                f"REDIS_PASS={os.getenv('REDIS_PASS', 'secret')}\n"
                f"REDIS_USER={os.getenv('REDIS_USER', 'redis')}\n"
                f"REDIS_SSL_CERTFILE={ssl_cert or ''}\n"
                f"REDIS_SSL_KEYFILE={ssl_key or ''}\n"
                f"KEYCLOAK_HOST={os.getenv('KEYCLOAK_HOST', 'http://localhost:8080')}\n"
                f"KEYCLOAK_REALM={os.getenv('KEYCLOAK_REALM', 'freva')}\n"
                f"KEYCLOAK_CLIENT_ID={keycloak_client_id}\n"
                f"KEYCLOAK_CLIENT_SECRET={keycloak_client_secret}\n"
                f"API_URL={defaults['API_URL']}\n"
                f"API_SERVICES={api_services}\n"
            ),
            encoding="utf-8",
        )
        uvicorn.run(
            "freva_rest.api:app",
            host="0.0.0.0",
            port=port,
            reload=dev,
            log_level=cfg.log_level,
            workers=workers[dev],
            env_file=temp_f.name,
        )


if __name__ == "__main__":
    cli()
