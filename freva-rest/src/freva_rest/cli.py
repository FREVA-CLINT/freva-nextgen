"""# Command line interface (cli) for running the rest server.

## Configuring


There are three fundamental different options to configure the service.

1. via command line arguments
1. via environment variables.
1. via the `config` ``.toml`` file.

Note, that the order here is important. First, given command line arguments
are evaluated, if present any of the below given environment variables
are evaluated, finally the api config file is evaluated. The following
environment variables are evaluated:

- ``DEBUG``: Start server in debug mode (1), (default: 0 -> no debug).
- ``API_PORT``: the port the rest service should be running on (default 8080).
- ``API_WORKER``: the number of multi-process work serving the API (default: 8).
- ``API_SOLR_HOST``: host name of the solr server, host name and port should be
                 separated by a ``:``, for example ``localhost:8983``
- ``API_SOLR_CORE`` : name of the solr core that contains datasets with multiple
                  versions
- ``API_MONGO_HOST``: host name of the mongodb server, where query statistics are
                 stored. Host name and port should separated by a ``:``, for
                 example ``localhost:27017``
- ``API_MONGO_USER``: user name for the mongodb.
- ``API_MONGO_PASSWORD``: password to log on to the mongodb.
- ``API_MONGO_DB``: database name of the mongodb instance.
- ``API_PROXY``: url of a proxy that servers the API - if any.
- ``API_CACHE_EXP``: expiry time in seconds of the cached data
- ``API_REDIS_HOST``: Host and port of the redis cache
                  Host name and port should separated by a ``:``, for
                  example ``localhost:5672``
- ``API_REDIS_PASSWORD``: Password for the redis connection.
- ``API_REDIS_USER``: Username for the redis connection.
- ``API_REDIS_SSL_CERTFILE``: Path to the TSL certificate file used to encrypt
                          the redis connection.
- ``API_REDIS_SSL_KEYFILE``: Path to the TSL key file used to encrypt the redis
                         connection.
- ``API_OIDC_URL``: Discovery of the open connect id service.
- ``API_OIDC_CLIENT_ID``: Name of the client (app) that is used to create
                          the access tokens, defaults to freva
- ``API_OIDC_CLIENT_SECRET``: You can set a client secret, if you have
- ``API_SERVICES``:  The services the api should serve.

📝  You can override the path to the default config file using the
    ``API_CONFIG`` environment variable. The default location of this config
    file is ``/opt/databrowser/api_config.toml``.
"""

import argparse
import asyncio
import logging
import os
from enum import Enum
from pathlib import Path
from socket import gethostname
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, Tuple, Union

import uvicorn
from pydantic.fields import FieldInfo
from rich.markdown import Markdown
from rich_argparse import ArgumentDefaultsRichHelpFormatter

from .config import ServerConfig
from .logger import logger


def create_arg_parser(
    fields: Dict[str, FieldInfo], forbidden: List[str]
) -> argparse.ArgumentParser:
    """Create the cli parser."""
    parser = argparse.ArgumentParser(
        prog="freva-rest-server",
        description=Markdown(__doc__),  # type: ignore
        formatter_class=ArgumentDefaultsRichHelpFormatter,
    )
    parser.add_argument(
        "--port",
        "-p",
        help="Set the port the server should be running on.",
        default=7777,
        type=int,
    )
    for key, field in fields.items():
        name = key.replace("_", "-")
        if field.annotation in (
            bool,
            Union[str, int, bool],
            Union[int, bool],
        ) or isinstance(field.default, bool):
            parser.add_argument(
                f"--{name}",
                help=field.description,
                action="store_true",
                default=False,
            )

        elif name not in forbidden:
            parser.add_argument(
                f"--{name}",
                help=field.description,
                default=field.default or None,
                type=type(field.default),
            )
    return parser


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


def cli(argv: Optional[List[str]] = None) -> None:
    """Start the freva rest API."""
    cfg = ServerConfig()
    parser = create_arg_parser(cfg.__fields__, ["api-services"])
    parser.add_argument("--dev", action="store_true", help="Enable development mode")
    parser.add_argument(
        "--n-workers",
        "-w",
        help="Number of parallel processes.",
        default=os.getenv("API_WORKER", "8"),
        type=int,
    )
    parser.add_argument(
        "--redis-ssl-certdir",
        help=(
            "The directory where the certficates are stored."
            " This can be used to instead of setting the cert, and key files."
        ),
        type=Path,
        default=None,
    ),
    parser.add_argument(
        "--services",
        "-s",
        help="The services the API should serve",
        nargs="+",
        default=os.getenv("API_SERVICES", "").split(","),
        choices=["databrowser", "zarr-stream"],
    )

    args = parser.parse_args(argv)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    ssl_cert, ssl_key = get_cert_file(
        args.redis_ssl_certdir, args.redis_ssl_certfile, args.redis_ssl_keyfile
    )
    defaults: Dict[str, str] = {
        "API_CONFIG": str(Path(args.config).expanduser().absolute()),
        "DEBUG": str(int(args.debug)),
        "API_SERVICES": ",".join(args.services or "").replace("_", "-"),
        "API_REDIS_SSL_KEYFILE": str(ssl_key or ""),
        "API_REDIS_SSL_CERTFILE": str(ssl_cert or ""),
        "API_PROXY": args.proxy or f"http://{gethostname()}:{args.port}",
    }
    cfg = ServerConfig(config=args.config, debug=args.debug)
    for key, value in args._get_kwargs():
        name = key.upper().removeprefix("API_")
        if key in cfg.model_fields_set and key not in ["debug"]:
            defaults.setdefault(f"API_{name}", str(value or ""))
    if args.dev:
        from freva_rest.databrowser_api.mock import read_data

        for core in cfg.solr_cores:
            asyncio.run(read_data(core, cfg.solr_url))
    workers = {False: args.n_workers, True: None}
    with NamedTemporaryFile(suffix=".conf", prefix="env") as temp_f:
        env = "\n".join(
            [f"{k}={v.strip()}" for (k, v) in defaults.items() if v.strip()]
        )
        Path(temp_f.name).write_text(env, encoding="utf-8")
        uvicorn.run(
            "freva_rest.api:app",
            host="0.0.0.0",
            port=args.port,
            reload=args.dev,
            log_level=cfg.log_level,
            workers=workers[args.dev],
            env_file=temp_f.name,
        )


if __name__ == "__main__":
    cli()
