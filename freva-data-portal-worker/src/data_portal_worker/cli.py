"""The freva data loading portal."""

import argparse
import json
import logging
import os
from base64 import b64decode
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional

import appdirs
from watchfiles import run_process

from ._version import __version__
from .load_data import CLIENT, ProcessQueue, RedisKw
from .utils import data_logger


def _main(
    config_file: Path,
    port: int = 40000,
    exp: int = 3600,
    redis_host: str = "redis://localhost:6379",
    dev: bool = False,
) -> None:
    """Run the loader process."""
    data_logger.debug("Loading cluster config from %s", config_file)
    cache_config: RedisKw = json.loads(b64decode(config_file.read_bytes()))
    env = os.environ.copy()
    try:
        os.environ["DASK_PORT"] = str(port)
        os.environ["API_CACHE_EXP"] = str(exp)
        os.environ["REDIS_HOST"] = redis_host
        os.environ["REDIS_USER"] = cache_config["user"]
        os.environ["REDIS_PASS"] = cache_config["passwd"]
        with TemporaryDirectory() as temp:
            if cache_config["ssl_cert"] and cache_config["ssl_key"]:
                cert_file = Path(temp) / "client-cert.pem"
                key_file = Path(temp) / "client-key.pem"
                cert_file.write_text(cache_config["ssl_cert"])
                key_file.write_text(cache_config["ssl_key"])
                key_file.chmod(0o600)
                cert_file.chmod(0o600)
                os.environ["REDIS_SSL_CERTFILE"] = str(cert_file)
                os.environ["REDIS_SSL_KEYFILE"] = str(key_file)

            data_logger.debug("Starting data-loader process")
            queue = ProcessQueue(dev_mode=dev)
            queue.run_for_ever("data-portal")
    except KeyboardInterrupt:
        pass
    finally:
        if CLIENT is not None:
            CLIENT.shutdown()  # pragma: no cover
        for handler in logging.root.handlers:
            handler.flush()
            handler.close()
            logging.root.removeHandler(handler)
        os.environ = env  # type: ignore


def run_data_loader(argv: Optional[List[str]] = None) -> None:
    """Daemon that waits for messages to load the data."""
    config_file = (
        Path(appdirs.user_cache_dir("freva")) / "data-portal-cluster-config.json"
    )

    redis_host, _, redis_port = (
        (os.environ.get("REDIS_HOST") or "localhost")
        .replace("redis://", "")
        .partition(":")
    )
    redis_port = redis_port or "6379"
    parser = argparse.ArgumentParser(
        prog="Data Loader",
        description=("Starts the data loading service."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config-file",
        help="Path to the config file.",
        type=Path,
        default=config_file,
    )
    parser.add_argument(
        "-e",
        "--exp",
        type=int,
        help="Set the expiry time of the redis cache.",
        default=os.environ.get("API_CACHE_EXP") or "3600",
    )
    parser.add_argument(
        "-r",
        "--redis-host",
        type=str,
        help="Host:Port of the redis cache.",
        default=f"redis://{redis_host}:{redis_port}",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        help="Dask scheduler port for loading data.",
        default=40000,
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Development mode",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Display debug messages.",
        default=False,
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=__version__),
    )
    args = parser.parse_args(argv)
    if args.verbose is True:
        data_logger.setLevel(logging.DEBUG)
    kwargs = {
        "port": args.port,
        "exp": args.exp,
        "redis_host": args.redis_host,
        "dev": args.dev,
    }
    if args.dev:
        run_process(
            Path.cwd(),
            target=_main,
            args=(args.config_file.expanduser(),),
            kwargs=kwargs,
        )
    else:
        _main(args.config_file.expanduser(), **kwargs)
