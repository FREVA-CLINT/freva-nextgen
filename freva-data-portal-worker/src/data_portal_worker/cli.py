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

from .load_data import CLIENT, ProcessQueue, RedisKw
from .utils import data_logger

__version__ = "2406.0.0"


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
        prog="Data Loder",
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
    args = parser.parse_args(argv)
    if args.verbose is True:
        data_logger.setLevel(logging.DEBUG)
    data_logger.debug("Loading cluster config from %s", args.config_file)
    cache_config: RedisKw = json.loads(b64decode(args.config_file.read_bytes()))
    env = os.environ.copy()
    try:
        os.environ["DASK_PORT"] = str(args.port)
        os.environ["API_CACHE_EXP"] = str(args.exp)
        os.environ["REDIS_HOST"] = args.redis_host
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
            queue = ProcessQueue(dev_mode=args.dev)
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
