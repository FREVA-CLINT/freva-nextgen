"""Definition of the central logging system."""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

import rich.logging
from rich.console import Console

THIS_NAME: str = "freva-rest"
LOG_DIR: Path = Path(
    os.environ.get("API_LOGDIR") or Path(f"/tmp/log/{THIS_NAME}")
)
logfmt = "%(name)s - %(message)s"
datefmt = "%Y-%m-%dT%H:%M"
LOG_DIR.mkdir(exist_ok=True, parents=True)
logger_format = logging.Formatter(
    "%(asctime)s %(levelname)s %(name)s - %(message)s", datefmt
)
logger_file_handle = RotatingFileHandler(
    LOG_DIR / f"{THIS_NAME}.log",
    mode="a",
    maxBytes=5 * 1024**2,
    backupCount=2,
    encoding="utf-8",
    delay=False,
)
logger_file_handle.setFormatter(logger_format)
logger_file_handle.setLevel(logging.INFO)
logger_stream_handle = rich.logging.RichHandler(
    rich_tracebacks=True,
    show_path=True,
    console=Console(
        soft_wrap=False,
        force_jupyter=False,
        stderr=True,
    ),
)
logger_stream_handle.setLevel(logging.INFO)
logger = logging.getLogger(THIS_NAME)
logger.setLevel(logging.INFO)


logging.basicConfig(
    level=logging.INFO,
    format=logfmt,
    datefmt=datefmt,
    handlers=[logger_file_handle, logger_stream_handle],
)


def reset_loggers() -> None:
    """Unify all loggers that we have currently aboard."""
    for name in logging.root.manager.loggerDict.keys():
        if name != THIS_NAME:
            logging.getLogger(name).handlers = [
                logger_file_handle,
                logger_stream_handle,
            ]
            logging.getLogger(name).propagate = True
            logging.getLogger(name).level = logging.INFO


reset_loggers()
