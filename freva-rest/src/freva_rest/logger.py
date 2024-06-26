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
LOG_DIR.mkdir(exist_ok=True, parents=True)
logger_format = logging.Formatter(
    "%(name)s - %(asctime)s - %(levelname)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
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

logger.addHandler(logger_file_handle)
logger.addHandler(logger_stream_handle)
