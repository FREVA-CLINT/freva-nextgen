"""Utilities for the command line interface."""

from typing import Dict, List

import typer
from freva_client import __version__
from freva_client.utils import logger
from rich import print as pprint

APP_NAME: str = "freva-client"


def version_callback(version: bool) -> None:
    """Print the version and exit."""
    if version:
        pprint(f"{APP_NAME}: {__version__}")
        raise typer.Exit()


def parse_cli_args(cli_args: List[str]) -> Dict[str, List[str]]:
    """Convert the cli arguments to a dictionary."""
    logger.debug("parsing command line arguments.")
    kwargs = {}
    for entry in cli_args:
        key, _, value = entry.partition("=")
        if value and key not in kwargs:
            kwargs[key] = [value]
        elif value:
            kwargs[key].append(value)
    logger.debug(kwargs)
    return kwargs
