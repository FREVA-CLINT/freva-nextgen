"""Command line interface for authentication."""

import json
from getpass import getuser
from typing import Optional

import typer
from freva_client import authenticate
from freva_client.utils import exception_handler

from .cli_utils import version_callback

auth_app = typer.Typer(
    name="auth",
    help="Create OAuth2 access and refresh token.",
    pretty_exceptions_short=False,
)


@exception_handler
def authenticate_cli(
    host: Optional[str] = typer.Option(
        None,
        "--host",
        help=(
            "Set the hostname of the databrowser, if not set (default) "
            "the hostname is read from a config file"
        ),
    ),
    username: str = typer.Option(
        getuser(),
        "--username",
        "-u",
        help="The username used for authentication.",
    ),
    refresh_token: Optional[str] = typer.Option(
        None,
        "--refresh-token",
        "-r",
        help=(
            "Instead of using a password, you can use a refresh token. "
            "refresh the access token. This is recommended for non-interactive"
            " environments."
        ),
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force token recreation, even if current token is still valid.",
    ),
    version: Optional[bool] = typer.Option(
        False,
        "-V",
        "--version",
        help="Show version an exit",
        callback=version_callback,
    ),
) -> None:
    """Create OAuth2 access and refresh token."""
    token_data = authenticate(
        host=host,
        username=username,
        refresh_token=refresh_token,
        force=force,
    )
    print(json.dumps(token_data, indent=3))
