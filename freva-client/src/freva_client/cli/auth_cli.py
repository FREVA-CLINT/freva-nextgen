"""Command line interface for authentication."""

import json
import os
from typing import Optional

import typer

from freva_client.auth import Auth
from freva_client.utils import exception_handler, logger
from freva_client.utils.auth_utils import TOKEN_ENV_VAR, get_default_token_file

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
    token_file: str = typer.Option(
        os.getenv(TOKEN_ENV_VAR, "").strip(),
        "--token-file",
        help=(
            "Instead of authenticating via code based authentication flow "
            "you can set the path to the json file that contains a "
            "`refresh token` containing a refresh_token key."
        ),
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force token recreation, even if current token is still valid.",
    ),
    verbose: int = typer.Option(0, "-v", help="Increase verbosity", count=True),
    version: Optional[bool] = typer.Option(
        False,
        "-V",
        "--version",
        help="Show version an exit",
        callback=version_callback,
    ),
) -> None:
    """Create OAuth2 access and refresh token."""
    logger.set_verbosity(verbose)
    token = Auth(token_file=token_file or get_default_token_file()).authenticate(
        host=host,
        force=force,
        _cli=True,
    )
    print(json.dumps(token, indent=3))
