"""Freva the Free Evaluation System command line interface."""

from typing import Optional

import typer
from freva_client.utils import logger

from .auth_cli import authenticate_cli
from .cli_utils import APP_NAME, version_callback
from .databrowser_cli import databrowser_app

app = typer.Typer(
    name=APP_NAME,
    help=__doc__,
    add_completion=True,
    callback=logger.set_cli,
)


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-V",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """The main cli app."""


app.add_typer(databrowser_app, name="databrowser")
app.command(name="auth", help=authenticate_cli.__doc__)(authenticate_cli)
