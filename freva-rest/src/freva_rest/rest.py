"""## Welcome to the RestAPI for freva 🧉

Freva, the free evaluation system framework, is a data search and analysis
platform developed by the atmospheric science community for the atmospheric
science community. With help of Freva researchers can:

- quickly and intuitively search for data stored at typical data centers that
  host many datasets.
- create a common interface for user defined data analysis tools.
- apply data analysis tools in a reproducible manner.

### Authentication

The API supports token-based authentication using OAuth2. To obtain an access
token, clients can use the `/api/freva/auth/v2/token` endpoint by providing valid
username and password credentials. The access token should then be included in
the Authorization header for secured endpoints.

"""

import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator
from uuid import NAMESPACE_URL, uuid5

import appdirs
import filelock
from fastapi import FastAPI
from fastapi.openapi.docs import get_redoc_html
from fastapi.requests import Request
from fastapi.responses import FileResponse, HTMLResponse

from freva_rest import __version__

from .config import ServerConfig
from .logger import logger, reset_loggers

metadata_tags = [
    {
        "name": "Data search",
        "description": (
            "The following endpoints can be used to search for data."
            "Search queries can be refined by applying "
            "`key=value` based contraints."
        ),
    },
    {
        "name": "User data",
        "description": (
            "With help of the following endpoints you can add your own data "
            "to the data search system, aka databrwoser."
        ),
    },
    {
        "name": "Load data",
        "description": (
            "With help of the following endpoints you can "
            "conviniently load and access data via `zarr`."
        ),
    },
    {
        "name": "Analysis Tools",
        "description": "Define, submit and monitor any data anaylsys tools.",
    },
    {
        "name": "Authentication",
        "description": "These endpoints are for authentication.",
    },
]

server_config = ServerConfig()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start and end things before and after shutdown.

    Things before yield are executed on startup. Things after on teardown.
    """
    cache_dir = Path(appdirs.user_cache_dir("freva-rest"))
    cache_dir.mkdir(exist_ok=True, parents=True)
    lock = filelock.FileLock(cache_dir / "client_secret.txt.lock")
    client_secret_file = cache_dir / "client_secret.txt"
    secret = str(uuid5(NAMESPACE_URL, datetime.now().isoformat()))
    if not client_secret_file.exists():  # pragma: no cover
        with lock:
            client_secret_file.write_text(secret, encoding="utf-8")
    try:
        yield
    finally:
        try:  # pragma: no cover
            server_config.mongo_client.close()
        except Exception as error:
            logger.warning("Could not shutdown mongodb connection: %s", error)


reset_loggers()

app = FastAPI(
    debug=bool(int(os.environ.get("DEBUG", "0"))),
    title="Freva RestAPI",
    version=__version__,
    description=__doc__,
    openapi_url="/api/freva-nextgen/help/openapi.json",
    docs_url=None,
    openapi_tags=metadata_tags,
    lifespan=lifespan,
    contact={"name": "DKRZ, Clint", "email": "freva@dkrz.de"},
    license_info={
        "name": "BSD 2-Clause License",
        "url": "https://opensource.org/license/bsd-2-clause",
        "x-logo": {
            "url": "https://freva-clint.github.io/freva-nextgen/_static/logo.png"
        },
    },
)


@app.get("/api/freva-nextgen/help", include_in_schema=False)
async def custom_redoc_ui_html(request: Request) -> HTMLResponse:
    return get_redoc_html(
        openapi_url="/api/freva-nextgen/help/openapi.json",
        title="Freva RestAPI",
        redoc_favicon_url="/favicon.ico",
    )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> FileResponse:
    return FileResponse(Path(__file__).parent / "favicon.ico")
