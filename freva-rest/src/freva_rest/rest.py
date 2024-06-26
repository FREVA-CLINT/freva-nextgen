"""Welcome to the RestAPI for freva 🧉
--------------------------------------

Freva, the free evaluation system framework, is a data search and analysis
platform developed by the atmospheric science community for the atmospheric
science community. With help of Freva researchers can:

- quickly and intuitively search for data stored at typical data centers that
  host many datasets.
- create a common interface for user defined data analysis tools.
- apply data analysis tools in a reproducible manner.

Authentication
--------------
The API supports token-based authentication using OAuth2. To obtain an access
token, clients can use the `/api/freva/auth/v2/token` endpoint by providing valid
username and password credentials. The access token should then be included in
the Authorization header for secured endpoints.

"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from freva_rest import __version__

from .config import ServerConfig, defaults
from .logger import logger

metadata_tags = [
    {
        "name": "Data search",
        "description": "Search for data based on `key=value` search queries.",
    },
    {"name": "Load data", "description": "Load the data via `zarr` files."},
    {"name": "Authentication", "description": "Create access tokens."},
]

server_config = ServerConfig(
    Path(os.environ.get("API_CONFIG", defaults["API_CONFIG"])),
    debug=bool(os.environ.get("DEBUG", int(defaults["DEBUG"]))),
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start and end things before and after shutdown.

    Things before yield are executed on startup. Things after on teardown.
    """
    try:
        yield
    finally:
        try:
            server_config.mongo_client.close()
        except Exception as error:  # pragma: no cover
            logger.warning("Could not shutdown mongodb connection: %s", error)


app = FastAPI(
    debug=bool(int(os.environ.get("DEBUG", "0"))),
    title="Freva RestAPI",
    version=__version__,
    description=__doc__,
    openapi_url="/api/freva/docs/openapi.json",
    docs_url="/api/freva/docs",
    redoc_url=None,
    lifespan=lifespan,
    contact={"name": "DKRZ, Clint", "email": "freva@dkrz.de"},
    license_info={
        "name": "BSD 2-Clause License",
        "url": "https://opensource.org/license/bsd-2-clause",
    },
)
