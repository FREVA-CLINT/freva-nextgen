"""Definition of the whole rest api."""

import asyncio

import uvloop

from freva_rest.auth import *  # noqa: F401
from freva_rest.databrowser_api.endpoints import *  # noqa: F401
from freva_rest.freva_data_portal.endpoints import *  # noqa: F401
from freva_rest.rest import app
from freva_rest.tool_api.endpoints import *  # noqa: F401

# Set uvloop as the default event loop policy


asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

__all__ = ["app"]
