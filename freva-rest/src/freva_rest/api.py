"""Definition of the whole rest api."""

from freva_rest.auth import *  # noqa: F401
from freva_rest.databrowser_api.endpoints import *  # noqa: F401
from freva_rest.freva_data_portal.endpoints import *  # noqa: F401
from freva_rest.rest import app
from freva_rest.tool_api.endpoints import *  # noqa: F401

__all__ = ["app"]
