"""Module for accessing basic server configuration.

The minimal configuration is accessed via environment variables. Entries can
be overridden with a specific toml file holding configurations or environment
variables.
"""

import logging
import os
from functools import cached_property
from pathlib import Path
from socket import gethostname
from typing import (
    Annotated,
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

import requests
import tomli
from fastapi import HTTPException
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pydantic import BaseModel, Field

from .logger import logger, logger_file_handle

ConfigItem = Union[str, int, float, None]


class ServerConfig(BaseModel):
    """Read the basic configuration for the server.

    The configuration can either be set via environment variables or a server
    config file.
    """

    config: Annotated[
        Union[str, Path],
        Field(
            title="API Config",
            description=("Path to a .toml file holding the API" "configuration"),
        ),
    ] = os.getenv("API_CONFIG", Path(__file__).parent / "api_config.toml")
    proxy: Annotated[
        str,
        Field(
            title="Proxy url",
            description="URL of a proxy that serves this API (if any).",
        ),
    ] = os.getenv("API_PROXY", "")
    debug: Annotated[
        Union[bool, int, str],
        Field(
            title="Debug mode",
            description="Turn on debug mode",
        ),
    ] = os.getenv("DEBUG", "0")
    mongo_host: Annotated[
        str,
        Field(
            title="MongoDB hostname",
            description="Set the <HOSTNAME>:<PORT> to the MongoDB service.",
        ),
    ] = os.getenv("API_MONGO_HOST", "")
    mongo_user: Annotated[
        str,
        Field(
            title="MongoDB user name.",
            description="The mongoDB user name to log on to the mongoDB.",
        ),
    ] = os.getenv("API_MONGO_USER", "")
    mongo_password: Annotated[
        str,
        Field(
            title="MongoDB password.",
            description="The MongoDb password to log on to the mongoDB.",
        ),
    ] = os.getenv("API_MONGO_PASSWORD", "")
    mongo_db: Annotated[
        str,
        Field(
            title="Mongo database",
            description="Name of the Mongo database that is used.",
        ),
    ] = os.getenv("API_MONGO_DB", "")
    solr_host: Annotated[
        str,
        Field(
            title="Solr hostname",
            description="Set the <HOSTNAME>:<PORT> to the Solr service.",
        ),
    ] = os.getenv("API_SOLR_HOST", "")
    solr_core: Annotated[
        str,
        Field(
            title="Solr core",
            description="Set the name of the core for the search index.",
        ),
    ] = os.getenv("API_SOLR_CORE", "")
    cache_exp: Annotated[
        str,
        Field(
            title="Cache expiration.",
            description=("The expiration time in sec" "of the data loading cache."),
        ),
    ] = os.getenv("API_CACHE_EXP", "")
    api_services: Annotated[
        str,
        Field(
            title="Services",
            description="The services that should be enabled.",
        ),
    ] = os.getenv("API_SERVICES", "databrowser,zarr-stream")
    redis_host: Annotated[
        str,
        Field(
            title="Rest Host",
            description="Url of the redis cache.",
        ),
    ] = os.getenv("API_REDIS_HOST", "")
    redis_ssl_certfile: Annotated[
        str,
        Field(
            title="Redis cert file.",
            description=(
                "Path to the public" "certfile to make" "connections to the" "cache"
            ),
        ),
    ] = os.getenv("API_REDIS_SSL_CERTFILE", "")
    redis_ssl_keyfile: Annotated[
        str,
        Field(
            title="Redis key file.",
            description=(
                "Path to the privat" "key file to make" "connections to the" "cache"
            ),
        ),
    ] = os.getenv("API_REDIS_SSL_KEYFILE", "")
    redis_password: Annotated[
        str,
        Field(
            title="Redis password",
            description=("Password for redis connections."),
        ),
    ] = os.getenv("API_REDIS_PASSWORD", "")
    redis_user: Annotated[
        str,
        Field(
            title="Redis username",
            description=("Username for redis connections."),
        ),
    ] = os.getenv("API_REDIS_USER", "")
    oidc_discovery_url: Annotated[
        str,
        Field(
            title="OIDC url",
            description="OpenID connect discovery url.",
        ),
    ] = os.getenv("API_OIDC_DISCOVERY_URL", "")
    oidc_client_id: Annotated[
        str,
        Field(
            title="OIDC client id",
            description="The OIDC client id used for authentication.",
        ),
    ] = os.getenv("API_OIDC_CLIENT_ID", "")
    oidc_client_secret: Annotated[
        str,
        Field(
            title="OIDC client secret",
            description="The OIDC client secret, if any, used for authentication.",
        ),
    ] = os.getenv("API_OIDC_CLIENT_SECRET", "")
    stacapi_host: Annotated[
        str,
        Field(
            title="STAC host",
            description=(
                "Set the <USERNAME>:<PASSWORD>@<HOSTNAME>:<PORT>/<ROO_PATH> "
                "to the STAC service."
            ),
        ),
    ] = os.getenv("API_STAC_HOST", "")
    stacbrowser_host: Annotated[
        str,
        Field(
            title="STAC browser host",
            description=(
                "Set the <HOSTNAME>:<PORT>/<PREFIX> tp the STAC "
                "browser service."
            ),
        ),
    ] = os.getenv("API_STACBROWSER_HOST", "")
    stacapi_max_items: Annotated[
        Union[int, str],
        Field(
            title="STAC max items",
            description="The maximum number of items to retrieve from the STAC API.",
        ),
    ] = os.getenv("API_STAC_MAX_ITEMS", 1000)

    def _read_config(self, section: str, key: str) -> Any:
        fallback = self._fallback_config[section][key] or None
        return self._config.get(section, {}).get(key, fallback)

    def model_post_init(self, __context: Any = None) -> None:
        self._fallback_config: Dict[str, Any] = tomli.loads(
            (Path(__file__).parent / "api_config.toml").read_text()
        )
        self._config: Dict[str, Any] = {}
        api_config = Path(self.config).expanduser().absolute()
        try:
            self._config = tomli.loads(api_config.read_text())
        except Exception as error:
            logger.critical("Failed to load config file: %s", error)
            self._config = self._fallback_config
        if isinstance(self.debug, str):
            self.debug = bool(int(self.debug))
        self.debug = bool(self.debug)
        self.set_debug(self.debug)
        self._mongo_client: Optional[AsyncIOMotorClient] = None
        self._solr_fields = self._get_solr_fields()
        self._oidc_overview: Optional[Dict[str, Any]] = None
        self.api_services = self.api_services or ",".join(
            self._read_config("restAPI", "services")
        )
        self.proxy = (
            self.proxy
            or self._read_config("restAPI", "proxy")
            or f"http://{gethostname()}"
        )
        self.oidc_discovery_url = self.oidc_discovery_url or self._read_config(
            "oidc", "discovery_url"
        )
        self.oidc_client_secret = self.oidc_client_secret or self._read_config(
            "oidc", "client_secret"
        )
        self.oidc_client_id = self.oidc_client_id or self._read_config(
            "oidc", "client_id"
        )
        self.mongo_host = self.mongo_host or self._read_config("mongo_db", "hostname")
        self.mongo_user = self.mongo_user or self._read_config("mongo_db", "user")
        self.mongo_password = self.mongo_password or self._read_config(
            "mongo_db", "password"
        )
        self.mongo_db = self.mongo_db or self._read_config("mongo_db", "name")
        self.solr_host = self.solr_host or self._read_config("solr", "hostname")
        self.solr_core = self.solr_core or self._read_config("solr", "core")
        self.redis_user = self.redis_user or self._read_config("cache", "user")
        self.redis_password = self.redis_password or self._read_config(
            "cache", "password"
        )
        self.cache_exp = self.cache_exp or self._read_config("cache", "exp")
        self.redis_ssl_keyfile = self.redis_ssl_keyfile or self._read_config(
            "cache", "key_file"
        )
        self.redis_ssl_certfile = self.redis_ssl_certfile or self._read_config(
            "cache", "cert_file"
        )
        self.redis_host = self.redis_host or self._read_config("cache", "hostname")
        self.stacapi_host = self.stacapi_host or \
            self._read_config("stacapi", "api_hostname")
        self.stacbrowser_host = self.stacbrowser_host or self._read_config(
            "stacapi", "browser_hostname"
        )
        self.stacapi_max_items = self.stacapi_max_items or self._read_config(
            "stacapi", "max_items"
        )

    @staticmethod
    def get_url(url: str, default_port: Union[str, int]) -> str:
        """Parse the url by constructing: <scheme>://<host>:<port>"""
        # Remove netloc, host from <scheme>://<host>:<port>
        port = url.split("://", 1)[-1].partition(":")[-1]
        if port:
            # The url has already a port
            return url
        return f"{url}:{default_port}"
        # If the original url has already a port in the suffix remove it

    @property
    def services(self) -> Set[str]:
        """Define the services that are served."""
        return set(s.strip() for s in self.api_services.split(",") if s.strip())

    @property
    def redis_url(self) -> str:
        """Construct the url to the redis service."""
        url = self.get_url(self.redis_host, self._read_config("cache", "port"))
        return url.split("://")[-1].partition(":")[0]

    @property
    def redis_port(self) -> int:
        """Get the port the redis host is listining on."""
        url = self.get_url(self.redis_host, self._read_config("cache", "port"))
        return int(url.split("://")[-1].partition(":")[-1])

    @property
    def mongo_client(self) -> AsyncIOMotorClient:
        """Create an async connection client to the mongodb."""
        if self._mongo_client is None:
            self._mongo_client = AsyncIOMotorClient(
                self.mongo_url, serverSelectionTimeoutMS=5000
            )
        return self._mongo_client

    @property
    def mongo_collection_search(self) -> AsyncIOMotorCollection:
        """Define the mongoDB collection for databrowser searches."""
        return cast(
            AsyncIOMotorCollection,
            self.mongo_client[self.mongo_db]["search_queries"],
        )

    @property
    def mongo_collection_userdata(self) -> AsyncIOMotorCollection:
        """Define the mongoDB collection for user data information."""
        return cast(
            AsyncIOMotorCollection,
            self.mongo_client[self.mongo_db]["user_data"],
        )

    def power_cycle_mongodb(self) -> None:
        """Reset an existing mongoDB connection."""
        if self._mongo_client is not None:
            self._mongo_client.close()
        self._mongo_client = None

    def reload(self) -> None:
        """Reload the configuration."""
        self.model_post_init()

    @property
    def oidc_overview(self) -> Dict[str, Any]:
        """Query the url overview from OIDC Service."""
        if self._oidc_overview is not None:
            return self._oidc_overview
        res = requests.get(self.oidc_discovery_url, verify=False, timeout=3)
        res.raise_for_status()
        self._oidc_overview = res.json()
        return self._oidc_overview

    @property
    def mongo_url(self) -> str:
        """Get the url to the mongodb."""
        url = self.get_url(
            self.mongo_host, self._read_config("mongo_db", "port")
        ).removeprefix("mongodb://")
        user_prefix = ""
        if self.mongo_user:
            user_prefix = f"{self.mongo_user}@"
            if self.mongo_password:
                user_prefix = f"{self.mongo_user}:{self.mongo_password}@"
        return f"mongodb://{user_prefix}{url}"

    @property
    def log_level(self) -> int:
        """Get the name of the current logger level."""
        return logger.getEffectiveLevel()

    @staticmethod
    def set_debug(debug: bool) -> None:
        """Set the logger levels to debug."""
        if debug:
            level = logging.DEBUG
        else:
            level = logging.INFO
        logger.setLevel(level)
        logger_file_handle.setLevel(level)

    @cached_property
    def solr_fields(self) -> List[str]:
        """Get all relevant solr facet fields."""
        return list(self._solr_fields)

    @property
    def solr_cores(self) -> Tuple[str, str]:
        """Get the names of the solr core."""
        return self.solr_core, "latest"

    def get_core_url(self, core: str) -> str:
        """Get the url for a specific solr core."""
        return f"{self.solr_url}/solr/{core}"

    @property
    def solr_url(self) -> str:
        """Construct the url to the solr server."""
        solr_port = str(self._read_config("solr", "port"))
        url = self.get_url(self.solr_host, solr_port)
        _, split, _ = url.partition("://")
        if not split:
            return f"http://{url}"
        return url

    def _get_solr_fields(self) -> Iterator[str]:
        url = f"{self.get_core_url(self.solr_cores[-1])}/schema/fields"
        try:
            for entry in requests.get(url, timeout=5).json().get("fields", []):
                if entry["type"] in ("extra_facet", "text_general") and entry[
                    "name"
                ] not in ("file_name", "file", "file_no_version"):
                    yield entry["name"]
        except requests.exceptions.ConnectionError as error:  # pragma: no cover
            logger.error("Connection to %s failed: %s", url, error)  # pragma: no cover
            yield ""  # pragma: no cover

    def get_stacapi_url(
        self,
        spec: Literal["collections", "items", "ping"],
        collection: Optional[str] = None,
    ) -> str:
        """Get the url of the STAC for transaction.

        Parameters:
        -----------
            spec: Literal["collections", "items", "ping"]: Spec type
            collection: Optional collection name, required for "items" spec

        Returns:
        --------
            Complete STAC URL as string
        """
        if not self.stacapi_host:
            raise HTTPException(
                status_code=503,
                detail=(
                    "STAC-API service is not set in the configuration "
                    "of the current Freva instance."
                ),
            )
        base_url = self.stacapi_host.rstrip('/')

        if spec == "collections":
            return f"{base_url}/{spec}"
        if spec == "items":
            logger.debug("Collection: %s", collection)
            if collection is None:
                raise ValueError(
                    "Collection name is required for 'items' spec"
                )  # pragma: no cover
            return f"{base_url}/collections/{collection}/bulk_items"
        if spec == "ping":
            return f"{base_url}/_mgmt/{spec}"
