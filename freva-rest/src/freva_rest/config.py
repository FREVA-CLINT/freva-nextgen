"""Module for accessing basic server configuration.

The minimal configuration is accessed via environment variables. Entries can
be overridden with a specific toml file holding configurations.
"""

import logging
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests
import tomli
from motor.motor_asyncio import AsyncIOMotorClient
from typing_extensions import TypedDict

from .logger import THIS_NAME, logger, logger_file_handle

CONFIG_TYPE = TypedDict(
    "CONFIG_TYPE",
    {
        "API_CONFIG": Path,
        "LOGGER": logging.Logger,
        "NAME": str,
        "DEBUG": bool,
        "API_CACHE_EXP": int,
        "API_URL": str,
        "REDIS_HOST": Optional[str],
        "REDIS_SSL_CERTFILE": Optional[str],
        "REDIS_SSL_KEYFILE": Optional[str],
        "REDIS_PASS": Optional[str],
        "REDIS_USER": Optional[str],
        "KEYCLOAK_HOST": str,
        "KEYCLOAK_REALM": str,
    },
)
defaults: CONFIG_TYPE = {
    "API_CONFIG": Path(__file__).parent / "api_config.toml",
    "LOGGER": logger,
    "NAME": THIS_NAME,
    "DEBUG": False,
    "API_CACHE_EXP": 3600,
    "REDIS_SSL_CERTFILE": None,
    "REDIS_SSL_KEYFILE": None,
    "REDIS_HOST": None,
    "REDIS_USER": None,
    "REDIS_PASS": None,
    "KEYCLOAK_HOST": "http://localhost:8080",
    "KEYCLOAK_REALM": "freva",
    "API_URL": "http://localhost:7777",
}


@dataclass
class ServerConfig:
    """Read the basic configuration for the server.

    The configuration can either be set via environment variables or a server
    config file.

    Parameters
    ----------
    config_file: pathlib.Path
        Path to the basic configuration file.
    debug: bool, default: False
        Set the logging level to DEBUG
    """

    config_file: Path = Path(os.environ.get("API_CONFIG") or defaults["API_CONFIG"])
    debug: bool = False

    def __post_init__(self) -> None:
        self.set_debug(self.debug)
        try:
            self._config = tomli.loads(self.config_file.read_text("utf-8"))
        except Exception as error:
            logger.critical("Failed to load %s", error)
            self._config = tomli.loads(defaults["API_CONFIG"].read_text("utf-8"))
        self.mongo_client = AsyncIOMotorClient(
            self.mongo_url, serverSelectionTimeoutMS=5000
        )
        self.mongo_collection = self.mongo_client[self.mongo_db]["search_queries"]
        self._solr_fields = self._get_solr_fields()
        self._keycloak_overview: Optional[Dict[str, Any]] = None

    def reload(self) -> None:
        """Reload the configuration."""
        self.config_file = Path(os.environ.get("API_CONFIG") or defaults["API_CONFIG"])
        self.debug = defaults["DEBUG"]
        self.__post_init__()

    @staticmethod
    def get_keycloak_url() -> str:
        """Construct the keycloak realm url."""
        keycloak_host = os.getenv("KEYCLOAK_HOST", defaults["KEYCLOAK_HOST"])
        keycloak_realm = os.getenv("KEYCLOAK_REALM", defaults["KEYCLOAK_REALM"])
        _, split, host = keycloak_host.partition("://")
        if split:
            host = keycloak_host
        else:
            host = f"https://{keycloak_host}"
        return f"{host}/realms/{keycloak_realm}"

    @property
    def keycloak_discovery_url(self) -> str:
        """Construct the discovery url."""
        return f"{self.get_keycloak_url()}/.well-known/openid-configuration"

    @property
    def keycloak_overview(self) -> Dict[str, Any]:
        """Query the url overview from keycloak."""
        if self._keycloak_overview is not None:
            return self._keycloak_overview
        res = requests.get(self.keycloak_discovery_url, verify=False, timeout=3)
        res.raise_for_status()
        self._keycloak_overview = res.json()
        return self._keycloak_overview

    @property
    def mongo_url(self) -> str:
        """Get the url to the mongodb."""
        host = self._config.get("mongo_db", {}).get("hostname", "") or os.environ.get(
            "MONGO_HOST", "localhost"
        )
        host, _, m_port = host.partition(":")
        port = m_port or str(self._config.get("mongo_db", {}).get("port", 27017))
        user = self._config.get("mongo_db", {}).get("user", "") or os.environ.get(
            "MONGO_USER", "mongo"
        )
        passwd = self._config.get("mongo_db", {}).get("password", "") or os.environ.get(
            "MONGO_PASSWORD", "secret"
        )

        return f"mongodb://{user}:{passwd}@{host}:{port}"

    @property
    def mongo_db(self) -> str:
        """Get the database name where the stats is stored."""
        return self._config.get("mongo_db", {}).get("name", "") or os.environ.get(
            "MONGO_DB", "search_stats"
        )

    @cached_property
    def solr_fields(self) -> List[str]:
        """Get all relevant solr facet fields."""
        return list(self._solr_fields)

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

    @property
    def solr_cores(self) -> Tuple[str, str]:
        """Get the names of the solr core."""
        core = self._config.get("solr", {}).get("core", "") or os.environ.get(
            "SOLR_CORE", "files"
        )
        return core, "latest"

    @property
    def solr_host(self) -> str:
        """Get the hostname of the running apache solr server."""
        return (
            self._config.get("solr", {}).get("hostname", "")
            or os.environ.get("SOLR_HOST", "localhost").partition(":")[0]
        )

    @property
    def solr_port(self) -> str:
        """Get the port of the running apache solr server."""
        return (
            str(self._config.get("solr", {}).get("port", ""))
            or os.environ.get("SOLR_HOST", "").partition(":")[-1]
        ) or "8983"

    def get_core_url(self, core: str) -> str:
        """Get the url for a specific solr core."""
        server = f"{self.solr_host}:{self.solr_port}"
        return f"http://{server}/solr/{core}"

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
