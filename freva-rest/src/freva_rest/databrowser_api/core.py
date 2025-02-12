"""The core functionality to interact with the apache solr search system."""

import ast
import asyncio
import io
import json
import tarfile
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property, wraps
from textwrap import dedent
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Sized,
    Tuple,
    Union,
    cast,
)

import httpx
import pystac
from dateutil import parser
from dateutil.parser import ParserError, parse
from fastapi import HTTPException, Request
from pydantic import BaseModel
from pymongo import UpdateOne, errors
from typing_extensions import TypedDict

from freva_rest import __version__
from freva_rest.config import ServerConfig
from freva_rest.logger import logger
from freva_rest.utils import create_redis_connection

FlavourType = Literal["freva", "cmip6", "cmip5", "cordex", "nextgems", "user"]
IntakeType = TypedDict(
    "IntakeType",
    {
        "esmcat_version": str,
        "attributes": List[Dict[str, str]],
        "assets": Dict[str, str],
        "id": str,
        "description": str,
        "title": str,
        "last_updated": str,
        "aggregation_control": Dict[str, Any],
    },
)


def ensure_future(
    async_func: Callable[..., Awaitable[Any]]
) -> Callable[..., Coroutine[Any, Any, asyncio.Task[Any]]]:
    """Decorator that runs any given asyncio function in the background."""

    @wraps(async_func)
    async def wrapper(*args: Any, **kwargs: Any) -> asyncio.Task[Any]:
        """Async wrapper function that creates the call."""
        try:
            loop = (
                asyncio.get_running_loop()
            )  # Safely get the current running event loop
        except RuntimeError:
            # No running event loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Schedule the coroutine for execution in the background
        return asyncio.ensure_future(async_func(*args, **kwargs))

    return wrapper


class SearchResult(BaseModel):
    """Return Model of a uniq key search."""

    total_count: int
    facets: Dict[str, List[Union[str, int]]]
    search_results: List[Dict[str, Union[str, float, List[str]]]]
    facet_mapping: Dict[str, str]
    primary_facets: List[str]


class IntakeCatalogue(BaseModel):
    """Return Model of a uniq key search."""

    catalogue: IntakeType
    total_count: int


@dataclass
class Translator:
    """Class that defines the flavour translation.

    Parameters
    ----------
    flavour: str
        The target flavour, the facet names should be translated to.
    translate: bool, default: True
        Translate the search keys. Not translating (default: True) can be
        useful if the actual translation of the facets should be done on the
        client side.

    Attributes
    ----------
    """

    flavour: str
    translate: bool = True
    flavours: tuple[FlavourType, ...] = (
        "freva",
        "cmip6",
        "cmip5",
        "cordex",
        "nextgems",
        "user",
    )

    @property
    def facet_hierarchy(self) -> list[str]:
        """Define the hierarchy of facets that define a dataset."""
        return [
            "project",
            "product",
            "institute",
            "model",
            "experiment",
            "time_frequency",
            "realm",
            "variable",
            "ensemble",
            "cmor_table",
            "fs_type",
            "grid_label",
            "grid_id",
            "format",
        ]

    @property
    def _freva_facets(self) -> Dict[str, str]:
        """Define the freva search facets and their relevance"""
        return {
            "project": "primary",
            "product": "primary",
            "institute": "primary",
            "model": "primary",
            "experiment": "primary",
            "time_frequency": "primary",
            "realm": "primary",
            "variable": "primary",
            "ensemble": "primary",
            "time_aggregation": "primary",
            "fs_type": "secondary",
            "grid_label": "secondary",
            "cmor_table": "secondary",
            "driving_model": "secondary",
            "format": "secondary",
            "grid_id": "secondary",
            "level_type": "secondary",
            "rcm_name": "secondary",
            "rcm_version": "secondary",
            "dataset": "secondary",
            "time": "secondary",
            "bbox": "secondary",
            "user": "secondary",
        }

    @property
    def _cmip5_lookup(self) -> Dict[str, str]:
        """Define the search facets for the cmip5 standard."""
        return {
            "experiment": "experiment",
            "ensemble": "member_id",
            "fs_type": "fs_type",
            "grid_label": "grid_label",
            "institute": "institution_id",
            "model": "model_id",
            "project": "project",
            "product": "product",
            "realm": "realm",
            "variable": "variable",
            "time": "time",
            "bbox": "bbox",
            "time_aggregation": "time_aggregation",
            "time_frequency": "time_frequency",
            "cmor_table": "cmor_table",
            "dataset": "dataset",
            "driving_model": "driving_model",
            "format": "format",
            "grid_id": "grid_id",
            "level_type": "level_type",
            "rcm_name": "rcm_name",
            "rcm_version": "rcm_version",
        }

    @property
    def _cmip6_lookup(self) -> Dict[str, str]:
        """Define the search facets for the cmip6 standard."""
        return {
            "experiment": "experiment_id",
            "ensemble": "member_id",
            "fs_type": "fs_type",
            "grid_label": "grid_label",
            "institute": "institution_id",
            "model": "source_id",
            "project": "mip_era",
            "product": "activity_id",
            "realm": "realm",
            "variable": "variable_id",
            "time": "time",
            "bbox": "bbox",
            "time_aggregation": "time_aggregation",
            "time_frequency": "frequency",
            "cmor_table": "table_id",
            "dataset": "dataset",
            "driving_model": "driving_model",
            "format": "format",
            "grid_id": "grid_id",
            "level_type": "level_type",
            "rcm_name": "rcm_name",
            "rcm_version": "rcm_version",
        }

    @property
    def _cordex_lookup(self) -> Dict[str, str]:
        """Define the search facets for the cordex5 standard."""
        return {
            "experiment": "experiment",
            "ensemble": "ensemble",
            "fs_type": "fs_type",
            "grid_label": "grid_label",
            "institute": "institution",
            "model": "model",
            "project": "project",
            "product": "domain",
            "realm": "realm",
            "variable": "variable",
            "time": "time",
            "bbox": "bbox",
            "time_aggregation": "time_aggregation",
            "time_frequency": "time_frequency",
            "cmor_table": "cmor_table",
            "dataset": "dataset",
            "driving_model": "driving_model",
            "format": "format",
            "grid_id": "grid_id",
            "level_type": "level_type",
            "rcm_name": "rcm_name",
            "rcm_version": "rcm_version",
        }

    @property
    def _nextgems_lookup(self) -> Dict[str, str]:
        """Define the search facets for the cmip5 standard."""
        return {
            "experiment": "experiment",
            "ensemble": "member_id",
            "fs_type": "fs_type",
            "grid_label": "grid_label",
            "institute": "institution_id",
            "model": "source_id",
            "project": "project",
            "product": "experiment_id",
            "realm": "realm",
            "variable": "variable_id",
            "time": "time",
            "bbox": "bbox",
            "time_aggregation": "time_reduction",
            "time_frequency": "time_frequency",
            "cmor_table": "cmor_table",
            "dataset": "dataset",
            "driving_model": "driving_model",
            "format": "format",
            "grid_id": "grid_id",
            "level_type": "level_type",
            "rcm_name": "rcm_name",
            "rcm_version": "rcm_version",
        }

    @cached_property
    def forward_lookup(self) -> Dict[str, str]:
        """Define how things get translated from the freva standard"""

        return {
            "freva": {k: k for k in self._freva_facets},
            "cmip6": self._cmip6_lookup,
            "cmip5": self._cmip5_lookup,
            "cordex": self._cordex_lookup,
            "nextgems": self._nextgems_lookup,
            "user": {k: k for k in self._freva_facets},
        }[self.flavour]

    @cached_property
    def valid_facets(self) -> list[str]:
        """Get all valid facets for a flavour."""
        if self.translate:
            return list(self.forward_lookup.values())
        return list(self.forward_lookup.keys())

    @property
    def cordex_keys(self) -> Tuple[str, ...]:
        """Define the keys that make a cordex dataset."""
        return ("rcm_name", "driving_model", "rcm_version")

    @cached_property
    def primary_keys(self) -> list[str]:
        """Define which search facets are primary for which standard."""
        if self.translate:
            _keys = [
                self.forward_lookup[k]
                for (k, v) in self._freva_facets.items()
                if v == "primary"
            ]
        else:
            _keys = [k for (k, v) in self._freva_facets.items() if v == "primary"]
        if self.flavour in ("cordex",):
            for key in self.cordex_keys:
                _keys.append(key)
        return _keys

    @cached_property
    def backward_lookup(self) -> Dict[str, str]:
        """Translate the schema to the freva standard."""
        return {v: k for (k, v) in self.forward_lookup.items()}

    def translate_facets(
        self,
        facets: Iterable[str],
        backwards: bool = False,
    ) -> List[str]:
        """Translate the facets names to a given flavour."""
        if self.translate:
            if backwards:
                return [self.backward_lookup.get(f, f) for f in facets]
            return [self.forward_lookup.get(f, f) for f in facets]
        return list(facets)

    def translate_query(
        self,
        query: Dict[str, Any],
        backwards: bool = False,
    ) -> Dict[str, Any]:
        """Translate the queries names to a given flavour."""
        return dict(
            zip(
                self.translate_facets(query.keys(), backwards=backwards),
                query.values(),
            )
        )


class Solr:
    """Definitions for making search queries on apache solr and
    ingesting the user data into the apache solr.

    Parameters
    ----------
    config: ServerConfig
        An instance of the server configuration class.
    uniq_key: str, default: file
        The type of search result, which can be either "file" or "uri". This
        parameter determines whether the search will be based on file paths or
        Uniform Resource Identifiers (URIs).
    flavour: str, default: freva
        The Data Reference Syntax (DRS) standard specifying the type of climate
        datasets to query. The available DRS standards can be retrieved using the
        ``GET /overview`` method.
    start: int, default: 0
        Specify the starting point for receiving results.
    multi_version: bool, default: False
        Use versioned datasets in stead of latest versions.
    translate: bool, default: True
        Translate the output to the required DRS flavour.

    Attributes
    ----------
    """

    uniq_keys: Tuple[str, str] = ("file", "uri")
    """The names of all unique keys in the indexing system."""

    timeout: httpx.Timeout = httpx.Timeout(30)
    """30 seconds for timeout."""

    batch_size: int = 150
    """Maximum solr batch query size for one single query result."""
    suffixes = [".nc", ".nc4", ".grb", ".grib", ".tar", ".zarr"]
    escape_chars: Tuple[str, ...] = (
        "+",
        "-",
        "&&",
        "||",
        "!",
        "(",
        ")",
        "{",
        "}",
        "[",
        "]",
        "^",
        "~",
        ":",
        "/",
    )
    """Lucene (solr) special characters that need escaping."""

    def __init__(
        self,
        config: ServerConfig,
        *,
        uniq_key: Literal["file", "uri"] = "file",
        flavour: FlavourType = "freva",
        start: int = 0,
        multi_version: bool = True,
        translate: bool = True,
        _translator: Union[None, Translator] = None,
        **query: list[str],
    ) -> None:
        self._config = config
        self.uniq_key = uniq_key
        self.multi_version = multi_version
        self.translator = _translator or Translator(flavour, translate)
        try:
            self.time = self.adjust_time_string(
                query.pop("time", [""])[0],
                query.pop("time_select", ["flexible"])[0],
            )
            self.bbox = self.adjust_bbox_string(
                query.pop("bbox", [""])[0],
                query.pop("bbox_select", ["flexible"])[0],
            )
        except ValueError as err:
            raise HTTPException(status_code=500, detail=str(err)) from err
        self.stac_dynamic = bool(query.pop("stac_dynamic", [True])[0])
        self.facets = self.translator.translate_query(query, backwards=True)
        self.url, self.query = self._get_url()
        self.query["start"] = start
        self.query["sort"] = "file desc"
        self.fwrites: Dict[str, str] = {}
        self.total_ingested_files = 0
        self.total_duplicated_files = 0
        self.current_batch: List[Dict[str, str]] = []
        self.suffixes = [".nc", ".nc4", ".grb", ".grib", ".zarr", "zar"]
        # TODO: If one adds a dataset from cloud storage, the file system type
        # should be changed to cloud storage type. We need to find an approach
        # to determine the file system
        self.fs_type: str = "posix"

    async def _is_query_duplicate(self, uri: str, file_path: str) -> bool:
        """
        Check if a document with the given URI or file path already exists in Solr.

        Parameters
        ----------
        uri : str
            The URI to check
        file_path : str
            The file path to check

        Returns
        -------
        bool
            True if document exists, False otherwise
        """
        original_url = getattr(self, "url", None)
        original_query = getattr(self, "query", None)
        try:
            core_url = self._config.get_core_url(self._config.solr_cores[-1])
            self.url = f"{core_url}/select"

            def escape_special_chars(value: str) -> str:
                for char in self.escape_chars:
                    if char in value:
                        value = value.replace(char, f"\\{char}")
                value = value.replace('"', '\\"')
                return value

            escaped_uri = escape_special_chars(str(uri))
            escaped_file = escape_special_chars(str(file_path))

            query_parts = [f'uri:"{escaped_uri}"', f'file:"{escaped_file}"']
            query_str = " OR ".join(query_parts)

            self.query = {
                "q": query_str,
                "fl": "id",
                "rows": "1",
                "wt": "json",
            }
            async with self._session_get() as res:
                response_status, response_data = res
                if response_status == 200:
                    if response_data["response"]["numFound"] > 0:
                        return True
                return False
        finally:
            self.url = original_url if original_url is not None else ""
            self.query = original_query if original_query is not None else {}

    @asynccontextmanager
    async def _session_get(self) -> AsyncIterator[Tuple[int, Dict[str, Any]]]:
        """Wrap the get request round a try and catch statement."""
        logger.info(
            "Query %s for uniq_key: %s with %s",
            self.url,
            self.uniq_key,
            self.query,
        )
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(self.url, params=self.query)
                status = response.status_code
                try:
                    await self.check_for_status(response)
                    search = response.json()
                except HTTPException:  # pragma: no cover
                    search = {}  # pragma: no cover
            except Exception as error:
                logger.exception("Connection to %s failed: %s", self.url, error)
                raise HTTPException(
                    status_code=503,
                    detail="Could not connect to Solr server",
                )
            yield status, search

    @asynccontextmanager
    async def _session_post(
        self,
        url: str,
        payload: Union[
            Dict[str, Any],
            Dict[str, pystac.Collection],
            List[Dict[Any, Any]],
        ],
    ) -> AsyncIterator[Tuple[int, Dict[str, Any]]]:
        """Wrap the post request round a try and catch statement."""
        logger.info(
            "Sending POST request to %s with payload: %s",
            url,
            payload,
        )
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, json=payload)
                try:
                    await self.check_for_status(response)
                    logger.info(
                        "POST request successful with status: %d",
                        response.status_code,
                    )
                    response_data = response.json()
                except HTTPException:  # pragma: no cover
                    logger.error("POST request failed: %s", response.text)
                    response_data = {}
            except Exception as error:
                logger.exception("Connection to %s failed: %s", url, error)
                raise HTTPException(
                    status_code=503,
                    detail="Could not connect to the instance",
                )
            yield response.status_code, response_data

    @asynccontextmanager
    async def _session_put(
        self,
        url: str,
        payload: Dict[str, Any],
    ) -> AsyncIterator[Tuple[int, Dict[str, Any]]]:
        """Wrap the put request round a try and catch statement."""
        logger.info(
            "Sending PUT request to %s: with payload: %s",
            url,
            payload,
        )
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.put(url, json=payload)
                try:
                    await self.check_for_status(response)
                    logger.info(
                        "PUT request successful with status: %d",
                        response.status_code,
                    )
                    response_data = response.json()
                except HTTPException:  # pragma: no cover
                    logger.error("PUT request failed: %s", response.text)
                    response_data = {}
            except ConnectionError:
                logger.exception("Connection to %s failed", url)
                raise HTTPException(
                    status_code=503,
                    detail="Could not connect to the instance",
                )
            yield response.status_code, response_data

    @classmethod
    async def validate_parameters(
        cls,
        config: ServerConfig,
        *,
        uniq_key: Literal["file", "uri"] = "file",
        flavour: FlavourType = "freva",
        start: int = 0,
        multi_version: bool = False,
        translate: bool = True,
        **query: list[str],
    ) -> "Solr":
        """Create an instance of an Solr class with parameter validation.

        Parameters
        ----------
        config: ServerConfig
            An instance of the server configuration class.
        uniq_key: str, default: file
            The type of search result, which can be either "file" or "uri". This
            parameter determines whether the search will be based on file paths or
            Uniform Resource Identifiers (URIs).
        flavour: str, default: freva
            The Data Reference Syntax (DRS) standard specifying the type of climate
            datasets to query. The available DRS standards can be retrieved using the
            ``GET /overview`` method.
        start: int, default: 0
            Specify the starting point for receiving results.
        multi_version: bool, default: False
            Use versioned datasets in stead of latest versions.
        translate: bool, default: True
            Translate the output to the required DRS flavour.
        """
        translator = Translator(flavour, translate)
        for key in query:
            key = key.lower().replace("_not_", "")
            if (
                key not in translator.valid_facets
                and key not in ("time_select", "bbox_select", "stac_dynamic")
                + cls.uniq_keys
            ):
                raise HTTPException(status_code=422, detail="Could not validate input.")
        return Solr(
            config,
            flavour=flavour,
            translate=translate,
            uniq_key=uniq_key,
            start=start,
            multi_version=multi_version,
            _translator=translator,
            **query,
        )

    @staticmethod
    def adjust_time_string(
        time: str,
        time_select: str = "flexible",
    ) -> List[str]:
        """Adjust the time select keys to a solr time query

        Parameters
        ----------

        time: str, default: ""
            Special search facet to refine/subset search results by time.
            This can be a string representation of a time range or a single
            time step. The time steps have to follow ISO-8601. Valid strings are
            ``%Y-%m-%dT%H:%M`` to ``%Y-%m-%dT%H:%M`` for time ranges and
            ``%Y-%m-%dT%H:%M``. **Note**: You don't have to give the full string
            format to subset time steps ``%Y``, ``%Y-%m`` etc are also valid.
        time_select: str, default: flexible
            Operator that specifies how the time period is selected. Choose from
            flexible (default), strict or file. ``strict`` returns only those files
            that have the *entire* time period covered. The time search ``2000 to
            2012`` will not select files containing data from 2010 to 2020 with
            the ``strict`` method. ``flexible`` will select those files as
            ``flexible`` returns those files that have either start or end period
            covered. ``file`` will only return files where the entire time
            period is contained within *one single* file.

        Raises
        ------
        ValueError: If parsing the dates failed.
        """
        if not time:
            return []
        time = "".join(time.split())
        select_methods: dict[str, str] = {
            "strict": "Within",
            "flexible": "Intersects",
            "file": "Contains",
        }
        try:
            solr_select = select_methods[time_select]
        except KeyError as exc:
            methods = ", ".join(select_methods.keys())
            raise ValueError(f"Choose `time_select` from {methods}") from exc
        start, _, end = time.lower().partition("to")
        try:
            start = parse(start or "1", default=datetime(1, 1, 1, 0, 0, 0)).isoformat()
            end = parse(
                end or "9999", default=datetime(9999, 12, 31, 23, 59, 59)
            ).isoformat()
        except ParserError as exc:
            raise ValueError(exc) from exc
        return [f"{{!field f=time op={solr_select}}}[{start} TO {end}]"]

    @staticmethod
    def adjust_bbox_string(
        bbox: str,
        bbox_select: str = "flexible",
    ) -> List[str]:
        """Adjust the bbox select keys to a solr spatial query using RPT
        (Recursive Prefix Tree)

        Parameters
        ----------

        bbox: str, default: ""
            Special search facet to refine/subset search results by spatial extent.
            This can be a string representation of a bounding box or a WKT polygon.
            Valid strings are ``min_lon,max_lon by min_lat,max_lat`` for bounding
            boxes and Well-Known Text (WKT) format for polygons. **Note**: Longitude
            values must be between -180 and 180, latitude values between -90 and 90.
        bbox_select: str, default: flexible
            Operator that specifies how the spatial extent is selected. Choose from
            flexible (default), strict or file. ``strict`` returns only those files
            that fully contain the query extent. The bbox search ``-10,10 by -10,10``
            will not select files covering only ``0,5 by 0,5`` with the ``strict``
            method. ``flexible`` will select those files as it returns files that
            have any overlap with the query extent. ``file`` will only return files
            where the entire spatial extent is contained by the query geometry.

        Raises
        ------
        ValueError: If parsing the coordinates failed or if bbox_select is invalid.
        """
        if not bbox:
            return []
        bbox = "".join(part.strip() for part in bbox.split())
        select_methods: dict[str, str] = {
            "strict": "Within",
            "flexible": "Intersects",
            "file": "Contains",
        }
        try:
            solr_select = select_methods[bbox_select.lower()]
        except KeyError as exc:
            methods = ", ".join(select_methods.keys())
            raise ValueError(f"Choose `bbox_select` from {methods}") from exc

        try:
            lon_part, lat_part = bbox.lower().split("by")
            min_lon, max_lon = map(float, lon_part.split(","))
            min_lat, max_lat = map(float, lat_part.split(","))

            if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
                raise ValueError("Longitude must be between -180 and 180")
            if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
                raise ValueError("Latitude must be between -90 and 90")

            bbox_str = f"ENVELOPE({min_lon},{max_lon},{max_lat},{min_lat})"
            return [f'bbox:"{solr_select}({bbox_str})"']
        except ValueError as exc:
            raise ValueError(f"Failed to parse bbox string: {exc}") from exc

    async def _create_intake_catalogue(self, *facets: str) -> IntakeType:
        var_name = self.translator.forward_lookup["variable"]
        catalogue: IntakeType = {
            "esmcat_version": "0.1.0",
            "attributes": [
                {
                    "column_name": v,
                    "vocabulary": "",
                }
                for v in facets
            ],
            "assets": {
                "column_name": self.uniq_key,
                "format_column_name": "format",
            },
            "id": "freva",
            "description": f"Catalogue from freva-databrowser v{__version__}",
            "title": "freva-databrowser catalogue",
            "last_updated": f"{datetime.now().isoformat()}",
            "aggregation_control": {
                "variable_column_name": var_name,
                "groupby_attrs": [],
                "aggregations": [
                    {
                        "type": "union",
                        "attribute_name": f,
                        "options": {},
                    }
                    for f in facets
                ],
            },
        }
        return catalogue

    def _set_catalogue_queries(self) -> None:
        """Set the query parameters for an catalogue search."""
        self.query["facet"] = "true"
        self.query["facet.mincount"] = "1"
        self.query["facet.limit"] = "-1"
        self.query["rows"] = self.batch_size
        self.query["facet.field"] = self._config.solr_fields
        self.query["fl"] = [self.uniq_key] + self._config.solr_fields
        self.query["wt"] = "json"

    async def init_intake_catalogue(self) -> Tuple[int, IntakeCatalogue]:
        """Create an intake catalogue from the solr search."""
        self._set_catalogue_queries()
        async with self._session_get() as res:
            search_status, search = res
        total_count = cast(int, search.get("response", {}).get("numFound", 0))
        facets = search.get("facet_counts", {}).get("facet_fields", {})
        facets = [
            self.translator.forward_lookup.get(v, v)
            for v in self.translator.facet_hierarchy
            if facets.get(v)
        ]
        catalogue = await self._create_intake_catalogue(*facets)
        return search_status, IntakeCatalogue(
            catalogue=catalogue, total_count=total_count
        )

    async def _delete_from_mongo(self, search_keys: Dict[str, Union[str, int]]) -> None:
        """
        Delete bulk user metadata from MongoDB based on the search keys.

        Parameters
        ----------
        search_keys: Dict[str, str]
            A dictionariy containing search keys used to identify
            data for deletion.
        """
        try:
            query = {
                key: value if key.lower() == "file" else str(value).lower()
                for key, value in search_keys.items()
            }
            await self._config.mongo_collection_userdata.delete_many(query)
            logger.info("[MONGO] Deleted metadata with query: %s", query)
        except Exception as error:
            logger.warning("[MONGO] Could not remove metadata: %s", error)

    async def _insert_to_mongo(self, metadata_batch: List[Dict[str, Any]]) -> None:
        """
        Bulk upsert user metadata into MongoDB.

        Parameters
        ----------
        metadata_batch: List[Dict[str, Any]]
            A list of dictionaries containing metadata to insert into MongoDB.
        """
        bulk_operations = []

        for metadata in metadata_batch:
            filter_query = {"file": metadata["file"], "uri": metadata["uri"]}
            update_query = {"$set": metadata}

            bulk_operations.append(UpdateOne(filter_query, update_query, upsert=True))
        if bulk_operations:
            try:
                result = await self._config.mongo_collection_userdata.bulk_write(
                    bulk_operations,
                    ordered=False,
                    bypass_document_validation=False,
                )
                successful_upsert = (
                    result.upserted_count
                    + result.modified_count
                    + result.inserted_count
                )
                logger.info(
                    "[MONGO] Successfully inserted or updated "
                    "%s records into MongoDB."
                    "%s records aleady up-to-date.",
                    successful_upsert,
                    result.matched_count,
                )

            except errors.BulkWriteError as bwe:
                for err in bwe.details["writeErrors"]:
                    error_index = err["index"]
                    logger.error(
                        "[MONGO] Error in document at index %s: %s",
                        error_index,
                        err["errmsg"],
                    )
                    logger.error(
                        "[MONGO] Problematic document: %s",
                        metadata_batch[error_index],
                    )
                    successful_writes = (
                        bwe.details.get("nInserted", 0)
                        + bwe.details.get("nUpserted", 0)
                        + bwe.details.get("nModified", 0)
                    )
                    nMatched = bwe.details.get("nMatched", 0)
                logger.info(
                    "[MONGO] Partially succeeded: %s "
                    "documents inserted/updated successfully."
                    "%s documents already up-to-date.",
                    successful_writes,
                    nMatched,
                )
            except Exception as error:
                logger.exception("[MONGO] Could not insert metadata: %s", error)

    @ensure_future
    async def store_results(self, num_results: int, status: int) -> None:
        """Store the query into a database.

        Parameters
        ----------
        num_results: int
            The number of files that has been found.
        status: int
            The HTTP request status
        """
        if num_results == 0:
            return
        data = {
            "num_results": num_results,
            "flavour": self.translator.flavour,
            "uniq_key": self.uniq_key,
            "server_status": status,
            "date": datetime.now(),
        }
        facets = {k: "&".join(v) for (k, v) in self.facets.items()}
        try:
            await self._config.mongo_collection_search.insert_one(
                {"metadata": data, "query": facets}
            )
        except Exception as error:
            logger.warning("Could not add stats to mongodb: %s", error)

    def _process_catalogue_result(self, out: Dict[str, List[Sized]]) -> Dict[str, Any]:
        return {
            k: (
                out[k][0]
                if isinstance(out.get(k), list) and len(out[k]) == 1
                else out.get(k)
            )
            for k in [self.uniq_key] + self.translator.facet_hierarchy
            if out.get(k)
        }

    async def _iterintake(self) -> AsyncIterator[str]:
        encoder = json.JSONEncoder(indent=3)
        self.query["cursorMark"] = "*"
        init = True
        yield ',\n   "catalog_dict": '
        while True:
            async with self._session_get() as res:
                _, results = res

            for result in results.get("response", {}).get("docs", [{}]):
                entry = self._process_catalogue_result(result)
                if init is True:
                    sep = "["
                else:
                    sep = ","
                yield f"{sep}\n   "
                init = False
                for line in list(encoder.iterencode(entry)):
                    yield line
            next_cursor_mark = results.get("nextCursorMark", None)
            if next_cursor_mark == self.query["cursorMark"] or not results:
                break
            self.query["cursorMark"] = next_cursor_mark

    async def intake_catalogue(
        self, catalogue: IntakeType, header_only: bool = False
    ) -> AsyncIterator[str]:
        """Create an intake catalogue from the solr search."""
        encoder = json.JSONEncoder(indent=3)
        for line in list(encoder.iterencode(catalogue))[:-1]:
            yield line
        if header_only is False:
            async for line in self._iterintake():
                yield line
            yield "\n   ]\n}"

    async def extended_search(
        self,
        facets: List[str],
        max_results: int,
    ) -> Tuple[int, SearchResult]:
        """Initialise the apache solr metadata search.

        Returns
        -------
        int: status code of the apache solr query.
        """
        search_facets = [f for f in facets if f not in ("*", "all")]
        self.query["facet"] = "true"
        self.query["rows"] = max_results
        self.query["facet.sort"] = "index"
        self.query["facet.mincount"] = "1"
        self.query["facet.limit"] = "-1"
        self.query["wt"] = "json"
        self.query["facet.field"] = self.translator.translate_facets(
            search_facets or self._config.solr_fields, backwards=True
        )
        self.query["fl"] = [self.uniq_key, "fs_type"]
        logger.info(
            "Query %s for uniq_key: %s with %s",
            self.url,
            self.uniq_key,
            self.query,
        )

        async with self._session_get() as res:
            search_status, search = res
        return search_status, SearchResult(
            total_count=search.get("response", {}).get("numFound", 0),
            facets=self.translator.translate_query(
                search.get("facet_counts", {}).get("facet_fields", {})
            ),
            search_results=[
                {
                    **{self.uniq_key: k[self.uniq_key]},
                    **{"fs_type": k.get("fs_type", "posix")},
                }
                for k in search.get("response", {}).get("docs", [])
            ],
            facet_mapping={
                k: self.translator.forward_lookup[k]
                for k in self.query["facet.field"]
                if k in self.translator.forward_lookup
            },
            primary_facets=self.translator.primary_keys,
        )

    async def init_stream(self) -> Tuple[int, int]:
        """Initialise the apache solr search.

        Returns
        -------
        int: status code of the apache solr query.
        """
        self.query["fl"] = ["file", "uri"]
        logger.info(
            "Query %s for uniq_key: %s with %s",
            self.url,
            self.uniq_key,
            self.query,
        )
        async with self._session_get() as res:
            search_status, search = res
        return search_status, search.get("response", {}).get("numFound", 0)

    def _join_facet_queries(self, key: str, facets: List[str]) -> Tuple[str, str]:
        """Create lucene search contain and NOT contain search queries"""

        negative, positive = [], []
        for search_value in facets:
            if key not in self.uniq_keys:
                search_value = search_value.lower()
            if search_value.lower().startswith("not "):
                "len('not ') = 4"
                negative.append(search_value[4:])
            elif search_value[0] in ("!", "-"):
                negative.append(search_value[1:])
            elif "_not_" in key:
                negative.append(search_value)
            else:
                positive.append(search_value)
        search_value_pos = " OR ".join(positive)
        search_value_neg = " OR ".join(negative)
        for char in self.escape_chars:
            search_value_pos = search_value_pos.replace(char, "\\" + char)
            search_value_neg = search_value_neg.replace(char, "\\" + char)
        return search_value_pos, search_value_neg

    def _get_url(self) -> tuple[str, Dict[str, Any]]:
        """Get the url for the solr query."""
        core = {
            True: self._config.solr_cores[0],
            False: self._config.solr_cores[-1],
        }[self.multi_version]
        url = f"{self._config.get_core_url(core)}/select/"
        query = []
        for key, value in self.facets.items():
            query_pos, query_neg = self._join_facet_queries(key, value)
            key = key.lower().replace("_not_", "")
            if query_pos:
                query.append(f"{key}:({query_pos})")
            if query_neg:
                query.append(f"-{key}:({query_neg})")
        # Cause we are adding a new query which effects
        # all queries, it might be a neckbottle of performance
        user_query = (
            "user:*" if self.translator.flavour == "user" else "{!ex=userTag}-user:*"
        )
        base_query = "*:*" if not (self.time or self.bbox) else None
        joined_query = " AND ".join(query) if query else base_query

        filter_queries = [
            *self.time,
            *self.bbox,
            user_query,
            joined_query,
        ]
        return url, {
            "q": "*:*",
            "fq": [q for q in filter_queries if q],
        }

    @property
    def _post_url(self) -> str:
        """Construct the URL and payload for a solr POST request."""
        # All user-data stores in the latest core
        core = {
            True: self._config.solr_cores[-1],
            False: self._config.solr_cores[0],
        }[self.multi_version]

        url = (
            f"{self._config.get_core_url(core)}/update/json"
            "?commit=true&overwrite=false"
        )
        return url

    async def check_for_status(self, response: httpx.Response) -> None:
        """Check if a query was successful

        Parameters
        ----------
        response: httpx.Response
            The response of the rest query.

        Raises
        ------
        fastapi.HTTPException: If anything went wrong an error is risen
        """
        if response.status_code not in (200, 201):
            raise HTTPException(
                status_code=response.status_code, detail=response.text
            )  # pragma: no cover

    async def _solr_page_response(self) -> AsyncIterator[Dict[str, Any]]:
        self.query["cursorMark"] = "*"
        self.query["rows"] = self.batch_size
        while True:
            async with self._session_get() as res:
                _, results = res
            for content in results.get("response", {}).get("docs", []):
                yield content
            next_cursor_mark = results.get("nextCursorMark", None)
            if next_cursor_mark == self.query["cursorMark"] or not results:
                break
            self.query["cursorMark"] = next_cursor_mark

    async def stream_response(self) -> AsyncIterator[str]:
        """Search for uniq keys matching given search facets.

        Returns
        -------
        AsyncIterator: Stream of search results.
        """
        async for result in self._solr_page_response():
            yield f"{result[self.uniq_key]}\n"

    async def zarr_response(
        self,
        catalogue_type: Literal["intake", None],
        num_results: int,
    ) -> AsyncIterator[str]:
        """Create a zarr endpoint from a given search.
        Parameters
        ----------
        search: SearchResult
            The search result object of the search query.

        Returns
        -------
        AsyncIterator: Stream of search results.
        """
        api_path = f"{self._config.proxy}/api/freva-nextgen/data-portal/zarr"
        if catalogue_type == "intake":
            _, intake = await self.init_intake_catalogue()
            async for string in self.intake_catalogue(intake.catalogue, True):
                yield string
            yield ',\n   "catalog_dict": ['
        num = 1
        async for result in self._solr_page_response():
            prefix = suffix = ""
            uri = result[self.uniq_key]
            uuid5 = str(uuid.uuid5(uuid.NAMESPACE_URL, uri))
            try:
                cache = await create_redis_connection()
                await cache.publish(
                    "data-portal",
                    json.dumps({"uri": {"path": uri, "uuid": uuid5}}).encode("utf-8"),
                )
            except Exception as error:
                logger.error("Cloud not connect to redis: %s", error)
                yield "Internal error, service not available\n"
                continue
            output = f"{api_path}/{uuid5}.zarr"
            if catalogue_type == "intake":
                result[self.uniq_key] = output
                if num < num_results:
                    suffix = ","
                else:
                    suffix = ""
                output = json.dumps(self._process_catalogue_result(result), indent=3)
                prefix = "   "
            num += 1
            yield f"{prefix}{output}{suffix}\n"

        if catalogue_type == "intake":
            yield "\n   ]\n}"

    async def _add_to_solr(
        self,
        metadata_batch: List[Dict[str, Union[str, List[str], Dict[str, str]]]],
    ) -> None:
        """
        Add a batch of metadata to the Apache Solr cataloguing system.

        Parameters
        ----------
        metadata_batch : List[Tuple[str, Dict[str, Union[str, List[str]]]]]
            A list of tuples, each containing a file identifier and its
            associated metadata.

        Returns
        -------
        None
        """

        for metadata in metadata_batch:
            async with self._session_post(self._post_url, [metadata]) as (status, _):
                if status == 200:
                    self.total_ingested_files += 1

    async def _delete_from_solr(self, search_keys: Dict[str, Union[str, int]]) -> None:
        """
        Delete user data from Apache Solr based on search keys.

        Parameters
        ----------
        search_keys : Dict[str, str]
            A dictionary of search keys used to identify data to be deleted.
            Keys are field names and values are search values.

        Returns
        -------
        None
        """

        def escape_special_chars(value: str) -> str:
            for char in self.escape_chars:
                if char in value:
                    value = value.replace(char, f"\\{char}")
            return value

        query_parts = []
        for key, value in search_keys.items():
            key_lower = key.lower()
            if key_lower == "file":
                escaped_value = escape_special_chars(str(value))
                query_parts.append(f"{key_lower}:{escaped_value}")
            else:
                escaped_value = escape_special_chars(str(value).lower())
                query_parts.append(f"{key_lower}:{escaped_value}")
        query_str = " AND ".join(query_parts)
        async with self._session_post(self._post_url, {"delete": {"query": query_str}}):
            pass

    async def _ingest_user_metadata(self, user_metadata: List[Dict[str, Any]]) -> None:
        """
        Ingest validated user metadata.

        Parameters
        ----------
        user_metadata: List[Dict[str, Any]]
            A list of validated user metadata dictionaries.

        Returns
        -------
        None
        """
        processed_metadata = [
            {**metadata, **self.fwrites} for metadata in user_metadata
        ]
        for i in range(0, len(processed_metadata), self.batch_size):
            batch = processed_metadata[i: i + self.batch_size]
            processed_batch = await self._process_metadata(batch)
            self.total_duplicated_files += len(batch) - len(processed_batch)
            if processed_batch:
                await self._add_to_solr(processed_batch)
                await self._insert_to_mongo(processed_batch)

    async def _process_metadata(
        self,
        metadata_batch: List[Dict[str, Union[str, List[str], Dict[str, str]]]],
    ) -> List[Dict[str, Union[str, List[str], Dict[str, str]]]]:
        """Process the metadata batch by removing duplicates before ingestion."""
        new_querie = []
        for metadata in metadata_batch:
            uri = metadata.get("uri", "")
            file_path = metadata.get("file", "")

            if not uri and not file_path:
                continue
            is_duplicate = await self._is_query_duplicate(str(uri), str(file_path))
            if not is_duplicate:
                new_querie.append(metadata)
        return [dict(t) for t in {tuple(sorted(d.items())) for d in new_querie}]

    async def _purge_user_data(self, search_keys: Dict[str, Union[str, int]]) -> None:
        """
        Purge the user data from both the Apache Solr search system and MongoDB.

        Parameters
        ----------
        search_keys: Dict[str, str]
            A list of dictionaries containing search keys used to identify the
            data to be purged.

        Returns
        -------
        None
        """
        await self._delete_from_solr(search_keys)
        await self._delete_from_mongo(search_keys)

    async def _validate_user_metadata(
        self,
        user_metadata: Sequence[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        Validate the user metadata by checking for required fields.

        parameters:
        ----------
        user_metadata: Sequence[Dict[str, str]]
            A sequence of user metadata entries, which can be
            dictionaries containing metadata.

        Returns:
        --------
            List[Dict[str, str]]: A list of validated metadata
            dictionaries ready for ingestion.

        Raises:
        -------
            HTTPException: If no valid metadata is found
            or if required fields are missing.
        """
        required_fields = {"file", "variable", "time", "time_frequency"}
        validated_user_metadata: List[Dict[str, str]] = []
        for metadata in user_metadata:
            if not required_fields.issubset(metadata):
                logger.warning(
                    f"Invalid metadata: missing one or"
                    f"more required fields {required_fields}"
                )
                continue
            metadata["uri"] = metadata["file"]
            validated_user_metadata.append(metadata)
        if not validated_user_metadata:
            raise HTTPException(
                status_code=422, detail="No valid metadata found in the input."
            )
        return validated_user_metadata

    async def add_user_metadata(
        self,
        user_name: str,
        user_metadata: List[Dict[str, Any]],
        **fwrites: Dict[str, str],
    ) -> str:
        """
        Add validated user metadata to the Apache Solr search system
        and MongoDB.

        parameters:
        ----------
        user_name: str
            The username associated with the metadata.
        user_metadata: List[Dict[str, Any]]
            The metadata to be ingested.
        fwrites: Dict[str, str]
            Optional additional metadata to be added to the user metadata.
        """
        self.fwrites |= {
            "user": user_name,
            "fs_type": self.fs_type,
        } | fwrites.get("facets", {})
        await self._ingest_user_metadata(user_metadata)

        logger.info(
            "Ingested %d files into Solr and MongoDB",
            self.total_ingested_files,
        )
        if self.total_ingested_files == 0:
            status_msg = (
                f"No data was added to the databrowser. "
                f"{self.total_duplicated_files} files were "
                f"duplicates and not added."
            )
        else:
            status_msg = (
                f"{self.total_ingested_files} have been successfully "
                f"added to the databrowser. {self.total_duplicated_files} "
                f"files were duplicates and not added."
            )
        return status_msg

    async def delete_user_metadata(
        self, user_name: str, search_keys: Dict[str, Union[str, int]]
    ) -> None:
        """
        Delete user data from the Apache Solr search system and MongoDB.

        This method deletes the data associated with a user from both the Solr
        search system and MongoDB, using specific search keys to find and then
        purge the data.

        Parameters:
        ----------
        user: str
            The identifier of the user whose data is being deleted.
        search_keys: Dict[str, Union[str, int]]:
            A dictionary of keys used to identify the data to be deleted.
        """
        search_keys["user"] = user_name
        await self._purge_user_data(search_keys)
        logger.info("Deleted files from Solr and MongoDB with keys: %s", search_keys)


class STAC(Solr):
    """STAC class to create static and dynamic STAC catalogues."""

    def __init__(
        self,
        config: ServerConfig,
        *,
        uniq_key: Literal["file", "uri"] = "file",
        flavour: FlavourType = "freva",
        start: int = 0,
        multi_version: bool = True,
        translate: bool = True,
        _translator: Union[None, Translator] = None,
        **query: list[str],
    ):
        super().__init__(
            config,
            uniq_key=uniq_key,
            flavour=flavour,
            start=start,
            multi_version=multi_version,
            translate=translate,
            _translator=_translator,
            **query
        )
        self.config = config
        self.buffer = io.BytesIO()
        self.tar = tarfile.open(fileobj=self.buffer, mode='w:gz')
        self.spatial_extent = {
            "minx": float("-180"),
            "miny": float("-90"),
            "maxx": float("180"),
            "maxy": float("90"),
        }
        self.temporal_extent: dict[str, Optional[datetime]] = {
            "start": None,
            "end": None,
        }
        self.assets_prereqs: Dict[str, Union[str, int]] = {}

    @classmethod
    async def validate_parameters(
        cls,
        config: ServerConfig,
        *,
        uniq_key: Literal["file", "uri"] = "file",
        flavour: FlavourType = "freva",
        start: int = 0,
        multi_version: bool = False,
        translate: bool = True,
        **query: list[str],
    ) -> "STAC":
        """Use Solr validate_parameters and return a STAC validated_parameters
        for validating the search params through Solr in STAC inheritated cls."""
        solr_instance = await super().validate_parameters(
            config,
            uniq_key=uniq_key,
            flavour=flavour,
            start=start,
            multi_version=multi_version,
            translate=translate,
            **query
        )

        return cls(
            config,
            uniq_key=uniq_key,
            flavour=flavour,
            start=start,
            multi_version=multi_version,
            translate=translate,
            _translator=solr_instance.translator,
            **query
        )

    async def _create_stac_collection(self, collection_id: str) -> pystac.Collection:
        """Create a STAC collection from the Solr search results."""

        usage_desc = dedent(
            f"""
            ## {self.translator.flavour.upper()} Dataset Collection

            This is a collection from flavour **{self.translator.flavour}**

            **`Attention`:** _Item ingestion in progress. The STAC collection
            is currently being populated with items. This collection will be
            ready for use when this notice disappears upon refresh._

            ___
            ℹ️ Contact the Freva team at [freva@dkrz.de](mailto:freva@dkrz.de)
            to ask questions or report issues.
        """
        ).strip()
        intake_desc = dedent(
            f"""
            # Installing Intake-ESM
            ```bash
            # Method 1: Using pip
            pip install intake-esm
            # Method 2: Using conda (recommended)
            conda install -c conda-forge intake-esm
            ```
            # Quick Guide: INTAKE-ESM Catalog on Levante (Python)
            ```python
            import intake
            # create a catalog object from a EMS JSON file containing dataset metadata
            cat = intake.open_esm_datastore(
            '{str(self.assets_prereqs.get('full_endpoint')).replace(
                "stac-catalogue", "intake-catalogue")}')
            ```
        """
        )

        stac_static_desc = dedent(
            f"""
            # STAC Static Catalog Setup
            ```bash
            pip install pystac
            ```
            # Load the STAC Catalog
            ```python
            import pystac
            import tarfile
            import tempfile
            import os
            temp_dir = tempfile.mkdtemp(dir='/tmp')
            with tarfile.open('stac-catalog-{collection_id}-{self.uniq_key}.tar.gz',
                            mode='r:gz') as tar:
                tar.extractall(path=temp_dir)
            cat = pystac.Catalog.from_file(os.path.join(temp_dir, 'catalog.json'))
            ```
            💡: This has been desigend to work with the data locally. So you
            can copy the catalog link from here and download and load the catalog
            locally via the provided script.
            """
        )

        params_dict = (
            ast.literal_eval(str(self.assets_prereqs.get("only_params")))
            if self.assets_prereqs.get("only_params", "")
            else {}
        )
        python_params = " ".join(
            f"{k}={v},"
            for k, v in params_dict.items()
            if k not in ("translate", "stac_dynamic")
        )
        cli_params = " ".join(
            f"{k}={v}" for k, v in params_dict.items()
            if k not in ("translate", "stac_dynamic")
        )
        api_params = "&".join(
            f"{k}={v}" for k, v in params_dict.items()
            if k not in ("translate", "stac_dynamic")
        )

        zarr_desc = dedent(
            f"""
            # Accessing Zarr Data
            1. Install freva-client
            ```bash
            pip install freva-client
            ```
            2. Get the auth token and access the zarr data (Python) - recommended
            ```python
            from freva_client import authenticate, databrowser
            import xarray as xr
            token_info = authenticate(username=<your_username>,\\
                                    host='{self.config.proxy}')
            db = databrowser({python_params} stream_zarr=True,\\
                                host='{self.config.proxy}')
            xarray_dataset = xr.open_mfdataset(list(db))
            ```
            3. Get the auth token and access the zarr data (CLI)
            ```bash
            token=$(freva-client auth -u <username> --host {self.config.proxy}\\
                                                        |jq -r .access_token)
            freva-client databrowser {cli_params} --stream-zarr\\
                            --host {self.config.proxy}
            ```
            4. Access the zarr data directly (API - language agnostic)
            ```bash
            curl -X GET {self.assets_prereqs.get('base_url')}api/ \\
            freva-nextgen/databrowser/load/\\
            {self.translator.flavour}?{api_params} \\
            -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
            ```
            💡: Read more about the
            [freva-client](https://freva-clint.github.io/freva-nextgen/)
            """
        )

        local_access_desc = dedent(
            f"""
            # Accessing data locally where the data is stored

            ```python
            import xarray as xr
            ds = xr.open_mfdataset(list('{str(self.assets_prereqs.get("full_endpoint"))
                                          .replace("stac-catalogue", "data-search")}'))
            ```

            💡: Please make sure to have the required xarray packages installed.
            """
        )
        collection = pystac.Collection(
            id=collection_id,
            title=collection_id,
            description=usage_desc.strip(),
            extent=pystac.Extent(
                # We need to define an initial temporal and spatial extent,
                # and then we update those values as we iterate over the items
                spatial=pystac.SpatialExtent([[-180.0, -90.0, 180.0, 90.0]]),
                temporal=pystac.TemporalExtent([[None, None]]),  # type: ignore
            ),
        )
        assets = {
            "freva-databrowser": pystac.Asset(
                href=(
                    f"{self.assets_prereqs.get('base_url')}databrowser/?"
                    f"{api_params}"
                ),
                title="Freva Web Data-Browser",
                description=(
                    "Interactive web interface for data exploration and analysis. "
                    "Access through any browser."
                ),
                roles=["overview"],
                media_type="text/html",
            ),
            "intake-catalogue": pystac.Asset(
                href=str(self.assets_prereqs.get("full_endpoint")).replace(
                    "stac-catalogue", "intake-catalogue"
                ),
                title="Intake-ESM Catalogue",
                description=intake_desc,
                roles=["metadata"],
                media_type="application/json",
            ),
            "stac-static-catalogue": pystac.Asset(
                href=str(self.assets_prereqs.get("full_endpoint")).replace(
                    "stac_dynamic=true", "stac_dynamic=false"
                ),
                title="STAC Static Catalogue",
                description=stac_static_desc,
                roles=["metadata"],
                media_type="application/geopackage+sqlite3",
            ),
            "local-access": pystac.Asset(
                href=str(self.assets_prereqs.get("full_endpoint")).replace(
                    "stac-catalogue", "data-search"
                ),
                title="Access data locally",
                description=local_access_desc,
                roles=["data"],
                media_type="application/netcdf",
            ),
            "zarr-access": pystac.Asset(
                href=(
                    f"{self.assets_prereqs.get('base_url')}api/freva-nextgen/"
                    f"databrowser/load/{self.translator.flavour}?"
                    f"{api_params}"
                ),
                title="Stream Zarr Dataset",
                description=zarr_desc,
                roles=["data"],
                media_type="application/vnd+zarr",
                extra_fields={
                    "requires": ["oauth2"],
                    "authentication": {
                        "type": "oauth2",
                        "description": (
                            "Authentication using your Freva credentials is required."
                        ),
                    },
                },
            ),
        }

        for key, asset in assets.items():
            collection.add_asset(key, asset)

        if self.facets:
            collection.extra_fields["search_keys"] = {
                key: values for key, values in self.facets.items()
            }
        else:
            logger.info("No search keys found for collection")

        collection.providers = [
            pystac.Provider(
                name="Freva DataBrowser",
                url="https://www.freva.dkrz.de",
            )
        ]
        return collection

    async def _iter_stac_items(self) -> AsyncIterator[List[pystac.Item]]:
        self.query["cursorMark"] = "*"
        items_batch = []
        while True:
            async with self._session_get() as res:
                _, results = res
            for result in results.get("response", {}).get("docs", [{}]):
                item = await self._create_stac_item(result)
                items_batch.append(item)

                if len(items_batch) >= self.batch_size:
                    yield items_batch
                    items_batch = []

            if items_batch:
                yield items_batch
            next_cursor_mark = results.get("nextCursorMark", None)
            if next_cursor_mark == self.query["cursorMark"] or not results:
                break
            self.query["cursorMark"] = next_cursor_mark

    def parse_datetime(self, time_str: str) -> Tuple[datetime, datetime]:
        """
        Parse a time range string into start and end datetimes.

        Parameters
        ----------
        time_str : str
            Time range string in rdate format '[start_time TO end_time]'

        Returns
        -------
        Tuple[datetime, datetime]
            Start and end datetime objects
        """
        clean_start_time = time_str.replace("[", "").split(" TO ")[0]
        clean_end_time = time_str.replace("]", "").split(" TO ")[1]
        return parser.parse(clean_start_time), parser.parse(clean_end_time)

    def parse_bbox(self, bbox_str: Union[str, List[str]]) -> List[float]:
        """
        Parse a bounding box string into coordinates.

        Parameters
        ----------
        bbox_str : Union[str, List[str]]
            Bounding box in ENVELOPE format: 'ENVELOPE(west,east,north,south)'
            or as a list with one element: ['ENVELOPE(west,east,north,south)']

        Returns
        -------
        List[float]
            Coordinates as [minx, miny, maxx, maxy]
        """
        bbox = bbox_str[0] if isinstance(bbox_str, list) else bbox_str

        nums = [
            float(x)
            for x in bbox.replace("ENVELOPE(", "").replace(")", "").split(",")
        ]
        return [nums[0], nums[3], nums[1], nums[2]]

    def _update_spatial_extent(self, bbox: List[float]) -> None:
        """
        Update collection's spatial extent based on item bbox.

        Parameters
        ----------
        bbox : List[float]
            Bounding box coordinates [minx, miny, maxx, maxy]
        """
        self.spatial_extent["minx"] = min(self.spatial_extent["minx"], bbox[0])
        self.spatial_extent["miny"] = min(self.spatial_extent["miny"], bbox[1])
        self.spatial_extent["maxx"] = max(self.spatial_extent["maxx"], bbox[2])
        self.spatial_extent["maxy"] = max(self.spatial_extent["maxy"], bbox[3])

    def _update_temporal_extent(self, start_time: datetime, end_time: datetime) -> None:
        """Update collection's temporal extent based on item timerange.

        Parameters
        ----------
        start_time : datetime
            Item's start datetime
        end_time : datetime
            Item's end datetime
        """
        if self.temporal_extent["start"] is None:
            self.temporal_extent["start"] = start_time
        elif start_time < self.temporal_extent["start"]:
            self.temporal_extent["start"] = start_time

        if self.temporal_extent["end"] is None:
            self.temporal_extent["end"] = end_time
        elif end_time > self.temporal_extent["end"]:
            self.temporal_extent["end"] = end_time

    async def _create_stac_item(self, result: Dict[str, Any]) -> pystac.Item:
        """Create a STAC Item from a result dictionary.

        Parameters
        ----------
            result (Dict[str, Any]): Dictionary containing item metadata

        Returns
        --------
            pystac.Item: Created STAC item
        """
        id = result.get(self.uniq_key, "")
        params_dict = (
            ast.literal_eval(str(self.assets_prereqs.get("only_params")))
            if self.assets_prereqs.get("only_params", "")
            else {}
        )
        python_params = " ".join(
            f"{k}={v},"
            for k, v in params_dict.items()
            if k not in ("translate", "stac_dynamic", "start")
        )

        cli_params = " ".join(
            f"{k}={v}" for k, v in params_dict.items()
            if k not in ("translate", "stac_dynamic", "start")
        )

        api_params = "&".join(
            f"{k}={v}" for k, v in params_dict.items()
            if k not in ("translate", "stac_dynamic", "start")
        )
        intake_desc = dedent(
            f"""
            # Installing Intake-ESM
            ```bash
            # Method 1: Using pip
            pip install intake-esm
            # Method 2: Using conda (recommended)
            conda install -c conda-forge intake-esm
            ```
            # Quick Guide: INTAKE-ESM Catalog (Python)
            ```python
            import intake
            # create a catalog object from a EMS JSON file containing dataset metadata
            cat = intake.open_esm_datastore('{
                str(self.assets_prereqs.get("full_endpoint")).replace(
                    "stac-catalogue",
                    "intake-catalogue"
                ) + f"&{self.uniq_key}={id}"
            }')
            ```
            """
        )

        zarr_desc = dedent(
            f"""
            # Accessing Zarr Data
            1. Install freva-client
            ```bash
            pip install freva-client
            ```
            2. Get the auth token and access the zarr data (Python) - recommended
            ```python
            from freva_client import authenticate, databrowser
            import xarray as xr
            token_info = authenticate(username=<your_username>, \\
                host='{self.config.proxy}')
            db = databrowser({python_params} {self.uniq_key}={id}, \\
                            stream_zarr=True, host='{self.config.proxy}')
            xarray_dataset = xr.open_mfdataset(list(db))
            ```
            3. Get the auth token and access the zarr data (CLI)
            ```bash
            token=$(freva-client auth -u <username> --host {self.config.proxy}\\
                                                        |jq -r .access_token)
            freva-client databrowser {cli_params} {self.uniq_key}={id} \\
                --stream-zarr --host {self.config.proxy}
            ```
            4. Access the zarr data directly (API - language agnostic)
            ```bash
            curl -X GET {self.assets_prereqs.get('base_url')}api/ \\
            freva-nextgen/databrowser/load/\\
            {self.translator.flavour}?{api_params}\\
            &{self.uniq_key}={id} \\
              -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
            ```
            💡: Read more about the
            [freva-client](https://freva-clint.github.io/freva-nextgen/)
            """
        )  # noqa: E501

        local_access_desc = dedent(
            f"""
            # Accessing data locally where the data is stored
            ```python
            import xarray as xr
            ds = xr.open_mfdataset('{id}')
            ```
            💡: Please make sure to have the required xarray packages installed.
            """
        )

        normalized_id = (
            id.replace("https://", "")
            .replace("http://", "")
            .replace("/", "-")
            .replace(".", "-")
            .strip()
        )
        bbox = result.get("bbox")
        if bbox:
            try:
                bbox = self.parse_bbox(bbox)
                self._update_spatial_extent(bbox)
            except ValueError as e:  # pragma: no cover
                logger.warning(f"Invalid bbox for {id}: {e}")
                bbox = None

        time = result.get("time")
        start_time = end_time = None
        if time:
            try:
                start_time, end_time = self.parse_datetime(time)
                self._update_temporal_extent(start_time, end_time)
            except ValueError as e:
                logger.warning(f"Invalid datetime for {id}: {e}")

        geometry = None
        if bbox:
            geometry = {
                "type": "Polygon",
                "coordinates": [
                    [
                        [bbox[0], bbox[1]],
                        [bbox[2], bbox[1]],
                        [bbox[2], bbox[3]],
                        [bbox[0], bbox[3]],
                        [bbox[0], bbox[1]],
                    ]
                ],
            }

        properties = {
            **{
                k: result.get(k)
                for k in self._config.solr_fields
                if k in result and result.get(k) is not None
            },
            "title": id,
        }
        item = pystac.Item(
            id=normalized_id,
            collection=self.collection.id,
            geometry=geometry,
            bbox=bbox,
            datetime=start_time or datetime.now(),
            properties=properties,
        )

        if start_time and end_time:
            item.common_metadata.start_datetime = start_time
            item.common_metadata.end_datetime = end_time
        assets = {
            "freva-databrowser": pystac.Asset(
                href=(
                    f"{self.assets_prereqs.get('base_url')}databrowser/?"
                    f"{api_params}"
                    f"&{self.uniq_key}={id}"
                ),
                title="Freva Web DataBrowser",
                description=(
                    "Access the Freva web interface for data exploration and analysis"
                ),
                roles=["overview"],
                media_type="text/html",
            ),
            "intake-catalogue": pystac.Asset(
                href=(
                    str(self.assets_prereqs.get("full_endpoint")).replace(
                        "stac-catalogue", "intake-catalogue"
                    )
                    + f"&{self.uniq_key}={id}"
                ),
                title="Intake Catalogue",
                description=intake_desc,
                roles=["metadata"],
                media_type="application/json",
            ),
            "zarr-access": pystac.Asset(
                href=(
                    f"{self.assets_prereqs.get('base_url')}api/freva-nextgen/"
                    f"databrowser/load/{self.translator.flavour}?"
                    f"{api_params}&{self.uniq_key}={id}"
                ),
                title="Stream Zarr Data",
                description=zarr_desc,
                roles=["data"],
                media_type="application/vnd+zarr",
                extra_fields={
                    "requires": ["oauth2"],
                    "authentication": {
                        "type": "oauth2",
                        "description": (
                            "Authentication using your Freva credentials is required."
                        ),
                    },
                },
            ),
            "local-access": pystac.Asset(
                href=(
                    f"{self.assets_prereqs.get('base_url')}api/freva-nextgen/"
                    f"databrowser/data-search/{self.translator.flavour}/"
                    f"{self.uniq_key}?"
                    f"{api_params}"
                    f"&{self.uniq_key}={id}"
                ),
                title="Access data locally",
                description=local_access_desc,
                roles=["data"],
                media_type="application/netcdf"
            ),
        }

        for key, asset in assets.items():
            item.add_asset(key, asset)

        return item

    def finalize_stac_collection(self) -> None:
        """
        Finalize STAC collection by updating spatial and temporal extents.

        Raises
        ------
        Exception
            If collection validation fails
        """
        collection_desc = dedent(
            f"""
            ## {self.translator.flavour.upper()} Flavour Dataset Collection

            A curated climate datasets STAC Collection from the
            `{self.translator.flavour.upper()}` flavour, and
            specific search parameters through the Freva
            databrowser. Includes standardized metadata and direct
            data access capabilities.

            ___
            ℹ️ Contact the Freva team at [freva@dkrz.de](mailto:freva@dkrz.de)
            to ask questions or report issues.
        """
        ).strip()
        if self.spatial_extent["minx"] != float("inf") and self.spatial_extent[
            "maxx"
        ] != float("-inf"):

            bbox = [
                self.spatial_extent["minx"],
                self.spatial_extent["miny"],
                self.spatial_extent["maxx"],
                self.spatial_extent["maxy"],
            ]
            self.collection.extent.spatial = pystac.SpatialExtent([bbox])

        if self.temporal_extent["start"] and self.temporal_extent["end"]:
            self.collection.extent.temporal = pystac.TemporalExtent(
                [[self.temporal_extent["start"], self.temporal_extent["end"]]]
            )
        self.collection.description = collection_desc
        try:
            self.collection.validate()
        except Exception as e:  # pragma: no cover
            logger.error(f"Collection validation failed: {e}")

    async def validate_stac(self) -> Tuple[int, int]:
        """Validate STAC API availability and get result counts."""
        self._set_catalogue_queries()
        self.query["facet.field"] = self._config.solr_fields + ["time", "bbox"]
        self.query["fl"] = [self.uniq_key] + self._config.solr_fields + ["time", "bbox"]
        async with self._session_get() as res:
            search_status, search = res
        total_count = int(search.get("response", {}).get("numFound", 0))
        return search_status, total_count

    async def init_stac_catalogue(
        self,
        request: Request,
    ) -> None:
        filtered_params = dict(request.query_params)
        filtered_params.pop('translate', None)
        filtered_params.pop('stac_dynamic', None)
        filtered_params.pop('start', None)

        self.assets_prereqs = {
            "base_url": str(self.config.proxy) + "/",
            "full_endpoint": (
                f"{self.config.proxy}/"
                f"{str(request.url).split(str(request.base_url))[-1]}"
            ),
            "only_params": str(filtered_params) if filtered_params != {} else "",
        }

    async def init_stac_dynamic_collection(
        self,
        collection_id: str
    ) -> None:
        try:
            self.collection = await self._create_stac_collection(collection_id)
            await self.ingest_stac_collection(self.collection)
        except Exception as e:
            logger.error(
                f"STAC collection creation failed for {collection_id}: {str(e)}"
            )
            raise

    async def stream_stac_catalogue(
        self,
        collection_id: str,
        stac_dynamic: bool,
    ) -> AsyncIterator[bytes]:
        """Initialize and stream a STAC catalogue from Databrowser search results.

        Parameters
        ----------
        collection_id : str
            Unique identifier for the STAC collection
        stac_dynamic : bool
            If True, use dynamic STAC API, if False create static catalog

        Yields
        ------
        bytes
            Chunks of the tar.gz archive when stac_dynamic is False
        """
        logger.info("Streaming STAC Catalogue for %s", collection_id)
        try:
            if stac_dynamic:
                async for item_batch in self._iter_stac_items():
                    await self.ingest_stac_item(item_batch)
                self.finalize_stac_collection()
                await self.update_stac_collection(self.collection)
            else:
                # STAC-Catalog
                async for chunk in self.stream_catalog(collection_id):
                    yield chunk

                # intial STAC-Collection
                self.collection = await self._create_stac_collection(collection_id)
                async for chunk in self.stream_collection(
                    self.collection.to_dict(),
                    collection_id
                ):
                    yield chunk

                # STAC-Items
                async for item_batch in self._iter_stac_items():
                    for item in item_batch:
                        async for chunk in self.stream_item(
                            item.to_dict(),
                            collection_id
                        ):
                            yield chunk

                # updated STAC-Collection
                self.finalize_stac_collection()
                async for chunk in self.stream_collection(
                    self.collection.to_dict(),
                    collection_id
                ):
                    yield chunk

                final_chunk = await self.close()
                if final_chunk:
                    yield final_chunk

        except Exception as e:
            logger.error(
                f"STAC collection creation failed for {collection_id}: {str(e)}"
            )
            raise

    async def ingest_stac_item(self, items: list[pystac.Item]) -> None:
        """
        Ingest bulk STAC Items into the catalog via API.

        Parameters
        ----------
        items: list[pystac.Item]
            List of STAC Items to be ingested using upsert method.

        Raises
        ------
        HTTPException
            If ingestion fails or server returns non-201 status.
        """
        url = self._config.get_stacapi_url("items", items[0].collection_id)
        items_dict = {
            "items": {item.id: item.to_dict() for item in items},
            "method": "upsert",
        }
        async with self._session_post(url, items_dict):
            pass

    async def ingest_stac_collection(self, collection: pystac.Collection) -> None:
        """
        Ingest a STAC Collection into the STAC-API Catalog.

        Parameters
        ----------
        collection: pystac.Collection
            STAC Collection to be ingested.

        Raises
        ------
        HTTPException
            If ingestion fails or server returns non-201 status.
        """
        url = self._config.get_stacapi_url("collections")
        async with self._session_post(url, collection.to_dict()):
            pass

    async def update_stac_collection(self, collection: pystac.Collection) -> None:
        """
        Update existing STAC Collection in the STAC_API Catalogue.

        Parameters
        ----------
        collection: pystac.Collection
            STAC Collection with updated data.

        Raises
        ------
        HTTPException
            If update fails or server returns non-200 status.
        """
        url = f"{self._config.get_stacapi_url('collections')}/{collection.id}"
        async with self._session_put(url, collection.to_dict()):
            pass

    async def stacapi_availability(self) -> bool:
        """
        Check STAC API server availability via ping/pong endpoint.

        Returns
        -------
        bool
            True if server is available and returns 200 status.

        Raises
        ------
        HTTPException
            If server is unreachable or returns non-200 status.
        """
        original_url = getattr(self, "url", None)
        self.url = self._config.get_stacapi_url("ping")
        try:
            async with self._session_get() as res:
                return res[0] == 200
        except Exception as error:
            logger.error("STAC server connection failed: %s", error)
            raise HTTPException(status_code=503, detail="STAC server unreachable")
        finally:
            self.url = original_url or ""

    def add_object(
            self, name: str,
            content: str,
            mtime: Optional[float] = None) -> bytes:
        """Add an object of STAC such as Catalog, Collection or Item
        to the tgz archive and return the bytes."""
        content_bytes = content.encode('utf-8')
        info = tarfile.TarInfo(name=name)
        info.size = len(content_bytes)
        info.mtime = mtime or datetime.now().timestamp()

        content_io = io.BytesIO(content_bytes)
        self.tar.addfile(info, content_io)

        chunk = self.buffer.getvalue()
        self.buffer.seek(0)
        self.buffer.truncate()
        return chunk

    async def finalize_tar(self) -> bytes:
        """Close the tgz file and return final bytes."""
        self.tar.close()
        final_chunk = self.buffer.getvalue()
        self.buffer.close()
        return final_chunk

    async def stream_catalog(self, collection_id: str) -> AsyncIterator[bytes]:
        catalog = {
            "type": "Catalog",
            "stac_version": "1.0.0",
            "id": "static-catalog",
            "description": "Static STAC catalog for Freva databrowser search",
            "links": [
                {
                    "rel": "root",
                    "href": "./catalog.json",
                    "type": "application/json"
                },
                {
                    "rel": "child",
                    "href": f"./collections/{collection_id}/collection.json",
                    "type": "application/json"
                }
            ]
        }
        chunk = self.add_object(
            "stac-catalog/catalog.json",
            json.dumps(catalog, indent=2)
        )
        if chunk:
            yield chunk

    async def stream_collection(
        self,
        collection: Dict[str, Any],
        collection_id: str
    ) -> AsyncIterator[bytes]:
        collection["links"] = [
            {
                "rel": "root",
                "href": "../../catalog.json",
                "type": "application/json"
            },
            {
                "rel": "parent",
                "href": "../../catalog.json",
                "type": "application/json"
            },
            {
                "rel": "items",
                "href": "./items",
                "type": "application/json"
            }
        ]

        chunk = self.add_object(
            f"stac-catalog/collections/{collection_id}/collection.json",
            json.dumps(collection, indent=2)
        )
        if chunk:
            yield chunk

    async def stream_item(
        self, item: Dict[str, Any], collection_id: str
    ) -> AsyncIterator[bytes]:
        chunk = self.add_object(
            f"stac-catalog/collections/{collection_id}/items/{item['id']}.json",
            json.dumps(item, indent=2)
        )
        if chunk:
            yield chunk

    async def close(self) -> bytes:
        """Close the tgz archive and return final bytes."""
        return await self.finalize_tar()
