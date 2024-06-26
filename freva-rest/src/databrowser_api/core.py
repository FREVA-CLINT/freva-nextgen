"""The core functionality to interact with the apache solr search system."""

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property, wraps
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
    Sized,
    Tuple,
    Union,
    cast,
)

import aiohttp
from databrowser_api import __version__
from dateutil.parser import ParserError, parse
from fastapi import HTTPException
from freva_rest.config import ServerConfig
from freva_rest.logger import logger
from freva_rest.utils import create_redis_connection
from pydantic import BaseModel
from typing_extensions import TypedDict

FlavourType = Literal["freva", "cmip6", "cmip5", "cordex", "nextgems"]
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
        loop = asyncio.get_event_loop()
        return asyncio.ensure_future(async_func(*args, **kwargs), loop=loop)

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
    )

    @property
    def facet_hierachy(self) -> list[str]:
        """Define the hierachy of facets that define a dataset."""
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
    def foreward_lookup(self) -> Dict[str, str]:
        """Define how things get translated from the freva standard"""

        return {
            "freva": {k: k for k in self._freva_facets},
            "cmip6": self._cmip6_lookup,
            "cmip5": self._cmip5_lookup,
            "cordex": self._cordex_lookup,
            "nextgems": self._nextgems_lookup,
        }[self.flavour]

    @cached_property
    def valid_facets(self) -> list[str]:
        """Get all valid facets for a flavour."""
        if self.translate:
            return list(self.foreward_lookup.values())
        return list(self.foreward_lookup.keys())

    @property
    def cordex_keys(self) -> Tuple[str, ...]:
        """Define the keys that make a cordex dataset."""
        return ("rcm_name", "driving_model", "rcm_version")

    @cached_property
    def primary_keys(self) -> list[str]:
        """Define which search facets are primary for which standard."""
        if self.translate:
            _keys = [
                self.foreward_lookup[k]
                for (k, v) in self._freva_facets.items()
                if v == "primary"
            ]
        else:
            _keys = [
                k for (k, v) in self._freva_facets.items() if v == "primary"
            ]
        if self.flavour in ("cordex",):
            for key in self.cordex_keys:
                _keys.append(key)
        return _keys

    @cached_property
    def backward_lookup(self) -> Dict[str, str]:
        """Translate the schema to the freva standard."""
        return {v: k for (k, v) in self.foreward_lookup.items()}

    def translate_facets(
        self,
        facets: Iterable[str],
        backwards: bool = False,
    ) -> List[str]:
        """Translate the facets names to a given flavour."""
        if self.translate:
            if backwards:
                return [self.backward_lookup.get(f, f) for f in facets]
            return [self.foreward_lookup.get(f, f) for f in facets]
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


class SolrSearch:
    """Definitions for makeing search queries on apache solr.

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

    timeout: aiohttp.ClientTimeout = aiohttp.ClientTimeout(total=30)
    """30 seconds for timeout."""

    batch_size: int = 150
    """Maximum solr batch query size for one single query result."""

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
        multi_version: bool = False,
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
        except ValueError as err:
            raise HTTPException(status_code=500, detail=str(err)) from err
        self.facets = self.translator.translate_query(query, backwards=True)
        self.url, self.query = self._get_url()
        self.query["start"] = start
        self.query["sort"] = "file desc"

    @asynccontextmanager
    async def _session_get(self) -> AsyncIterator[Tuple[int, Dict[str, Any]]]:
        """Wrap the get request round a try and catch statement."""
        logger.info(
            "Query %s for uniq_key: %s with %s",
            self.url,
            self.uniq_key,
            self.query,
        )
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.get(self.url, params=self.query) as res:
                    status = res.status
                    try:
                        await self.check_for_status(res)
                        search = await res.json()
                    except HTTPException:  # pragma: no cover
                        search = {}  # pragma: no cover
            except Exception as error:
                logger.error("Connection to %s failed: %s", self.url, error)
                raise HTTPException(
                    status_code=503,
                    detail="Could not connect to search instance",
                )
        yield status, search

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
    ) -> "SolrSearch":
        """Create an instance of an SolrSearch class with parameter validation.

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
            if (
                key not in translator.valid_facets
                and key not in ("time_select",) + cls.uniq_keys
            ):
                raise HTTPException(
                    status_code=422, detail="Could not validate input."
                )
        return SolrSearch(
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
            start = parse(
                start or "1", default=datetime(1, 1, 1, 0, 0, 0)
            ).isoformat()
            end = parse(
                end or "9999", default=datetime(9999, 12, 31, 23, 59, 59)
            ).isoformat()
        except ParserError as exc:
            raise ValueError(exc) from exc
        return [f"{{!field f=time op={solr_select}}}[{start} TO {end}]"]

    async def _create_intake_catalogue(self, *facets: str) -> IntakeType:
        var_name = self.translator.foreward_lookup["variable"]
        catalogue: IntakeType = {
            "esmcat_version": "0.1.0",
            "attributes": [
                {
                    "column_name": v,
                    "vocabulary": "",
                }
                for v in facets
            ],
            "assets": {"column_name": "uri", "format_column_name": "format"},
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
            self.translator.foreward_lookup.get(v, v)
            for v in self.translator.facet_hierachy
            if facets.get(v)
        ]
        catalogue = await self._create_intake_catalogue(*facets)
        return search_status, IntakeCatalogue(
            catalogue=catalogue, total_count=total_count
        )

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
            await self._config.mongo_collection.insert_one(
                {"metadata": data, "query": facets}
            )
        except Exception as error:
            logger.warning("Could not add stats to mongodb: %s", error)

    def _process_catalogue_result(
        self, out: Dict[str, List[Sized]]
    ) -> Dict[str, Any]:
        return {
            k: (
                out[k][0]
                if isinstance(out.get(k), list) and len(out[k]) == 1
                else out.get(k)
            )
            for k in [self.uniq_key] + self.translator.facet_hierachy
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
                k: self.translator.foreward_lookup[k]
                for k in self.query["facet.field"]
                if k in self.translator.foreward_lookup
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

    def _get_url(self) -> tuple[str, Dict[str, Any]]:
        """Get the url for the solr query."""
        core = {
            True: self._config.solr_cores[0],
            False: self._config.solr_cores[-1],
        }[self.multi_version]
        url = f"{self._config.get_core_url(core)}/select/"
        query = []
        for key, value in self.facets.items():
            if key in self.uniq_keys:
                search_value = " OR ".join(map(str, value))
            else:
                search_value = " OR ".join(map(str.lower, value))
            for char in self.escape_chars:
                search_value = search_value.replace(char, "\\" + char)
            query.append(f"{key.lower()}:({search_value})")
        return url, {
            "fq": self.time + ["", " AND ".join(query) or "*:*"],
            "q": "*:*",
        }

    async def _post_url(self) -> tuple[str, Dict[str, Any]]:
        return "", {}  # pragma: no cover

    async def check_for_status(
        self, response: aiohttp.client_reqrep.ClientResponse
    ) -> None:
        """Ceck if a query was successful

        Parameters
        ----------
        response: aiohttp.client_reqrep.ClientResponse
            The response of the rest query.

        Raises
        ------
        fastapi.HTTPException: If anything went wrong an error is risen
        """
        if response.status not in (200, 201):
            raise HTTPException(
                status_code=response.status, detail=response.text
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
        api_path = (
            f"{os.environ.get('API_URL', '')}/api/freva-data-portal/zarr"
        )
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
            cache = await create_redis_connection()
            try:
                await cache.publish(
                    "data-portal",
                    json.dumps({"uri": {"path": uri, "uuid": uuid5}}).encode(
                        "utf-8"
                    ),
                )
            except Exception as error:
                logger.error("Cloud not connect to reddis: %s", error)
                yield "Internal error, service not available\n"
                continue
            output = f"{api_path}/{uuid5}.zarr"
            if catalogue_type == "intake":
                result[self.uniq_key] = output
                if num < num_results:
                    suffix = ","
                else:
                    suffix = ""
                output = json.dumps(
                    self._process_catalogue_result(result), indent=3
                )
                prefix = "   "
            num += 1
            yield f"{prefix}{output}{suffix}\n"

        if catalogue_type == "intake":
            yield "\n   ]\n}"
