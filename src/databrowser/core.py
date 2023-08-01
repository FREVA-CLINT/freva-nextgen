"""The core functionality to interact with the apache solr search system."""


from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from json import JSONEncoder
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Literal,
    Tuple,
    TypedDict,
    Union,
    cast,
)

import aiohttp
from dateutil.parser import ParserError, parse
from fastapi import HTTPException
from pydantic import BaseModel

from ._version import __version__
from .config import ServerConfig
from .logger import logger

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
        "catalog_dict": List[Dict[str, Any]],
    },
)


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
    def _freva_facets(self) -> dict[str, str]:
        """Define the freva search facets and their relevance"""
        return {
            "experiment": "primary",
            "ensemble": "primary",
            "fs_type": "secundary",
            "grid_label": "secundary",
            "institute": "primary",
            "model": "primary",
            "project": "primary",
            "product": "primary",
            "realm": "primary",
            "variable": "parimary",
            "time_aggregation": "primary",
            "time_frequency": "primary",
            "cmor_table": "secundary",
            "dataset": "secundary",
            "driving_model": "secundary",
            "format": "secundary",
            "grid_id": "secundary",
            "level_type": "secundary",
            "rcm_name": "secundary",
            "rcm_version": "secundary",
        }

    @property
    def _cmip5_lookup(self) -> dict[str, str]:
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
    def _cmip6_lookup(self) -> dict[str, str]:
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
    def _cordex_lookup(self) -> dict[str, str]:
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
    def _nextgems_lookup(self) -> dict[str, str]:
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
    def foreward_lookup(self) -> dict[str, str]:
        """Define how things get translated from the freva standard"""
        return {
            "freva": {k: k for k in self._freva_facets},
            "cmip6": self._cmip6_lookup,
            "cmip5": self._cmip5_lookup,
            "cordex": self._cordex_lookup,
            "nextgems": self._nextgems_lookup,
        }[self.flavour]

    @property
    def cordex_keys(self) -> Tuple[str, ...]:
        """Define the keys that make a cordex dataset."""
        return ("rcm_name", "driving_model", "rcm_version")

    @cached_property
    def primary_keys(self) -> list[str]:
        """Define which search facets are primary for which standard."""
        _keys = [
            self.foreward_lookup[k]
            for (k, v) in self._freva_facets.items()
            if v == "primary"
        ]
        if self.flavour in ("cordex",):
            for key in self.cordex_keys:
                _keys.append(key)
        return _keys

    @cached_property
    def backward_lookup(self) -> dict[str, str]:
        """Translate the schema to the freva standard."""
        return {v: k for (k, v) in self.foreward_lookup.items()}

    def translate_facets(
        self,
        facets: Dict[str, Any],
        backwards: bool = False,
    ) -> Dict[str, Any]:
        """Translate the facets names to a given flavour."""
        if self.translate:
            if backwards:
                return {
                    self.backward_lookup.get(k, k): v
                    for (k, v) in facets.items()
                }
            return {
                self.foreward_lookup.get(k, k): v for (k, v) in facets.items()
            }
        return facets


class SolrSearch:
    """Definitions for makeing search queries on apache solr.

    Parameters
    ----------
    config: ServerConfig
        An instance of the server configuration class.
    query_params: str
        String representation of the apache solr search facets used for qeury.

    Attributes
    ----------
    """

    uniq_keys: Tuple[str, str] = ("file", "uri")
    """The names of all unique keys in the indexing system."""

    timeout: aiohttp.ClientTimeout = aiohttp.ClientTimeout(total=5)
    """5 seconds for timeout."""

    batch_size: int = 150
    """Maximum solr batch query size for one single query result."""

    def __init__(
        self,
        config: ServerConfig,
        *,
        uniq_key: Literal["file", "uri"] = "file",
        flavour: FlavourType = "freva",
        batch_size: int = 150,
        start: int = 0,
        multi_version: bool = False,
        translate: bool = True,
        **query: list[str],
    ) -> None:
        self._config = config
        self.uniq_key = uniq_key
        self.batch_size = batch_size
        self.multi_version = multi_version
        self.translator = Translator(flavour, translate)
        translate = query.pop("translate", ["1"])[0].lower() in ("1", "true")
        try:
            self.time = self.adjust_time_string(
                query.pop("time", [""])[0],
                query.pop("time_select", ["flexible"])[0],
            )
        except ValueError as err:
            raise HTTPException(status_code=500, detail=str(err)) from err
        self.facets = self.translator.translate_facets(query, backwards=True)
        self.url, self.query = self._get_url()
        self.query["start"] = start
        self.query["sort"] = f"{self.uniq_key} desc"

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

    async def init_intake_catalogue(self) -> Tuple[int, IntakeCatalogue]:
        """Create an intake catalogue from the solr search."""
        self.query["start"] = 0
        self.query["facet"] = "true"
        self.query["facet.mincount"] = "1"
        self.query["facet.limit"] = "-1"
        self.query["rows"] = self.batch_size
        self.query["facet.field"] = self._config.solr_fields
        self.query["fl"] = [self.uniq_key] + self._config.solr_fields
        self.query["wt"] = "json"
        logger.info(
            "Query %s for uniq_key: %s with %s",
            self.url,
            self.uniq_key,
            self.query,
        )
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.get(self.url, params=self.query) as res:
                search_status = res.status
                try:
                    await self.check_for_status(res)
                    search = await res.json()
                except HTTPException:
                    search = {}
        total_count = cast(int, search.get("response", {}).get("numFound", 0))
        facets = search.get("facet_counts", {}).get("facet_fields", {})
        var_name = self.translator.foreward_lookup["variable"]
        facets = [
            self.translator.foreward_lookup.get(v, v)
            for v in self.translator.facet_hierachy
            if facets.get(v)
        ]
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
            "catalog_dict": [],
        }
        for result in search.get("response", {}).get("docs", []):
            source = {
                k: result[k]
                for k in [self.uniq_key] + self.translator.facet_hierachy
                if result.get(k)
            }
            catalogue["catalog_dict"].append(
                self.translator.translate_facets(source)
            )
        return search_status, IntakeCatalogue(
            catalogue=catalogue, total_count=total_count
        )

    async def _iterintake(self) -> AsyncIterator[str]:
        encoder = JSONEncoder(indent=3)
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.get(self.url, params=self.query) as res:
                results = await res.json()
                for out in results.get("response", {}).get("docs", [{}]):
                    source = {
                        k: out[k]
                        for k in [self.uniq_key]
                        + self.translator.facet_hierachy
                        if out.get(k)
                    }
                    entry = self.translator.translate_facets(source)
                    yield ",\n   "
                    for line in list(encoder.iterencode(entry)):
                        yield line

    async def intake_catalogue(
        self, search: IntakeCatalogue
    ) -> AsyncIterator[str]:
        """Create an intake catalogue from the solr search."""
        iteritems = tuple(
            range(self.batch_size + 1, search.total_count, self.batch_size)
        )
        encoder = JSONEncoder(indent=3)
        for line in list(encoder.iterencode(search.catalogue))[:-4]:
            yield line
        for i in iteritems:
            self.query["start"] = i
            self.query["rows"] = self.batch_size
            async for line in self._iterintake():
                yield line
        if (
            iteritems
            and iteritems[-1] < search.total_count
            and search.total_count > self.batch_size
        ):
            self.query["start"] = iteritems[-1]
            self.query["rows"] = search.total_count - iteritems[-1]
            async for line in self._iterintake():
                yield line
        yield "\n]\n}"

    async def metadata_search(
        self,
    ) -> Tuple[int, SearchResult]:
        """Initialise the apache solr metadata search.

        Returns
        -------
        int: status code of the apache solr query.
        """
        self.query["facet"] = "true"
        self.query["rows"] = self.batch_size
        self.query["facet.sort"] = "index"
        self.query["facet.mincount"] = "1"
        self.query["wt"] = "json"
        self.query["facet.field"] = self._config.solr_fields
        self.query["fl"] = [self.uniq_key, "fs_type"]
        logger.info(
            "Query %s for uniq_key: %s with %s",
            self.url,
            self.uniq_key,
            self.query,
        )
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.get(self.url, params=self.query) as res:
                search_status = res.status
                try:
                    await self.check_for_status(res)
                    search = await res.json()
                except HTTPException:
                    search = {}
        return search_status, SearchResult(
            total_count=search.get("response", {}).get("numFound", 0),
            facets=self.translator.translate_facets(
                search.get("facet_counts", {}).get("facet_fields", {})
            ),
            search_results=[
                {
                    **{self.uniq_key: k[self.uniq_key]},
                    **{"fs_type": k.get("fs_type", "posix")},
                }
                for k in search.get("response", {}).get("docs", [])
            ],
            facet_mapping=self.translator.foreward_lookup,
            primary_facets=self.translator.primary_keys,
        )

    async def init_stream(self) -> Tuple[int, SearchResult]:
        """Initialise the apache solr search.

        Returns
        -------
        int: status code of the apache solr query.
        """
        self.query["fl"] = [self.uniq_key]
        logger.info(
            "Query %s for uniq_key: %s with %s",
            self.url,
            self.uniq_key,
            self.query,
        )
        self.query["start"] = 0
        self.query["rows"] = self.batch_size
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.get(self.url, params=self.query) as res:
                search_status = res.status
                try:
                    await self.check_for_status(res)
                    search = await res.json()
                except HTTPException:
                    search = {}
        total_count = search.get("response", {}).get("numFound", 0)
        if not total_count:
            search_status = 400
        return search_status, SearchResult(
            total_count=search.get("response", {}).get("numFound", 0),
            facets={},
            search_results=search.get("response", {}).get("docs", []),
            facet_mapping=self.translator.foreward_lookup,
            primary_facets=[],
        )

    def _get_url(self) -> tuple[str, Dict[str, Any]]:
        """Get the url for the solr query."""
        core = {
            True: self._config.solr_cores[0],
            False: self._config.solr_cores[-1],
        }[self.multi_version]
        url = f"{self._config.get_core_url(core)}/select/"
        query = []
        for key, value in self.facets.items():
            search_value = " AND ".join(map(str.lower, value))
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
            )

    async def _make_iterable(self) -> AsyncIterator[str]:
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.get(self.url, params=self.query) as res:
                results = await res.json()
        for out in results.get("response", {}).get("docs", []):
            yield f"{out[self.uniq_key]}\n"

    async def stream_response(
        self,
        search: SearchResult,
    ) -> AsyncIterator[str]:
        """Search for uniq keys matching given search facets.

        Parameters
        ----------
        uniq_key: str, default: file
            The name of the unique key that is should be search for
        **facets: str,
            Search facets to refine the solr query.

        Returns
        -------
        UniqKeys: An instance of the pydantic UniqKey base model.
        """
        iteritems = tuple(
            range(self.batch_size + 1, search.total_count, self.batch_size)
        )
        for content in search.search_results:
            yield f"{content[self.uniq_key]}\n"
        for i in iteritems:
            self.query["start"] = i
            self.query["rows"] = self.batch_size
            async for uri in self._make_iterable():
                yield uri
        if (
            iteritems
            and iteritems[-1] < search.total_count
            and search.total_count > self.batch_size
        ):
            self.query["start"] = iteritems[-1]
            self.query["rows"] = search.total_count - iteritems[-1]
            async for uri in self._make_iterable():
                yield uri
