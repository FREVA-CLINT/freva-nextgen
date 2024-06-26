"""Schema definitions for the FastAPI enpoints."""

from typing import Any, Dict, List, Union
from urllib.parse import parse_qs

from fastapi import Query, Request
from pydantic import BaseModel

from .core import FlavourType

Required: Any = Ellipsis


class LoadFiles(BaseModel):
    """Schema for the load file endpoint response."""

    urls: List[str]


class SolrSchema:
    """Class holding all apache solr config parameters."""

    params: Dict[str, Any] = {
        "batch_size": Query(
            alias="max-results",
            title="Max. results",
            description="Control the number of maximum result items returned.",
            ge=0,
            le=1500,
        ),
        "start": Query(
            alias="start",
            title="Start",
            description="Specify the starting point for receiving results.",
            ge=0,
        ),
        "multi_version": Query(
            alias="multi-version",
            title="Multi Version",
            description="Use versioned datasets instead of latest versions.",
        ),
        "translate": Query(
            title="Translate",
            alias="translate",
            description="Translate the output to the required DRS flavour.",
        ),
        "facets": Query(
            title="Facets",
            alias="facets",
            description=(
                "The facets that should be part of the output, "
                "by default all facets will be returned."
            ),
        ),
        "max_results": Query(
            title="Max. Results",
            alias="max-results",
            description=(
                "Raise an Error if more results are found than that"
                "number, -1 for all results."
            ),
        ),
    }

    @classmethod
    def process_parameters(
        cls, request: Request, *parameters_not_to_process: str
    ) -> Dict[str, list[str]]:
        """Convert Starlette Request QueryParams to a dictionary."""

        query = parse_qs(str(request.query_params))
        for key in ("uniq_key", "flavour") + parameters_not_to_process:
            _ = query.pop(key, [""])
        for key, param in cls.params.items():
            _ = query.pop(key, [""])
            _ = query.pop(param.alias, [""])

        return query


class FacetResults(BaseModel):
    """Schema for facets results."""

    facets: Dict[str, List[Union[str, int]]]


class SearchFlavours(BaseModel):
    """Schema for search flavours."""

    flavours: List[FlavourType]
    attributes: Dict[FlavourType, List[str]]
