"""Main script that runs the rest API."""

import os
from typing import Annotated, List, Literal, Union

from fastapi import Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from freva_rest.auth import TokenPayload, get_current_user
from freva_rest.rest import app, server_config

from .core import FlavourType, SolrSearch, Translator
from .schema import Required, SearchFlavours, SolrSchema


@app.get("/api/databrowser/overview", tags=["Data search"])
async def overview() -> SearchFlavours:
    """Get all available search flavours and thier attributes.

    This endpoint allows you to retrieve an overview of the different
    Data Reference Syntax (DRS) standards implemented in the Freva Databrowser
    REST API. The DRS standards define the structure and metadata organisation
    for climate datasets, and each standard offers specific attributes for
    searching and filtering datasets.
    """
    attributes = {}
    for flavour in Translator.flavours:
        translator = Translator(flavour)
        if flavour in ("cordex",):
            attributes[flavour] = list(translator.foreward_lookup.values())
        else:
            attributes[flavour] = [
                f
                for f in translator.foreward_lookup.values()
                if f not in translator.cordex_keys
            ]
    return SearchFlavours(flavours=list(Translator.flavours), attributes=attributes)


@app.get(
    "/api/databrowser/intake_catalogue/{flavour}/{uniq_key}",
    tags=["Data search"],
)
async def intake_catalogue(
    flavour: FlavourType,
    uniq_key: Literal["file", "uri"],
    start: Annotated[int, SolrSchema.params["start"]] = 0,
    multi_version: Annotated[bool, SolrSchema.params["multi_version"]] = False,
    translate: Annotated[bool, SolrSchema.params["translate"]] = True,
    max_results: Annotated[int, SolrSchema.params["max_results"]] = -1,
    request: Request = Required,
) -> StreamingResponse:
    """Create an intake catalogue from a freva search.

    This endpoint generates an intake-esm catalogue in JSON format from a
    `freva` search. The catalogue includes metadata about the datasets found in
    the search results. Intake-esm is a data cataloging system that allows
    easy organization, discovery, and access to Earth System Model (ESM) data.
    The generated catalogue can be used by tools compatible with intake-esm,
    such as Pangeo.
    """
    solr_search = await SolrSearch.validate_parameters(
        server_config,
        flavour=flavour,
        uniq_key=uniq_key,
        start=start,
        multi_version=multi_version,
        translate=translate,
        **SolrSchema.process_parameters(request),
    )
    status_code, result = await solr_search.init_intake_catalogue()
    await solr_search.store_results(result.total_count, status_code)
    if result.total_count == 0:
        raise HTTPException(status_code=404, detail="No results found.")
    if result.total_count > max_results and max_results > 0:
        raise HTTPException(status_code=413, detail="Result stream too big.")
    file_name = f"IntakeEsmCatalogue_{flavour}_{uniq_key}.json"
    return StreamingResponse(
        solr_search.intake_catalogue(result.catalogue),
        status_code=status_code,
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="{file_name}"'},
    )


@app.get(
    "/api/databrowser/metadata_search/{flavour}/{uniq_key}",
    tags=["Data search"],
)
async def metadata_search(
    flavour: FlavourType,
    uniq_key: Literal["file", "uri"],
    multi_version: Annotated[bool, SolrSchema.params["multi_version"]] = False,
    translate: Annotated[bool, SolrSchema.params["translate"]] = True,
    facets: Annotated[Union[List[str], None], SolrSchema.params["facets"]] = None,
    request: Request = Required,
) -> JSONResponse:
    """Get the search facets.

    This endpoint allows you to search metadata (facets) based on the
    specified Data Reference Syntax (DRS) standard (`flavour`) and the type of
    search result (`uniq_key`), which can be either `file` or `uri`.
    Facets represent the metadata categories associated with the climate
    datasets, such as experiment, model, institute, and more. This method
    provides a comprehensive view of the available facets and their
    corresponding counts based on the provided search criteria.
    """
    solr_search = await SolrSearch.validate_parameters(
        server_config,
        flavour=flavour,
        uniq_key=uniq_key,
        multi_version=multi_version,
        translate=translate,
        start=0,
        **SolrSchema.process_parameters(request),
    )
    status_code, result = await solr_search.extended_search(facets or [], max_results=0)
    await solr_search.store_results(result.total_count, status_code)
    output = result.dict()
    del output["search_results"]
    return JSONResponse(content=output, status_code=status_code)


@app.get(
    "/api/databrowser/extended_search/{flavour}/{uniq_key}",
    tags=["Data search"],
)
async def extended_search(
    flavour: FlavourType,
    uniq_key: Literal["file", "uri"],
    start: Annotated[int, SolrSchema.params["start"]] = 0,
    multi_version: Annotated[bool, SolrSchema.params["multi_version"]] = False,
    translate: Annotated[bool, SolrSchema.params["translate"]] = True,
    max_results: Annotated[int, SolrSchema.params["batch_size"]] = 150,
    facets: Annotated[Union[List[str], None], SolrSchema.params["facets"]] = None,
    request: Request = Required,
) -> JSONResponse:
    """Get the search facets."""
    solr_search = await SolrSearch.validate_parameters(
        server_config,
        flavour=flavour,
        uniq_key=uniq_key,
        start=start,
        multi_version=multi_version,
        translate=translate,
        **SolrSchema.process_parameters(request),
    )
    status_code, result = await solr_search.extended_search(
        facets or [], max_results=max_results
    )
    await solr_search.store_results(result.total_count, status_code)
    return JSONResponse(content=result.dict(), status_code=status_code)


@app.get("/api/databrowser/data_search/{flavour}/{uniq_key}", tags=["Data search"])
async def data_search(
    flavour: FlavourType,
    uniq_key: Literal["file", "uri"],
    start: Annotated[int, SolrSchema.params["start"]] = 0,
    multi_version: Annotated[bool, SolrSchema.params["multi_version"]] = False,
    translate: Annotated[bool, SolrSchema.params["translate"]] = True,
    request: Request = Required,
) -> StreamingResponse:
    """Search for datasets.

    This endpoint allows you to search for climate datasets based on the
    specified Data Reference Syntax (DRS) standard (`flavour`) and the type of
    search result (`uniq_key`), which can be either "file" or "uri". The
    `databrowser` method provides a flexible and efficient way to query
    datasets matching specific search criteria and retrieve a list of data
    files or locations that meet the query parameters.
    """
    solr_search = await SolrSearch.validate_parameters(
        server_config,
        flavour=flavour,
        uniq_key=uniq_key,
        start=start,
        multi_version=multi_version,
        translate=translate,
        **SolrSchema.process_parameters(request),
    )
    status_code, total_count = await solr_search.init_stream()
    await solr_search.store_results(total_count, status_code)
    return StreamingResponse(
        solr_search.stream_response(),
        status_code=status_code,
        media_type="text/plain",
    )


@app.get(
    "/api/databrowser/load/{flavour}",
    status_code=status.HTTP_201_CREATED,
    tags=["Load data"],
)
async def load_data(
    flavour: FlavourType,
    start: Annotated[int, SolrSchema.params["start"]] = 0,
    multi_version: Annotated[bool, SolrSchema.params["multi_version"]] = False,
    translate: Annotated[bool, SolrSchema.params["translate"]] = True,
    catalogue_type: Annotated[
        Literal["intake", None],
        Query(
            title="Catalogue type",
            alias="catalogue-type",
            description=(
                "Set the type of catalogue you want to create from this" "query"
            ),
        ),
    ] = None,
    request: Request = Required,
    current_user: TokenPayload = Depends(get_current_user),
) -> StreamingResponse:
    """Search for datasets and stream the results as zarr."""
    if "zarr-stream" not in os.getenv("API_SERVICES", ""):
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Service not enabled.",
        )
    solr_search = await SolrSearch.validate_parameters(
        server_config,
        flavour=flavour,
        uniq_key="uri",
        start=start,
        multi_version=multi_version,
        translate=translate,
        **SolrSchema.process_parameters(request, "catalogue-type"),
    )
    _, total_count = await solr_search.init_stream()
    status_code = status.HTTP_201_CREATED
    if total_count < 1:
        status_code = status.HTTP_400_BAD_REQUEST
    await solr_search.store_results(total_count, status_code)
    return StreamingResponse(
        solr_search.zarr_response(catalogue_type, total_count),
        status_code=status_code,
        media_type="text/plain",
    )
