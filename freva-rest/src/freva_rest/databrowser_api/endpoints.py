"""Main script that runs the rest API."""

from typing import Annotated, Any, Dict, List, Literal, Union

from fastapi import Body, Depends, HTTPException, Query, Request, status
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from pydantic import BaseModel, Field

from freva_rest.auth import TokenPayload, auth
from freva_rest.logger import logger
from freva_rest.rest import app, server_config

from .core import FlavourType, SearchResult, Solr, Translator
from .schema import Required, SearchFlavours, SolrSchema


class AddUserDataRequestBody(BaseModel):
    """Request body schema for adding user data."""

    user_metadata: List[Dict[str, str]] = Field(
        ...,
        description="List of user metadata objects or strings to be"
        " added to the databrowser.",
        examples=[
            [
                {
                    "variable": "tas",
                    "time_frequency": "mon",
                    "time": "[1979-01-16T12:00:00Z TO 1979-11-16T00:00:00Z]",
                    "file": "path of the file",
                },
            ]
        ],
    )
    facets: Dict[str, Any] = Field(
        ...,
        description="Key-value pairs representing metadata search attributes.",
        examples=[
            {"project": "user-data", "product": "new", "institute": "globe"}
        ],
    )


@app.get(
    "/api/freva-nextgen/databrowser/overview",
    tags=["Data search"],
    status_code=200,
    response_model=SearchFlavours,
)
async def overview() -> SearchFlavours:
    """Get all available search flavours and their attributes.

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
            attributes[flavour] = list(translator.forward_lookup.values())
        else:
            attributes[flavour] = [
                f
                for f in translator.forward_lookup.values()
                if f not in translator.cordex_keys
            ]
    return SearchFlavours(
        flavours=list(Translator.flavours), attributes=attributes
    )


@app.get(
    "/api/freva-nextgen/databrowser/metadata-search/{flavour}/{uniq_key}",
    tags=["Data search"],
    status_code=200,
    response_model=SearchResult,
    responses={
        422: {"description": "Invalid flavour or search keys."},
        503: {"description": "Search backend error"},
    },
)
async def metadata_search(
    flavour: FlavourType,
    uniq_key: Literal["file", "uri"],
    multi_version: Annotated[bool, SolrSchema.params["multi_version"]] = False,
    translate: Annotated[bool, SolrSchema.params["translate"]] = True,
    facets: Annotated[Union[List[str], None], SolrSchema.params["facets"]] = None,
    request: Request = Required,
) -> JSONResponse:
    """Query the available metadata.

    This endpoint allows you to search metadata (facets) based on the
    specified Data Reference Syntax (DRS) standard (`flavour`) and the type of
    search result (`uniq_key`), which can be either `file` or `uri`.
    Facets represent the metadata categories associated with the climate
    datasets, such as experiment, model, institute, and more. This method
    provides a comprehensive view of the available facets and their
    corresponding counts based on the provided search criteria.
    """
    solr_search = await Solr.validate_parameters(
        server_config,
        flavour=flavour,
        uniq_key=uniq_key,
        multi_version=multi_version,
        translate=translate,
        start=0,
        **SolrSchema.process_parameters(request),
    )
    status_code, result = await solr_search.extended_search(
        facets or [], max_results=0
    )
    await solr_search.store_results(result.total_count, status_code)
    output = result.dict()
    del output["search_results"]
    return JSONResponse(content=output, status_code=status_code)


@app.get(
    "/api/freva-nextgen/databrowser/data-search/{flavour}/{uniq_key}",
    tags=["Data search"],
    status_code=200,
    responses={
        422: {"description": "Invalid flavour or search keys."},
        503: {"description": "Search backend error"},
    },
    response_class=PlainTextResponse,
)
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
    solr_search = await Solr.validate_parameters(
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
    "/api/freva-nextgen/databrowser/intake-catalogue/{flavour}/{uniq_key}",
    tags=["Data search"],
    status_code=200,
    responses={
        422: {"description": "Invalid flavour or search keys."},
        503: {"description": "Search backend error"},
    },
    response_class=JSONResponse,
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
    solr_search = await Solr.validate_parameters(
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
    "/api/freva-nextgen/databrowser/extended-search/{flavour}/{uniq_key}",
    include_in_schema=False,
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
    """This endpoint is used by the databrowser web ui client."""
    solr_search = await Solr.validate_parameters(
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


@app.get(
    "/api/freva-nextgen/databrowser/load/{flavour}",
    status_code=status.HTTP_201_CREATED,
    tags=["Load data"],
    responses={
        401: {"description": "Unauthorised / not a valid token."},
        422: {"description": "Invalid flavour or search keys."},
        503: {"description": "Search backend error"},
    },
    response_class=PlainTextResponse,
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
    current_user: TokenPayload = Depends(auth.required_dependency()),
) -> StreamingResponse:
    """Search for datasets and stream the results as zarr.

    This endpoint works essentially just like the `data-search` endpoint with
    the only difference that you will get *temporary* endpoints to `zarr` urls.
    You can use these endpoints to access data via http.

    [!NOTE]
    The urls are only temporary and will be invalidated.
    """
    if "zarr-stream" not in server_config.services:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not enabled.",
        )
    solr_search = await Solr.validate_parameters(
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


@app.post(
    "/api/freva-nextgen/databrowser/userdata",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["User data"],
    response_class=JSONResponse,
    responses={
        401: {"description": "Unauthorised / not a valid token."},
        500: {"description": "Search backend error"},
    },
)
async def post_user_data(
    request: Annotated[AddUserDataRequestBody, Body(...)],
    current_user: TokenPayload = Depends(auth.required_dependency()),
) -> Dict[str, str]:
    """Index your own metadata and make it searchable.


    With help of this endpoint you can add your own data to the search index.
    After the data has been successfully added you can use the other endpoints
    like `data-search` or `metadata-search` to search for the data you've
    indexed.
    """

    solr_instance = Solr(server_config)
    try:
        try:
            validated_user_metadata = await solr_instance._validate_user_metadata(
                request.user_metadata
            )
        except HTTPException as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid request data: {error}",
            )
        status_msg = await solr_instance.add_user_metadata(
            current_user.preferred_username,  # type: ignore
            validated_user_metadata,
            facets=request.facets,
        )
    except Exception as error:
        logger.exception(
            "An unexpected error occurred while adding user data: %s", error
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while adding user data: {error}",
        )
    return {"status": str(status_msg)}


@app.delete(
    "/api/freva-nextgen/databrowser/userdata",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["User data"],
    responses={
        401: {"description": "Unauthorised / not a valid token."},
        500: {"description": "Search backend error."},
    },
    response_class=JSONResponse,
)
async def delete_user_data(
    request: Dict[str, Union[str, int]] = Body(
        ...,
        examples=[
            {
                "project": "user-data",
                "product": "new",
                "institute": "globe",
            }
        ],
    ),
    current_user: TokenPayload = Depends(auth.required_dependency()),
) -> Dict[str, str]:
    """This endpoint lets you delete metadata that has been indexed."""

    solr_instance = Solr(server_config)
    try:
        await solr_instance.delete_user_metadata(
            current_user.preferred_username, request  # type: ignore
        )
    except Exception as error:
        logger.exception("Failed to delete user data: %s", error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete user data: {error}",
        )

    return {
        "status": "User data has been deleted successfully from the databrowser."
    }
