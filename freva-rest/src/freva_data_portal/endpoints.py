"""Definition of endpoints for loading/streaming and manipulating data."""

import asyncio
import json
from typing import Annotated, Any, Dict, Optional

import cloudpickle
from fastapi import Depends, Path, Query, status
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse, Response
from freva_rest.auth import TokenPayload, auth
from freva_rest.rest import app
from freva_rest.utils import create_redis_connection
from zarr.storage import array_meta_key, attrs_key, group_meta_key


async def read_redis_data(
    key: str,
    subkey: Optional[str] = None,
    timeout: int = 1,
) -> Any:
    """Read the cache data given by a key.

    Parameters
    ----------
    key: str
        The key under which the data is stored.
    subkey: str|None
        If the data under key is a pickled dict the the will be
        unpickled and and the value of that subkey will be returned
    timeout: int
        Wait for timeout seconds until a not found error is risen.
    """

    cache = await create_redis_connection()
    data: Optional[bytes] = await cache.get(key)
    npolls = 0
    while data is None:
        npolls += 1
        await asyncio.sleep(1)
        data = await cache.get(key)
        if npolls >= timeout:
            break
    if data is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=f"{key} uuid does not exist (anymore).",
        )
    if not subkey:
        return data
    lookup = {
        0: "finished, ok",
        1: "finished, failed",
        2: "waiting",
        3: "processing",
    }
    p_data: Dict[str, Any] = cloudpickle.loads(data)
    task_status = p_data.get("status", 1)
    if task_status != 0:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=lookup.get(task_status, "unknown"),
        )
    return p_data[subkey]


@app.get(
    "/api/freva-data-portal/zarr/{uuid5}.zarr/status",
    tags=["Load data"],
)
async def get_status(
    uuid5: Annotated[
        str,
        Path(
            title="uuid",
            description=(
                (
                    "The uuid that was generated, when task to stream data was "
                    "created."
                )
            ),
        ),
    ],
    timeout: Annotated[
        int,
        Query(
            alias="timeout",
            title="Cache timeout for getting results.",
            description="Set a timeout to wait for results.",
            ge=0,
            le=1500,
        ),
    ] = 1,
    current_user: TokenPayload = Depends(auth.required),
) -> JSONResponse:
    """Get the status of a loading process."""
    meta: Dict[str, Any] = await read_redis_data(uuid5, "status", timeout=timeout)
    return JSONResponse(content={"status": meta}, status_code=status.HTTP_200_OK)


@app.get(
    "/api/freva-data-portal/zarr/{uuid5}.zarr/.zmetadata",
    tags=["Load data"],
)
async def zemtadata(
    uuid5: Annotated[
        str,
        Path(
            title="uuid",
            description=(
                (
                    "The uuid that was generated, when task to stream data was "
                    "created."
                )
            ),
        ),
    ],
    timeout: Annotated[
        int,
        Query(
            alias="timeout",
            title="Cache timeout for getting results.",
            description="Set a timeout to wait for results.",
            ge=0,
            le=1500,
        ),
    ] = 1,
    current_user: TokenPayload = Depends(auth.required),
) -> JSONResponse:
    """Consolidate zarr metadata

    This endpoint returns the metadata about the structure and organization of
    data within the particular zarr store in question.
    """

    meta: Dict[str, Any] = await read_redis_data(uuid5, "json_meta", timeout=timeout)
    return JSONResponse(
        content=meta,
        status_code=status.HTTP_200_OK,
    )


@app.get(
    "/api/freva-data-portal/zarr/{uuid5}.zarr/.zgroup",
    tags=["Load data"],
    status_code=status.HTTP_200_OK,
)
async def zgroup(
    uuid5: Annotated[
        str,
        Path(
            title="uuid",
            description=(
                (
                    "The uuid that was generated, when task to stream data was "
                    "created."
                )
            ),
        ),
    ],
    timeout: Annotated[
        int,
        Query(
            alias="timeout",
            title="Cache timeout for getting results.",
            description="Set a timeout to wait for results.",
            ge=0,
            le=1500,
        ),
    ] = 1,
    current_user: TokenPayload = Depends(auth.required),
) -> JSONResponse:
    """Zarr group data.

    This `.zarrgroup` metadata includes information about the arrays and
    subgroups contained within the group, as well as their attributes such as
    data types, shapes, and chunk sizes. The `.zarrgroup` endpoint helps in
    organizing and managing the structure of data within a Zarr group,
    allowing users to access and manipulate arrays and subgroups efficiently.
    """
    meta: Dict[str, Any] = await read_redis_data(uuid5, "json_meta", timeout=timeout)
    return JSONResponse(
        content=meta["metadata"][".zgroup"],
        status_code=status.HTTP_200_OK,
    )


@app.get(
    "/api/freva-data-portal/zarr/{uuid5}.zarr/.zattrs",
    tags=["Load data"],
)
async def zattrs(
    uuid5: Annotated[
        str,
        Path(
            title="uuid",
            description=(
                (
                    "The uuid that was generated, when task to stream data was "
                    "created."
                )
            ),
        ),
    ],
    timeout: Annotated[
        int,
        Query(
            alias="timeout",
            title="Cache timeout for getting results.",
            description="Set a timeout to wait for results.",
            ge=0,
            le=1500,
        ),
    ] = 1,
    current_user: TokenPayload = Depends(auth.required),
) -> JSONResponse:
    """Get zarr Attributes.

    Get metadata attributes associated with the dataset or arrays within the
    dataset. These attributes provide additional information about the dataset
    or arrays, such as descriptions, units, creation dates, or any other
    custom metadata relevant to the data.
    """
    meta: Dict[str, Any] = await read_redis_data(uuid5, "json_meta", timeout=timeout)
    return JSONResponse(
        content=meta["metadata"][".zattrs"], status_code=status.HTTP_200_OK
    )


@app.get(
    "/api/freva-data-portal/zarr/{uuid5}.zarr/{variable}/{chunk}",
    tags=["Load data"],
)
async def chunk_data(
    uuid5: Annotated[
        str,
        Path(
            title="uuid",
            description=(
                (
                    "The uuid that was generated, when task to stream data was "
                    "created."
                )
            ),
        ),
    ],
    variable: Annotated[
        str,
        Path(
            title="variable",
            description=("The variable name that should be read."),
        ),
    ],
    chunk: Annotated[
        str,
        Path(title="chunk", description="The chnuk number that should be read."),
    ],
    timeout: Annotated[
        int,
        Query(
            alias="timeout",
            title="Cache timeout for getting results.",
            description="Set a timeout to wait for results.",
            ge=0,
            le=1500,
        ),
    ] = 1,
    current_user: TokenPayload = Depends(auth.required),
) -> Response:
    """Get a zarr array chunk.

    This method reads the zarr data."""

    if array_meta_key in chunk or attrs_key in chunk:
        json_meta: Dict[str, Any] = await read_redis_data(
            uuid5, "json_meta", timeout=timeout
        )
        if attrs_key in chunk:
            key = f"{variable}/{attrs_key}"
        else:
            key = f"{variable}/{array_meta_key}"
        try:
            content = json_meta["metadata"][key]
        except KeyError as error:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(error))
        return JSONResponse(
            content=content,
            status_code=status.HTTP_200_OK,
        )
    if group_meta_key in chunk:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Sub groups are not supported.",
        )
    chunk_key = f"{uuid5}-{variable}-{chunk}"
    detail = {"chunk": {"uuid": uuid5, "variable": variable, "chunk": chunk}}
    cache = await create_redis_connection()
    await cache.publish("data-portal", json.dumps(detail).encode("utf-8"))
    data: bytes = await read_redis_data(chunk_key, timeout=timeout)
    return Response(data, media_type="application/octet-stream")
