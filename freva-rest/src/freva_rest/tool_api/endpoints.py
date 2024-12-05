"""Define all public endpoints."""

import asyncio
import os
import shutil
from tempfile import mktemp
from typing import Annotated, Any, Dict, List, Literal, Union
from uuid import UUID

from fastapi import Body, Depends, HTTPException, Path, Query, status
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)

from freva_rest.auth import TokenPayload, auth
from freva_rest.config import ServerConfig
from freva_rest.rest import app
from freva_rest.utils import get_userinfo

from .db import ToolConfig
from .internal_endpoints import tool_communication
from .schemas import (
    CancelJobStatus,
    ChangeToolStatus,
    SubmitJobStatus,
    ToolAddPayload,
    ToolChangePayload,
    ToolParameterPayload,
)
from .utils import Stream, add_tool_on_remote_machine, download_tool

__all__ = [
    "tool_communication",
    "tool_overview",
    "add_tool",
    "change_tool",
    "submit_tool",
    "cancel_job",
]


@app.get(
    "/api/freva-nextgen/tool/overview",
    tags=["Analysis Tools"],
    response_model=Dict[str, List[ToolConfig]],
    status_code=status.HTTP_200_OK,
    responses={
        401: {"description": "Unauthorised / not a valid token."},
        503: {"description": "Service is currently unavailable."},
        200: {"description": "List of tools that are currently available"},
    },
)
async def tool_overview(
    display_all: Annotated[
        bool,
        Query(
            alias="display-all",
            title="Display all",
            description=(
                "Display all tools also the ones that have been " "disabled."
            ),
        ),
    ] = False,
    current_user: TokenPayload = Depends(auth.required),
) -> JSONResponse:
    """Get all available tools and their attributes."""
    cfg = ServerConfig()
    collection = cfg.mongo_client[cfg.mongo_db]["tool_definitions"]
    user_info = get_userinfo(
        {k.lower(): str(v) for (k, v) in dict(current_user).items()}
    )
    search_query: Dict[str, Union[str, Literal[True]]] = {
        "username": user_info["username"]
    }
    if display_all is False:
        search_query["visible"] = True
    # if await collection.find(
    pipeline = [
        # Match either 'admin' or the given username
        {
            "$match": {
                "$or": [
                    search_query,
                    {"username": "admin", "visible": True},
                ]
            }
        },
        # Add a priority field: 1 for admin, 2 for others
        {
            "$addFields": {
                "priority": {"$cond": [{"$eq": ["$username", "admin"]}, 1, 2]}
            }
        },
        # Sort by priority (1 first) and then by name
        {"$sort": {"priority": 1, "name": 1}},
        {"$project": {"_id": 0}},
    ]
    cursor = collection.aggregate(pipeline)
    documents: Dict[str, Dict[str, ToolConfig]] = {}
    results: Dict[str, List[Dict[str, Any]]] = {}
    async for doc in cursor:
        tool = ToolConfig(**doc)
        documents.setdefault(tool.name, {})
        documents[tool.name][tool.version] = tool
    for name, tools in documents.items():
        results[name] = []
        for version in sorted(tools):
            results[name].append(tools[version].model_dump(mode="json"))
    return JSONResponse(status_code=status.HTTP_200_OK, content=results)


@app.post(
    "/api/freva-nextgen/tool/change",
    tags=["Analysis Tools"],
    status_code=status.HTTP_200_OK,
    responses={
        401: {"description": "Unauthorised / not a valid token."},
        422: {"description": "Wrong tool setup."},
        500: {"description": "Changing the tool was unsuccessful."},
    },
    response_model=ChangeToolStatus,
)
async def change_tool(
    properties: Annotated[ToolChangePayload, Body(...)],
    current_user: TokenPayload = Depends(auth.required),
) -> ChangeToolStatus:
    """Change the visibility of a tool or make the admin owner of the tool."""
    cfg = ServerConfig()
    collection = cfg.mongo_client[cfg.mongo_db]["tool_definitions"]
    user_info = get_userinfo(
        {k.lower(): str(v) for (k, v) in dict(current_user).items()}
    )
    if properties.visible is None and properties.make_global is None:
        raise HTTPException(
            status_code=422,
            detail="At least one of 'visible' or 'make-global' must be provided.",
        )

    username = user_info["username"]
    if user_info["username"] in (cfg.admin_users or []):
        username = "admin"
    query: Dict[str, Any] = {"name": properties.tool, "username": username}
    if properties.versions:
        query["version"] = {"$in": properties.versions}
    update_doc: Dict[str, Dict[str, Any]] = {"$set": {}}
    if properties.visible is not None:
        update_doc["$set"]["visible"] = properties.visible
    elif properties.make_global is not None:
        update_doc["$set"]["username"] = "admin"
    result = await collection.update_many(query, update_doc)
    return ChangeToolStatus(
        found_items=result.matched_count,
        modified_items=result.modified_count,
    )


@app.post(
    "/api/freva-nextgen/tool/add",
    tags=["Analysis Tools"],
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Invalid tool configuration."},
        401: {"description": "Unauthorised / not a valid token."},
        500: {"description": "Adding the tool was unsuccessful."},
        503: {"description": "Service is currently unavailable."},
    },
    response_class=PlainTextResponse,
)
async def add_tool(
    properties: Annotated[
        ToolAddPayload, Body(..., description=ToolAddPayload.__doc__)
    ],
    current_user: TokenPayload = Depends(auth.required),
) -> StreamingResponse:
    """Add a tool to the MongoDB or update it if it already exists."""

    cfg = ServerConfig()
    user_info = get_userinfo(
        {k.lower(): str(v) for (k, v) in dict(current_user).items()}
    )
    extra_args = []
    force = False
    if user_info["username"] in (cfg.admin_users or []):
        username = "admin"
        force = properties.force
        if properties.force:
            extra_args.append("--force")
    else:
        username = user_info["username"]

    public_url = "https://github.com/FREVA-CLINT/data-analysis-tools.git"
    git_url = properties.url or public_url
    branch = properties.branch or "add-example-tool"
    if not properties.tool_path and not properties.tool_name:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "At least a tool name or a path to the tool "
                "definition in the repository must be "
                "provided."
            ),
        )
    tool_path = properties.tool_path or f"tools/{properties.tool_name}"
    temp_dir = mktemp()
    stream = Stream()
    try:
        tool_model = await download_tool(
            temp_dir, git_url, tool_path, branch=branch, username=username
        )
    except HTTPException:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise
    if force is False:
        await tool_model.check_for_version()
    asyncio.create_task(
        add_tool_on_remote_machine(
            tool_model, stream, temp_dir, tool_path, *extra_args
        )
    )

    return StreamingResponse(
        stream.stream_content(),
        status_code=status.HTTP_200_OK,
        media_type="text/plain",
    )


@app.post(
    "/api/freva-nextgen/tool/submit/{tool}",
    tags=["Analysis Tools"],
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"description": "Invalid input parameters were given."},
        401: {"description": "Unauthorised / not a valid token."},
        404: {"description": "If the tool is not known to the system."},
        500: {"description": "If the job submission failed."},
        503: {"description": "If the service is currently unavailable."},
        202: {"description": "The job has been successfully submitted."},
    },
    response_model=SubmitJobStatus,
)
async def submit_tool(
    tool: Annotated[
        str,
        Path(
            title="Tool",
            description=(
                "The name of the data analysis tool that should be " "applied."
            ),
            examples=["animator"],
        ),
    ],
    payload: Annotated[ToolParameterPayload, Body(...)],
    current_user: TokenPayload = Depends(auth.required),
) -> None:
    """Submit an existing tool in via the workload manager.

    This endpoint should be submitting a tool that is knwon to the system.

    Business Logic:
        - Check if tool exists -> if not: 404.
        - Parse given input parameters -> if failed: 400.
        - Connect to remote machine. -> if failed: 503
        - Create input script on remote machine.
        - Submit script. -> if failed: 500.
        - Exit 201
    """


@app.get(
    "/api/freva-nextgen/tool/log/{uuid}",
    tags=["Analysis Tools"],
    status_code=status.HTTP_200_OK,
    responses={
        401: {"description": "Unauthorised / not a valid token."},
        404: {"description": "The job id is not known to the system."},
        500: {"description": "If reading the logs failed."},
    },
    response_class=PlainTextResponse,
)
async def job_log(
    uuid: Annotated[
        UUID,
        Path(
            title="Tool id.",
            description="Unique identifyer for the tool.",
            examples=["095be615-a8ad-4c33-8e9c-c7612fbf6c9f"],
        ),
    ],
    current_user: TokenPayload = Depends(auth.required),
) -> None:
    """Read the logs from a tool.

    This endpoint streams the logs (stdout & stderr) of a tool.

    Business Logic:
        - Check if job exists -> if not: 404
        - Check if output exists and can be read -> if not: 500
    """


@app.post(
    "/api/freva-nextgen/tool/cancel/{uuid}",
    tags=["Analysis Tools"],
    status_code=status.HTTP_200_OK,
    responses={
        401: {"description": "Unauthorised / not a valid token."},
        404: {"description": "The job id is not known to the system."},
        500: {"description": "If the job cancelation failed."},
    },
    response_model=CancelJobStatus,
)
async def cancel_job(
    uuid: Annotated[
        UUID,
        Path(
            title="Tool id.",
            description="Unique identifyer for the tool.",
            examples=["095be615-a8ad-4c33-8e9c-c7612fbf6c9f"],
        ),
    ],
    current_user: TokenPayload = Depends(auth.required),
) -> None:
    """Cancel a running job of a given uuid.

    This endpoint cancels a *running* of a given uuid that belongs to a user.

    Business Logic:
        - Check if job exists -> if not: 404
        - Check if job has already bin cacneled -> if so: 200 status: 1
        - Check if job belongs to user: if not: 401
        - Cancel job: if failed: 500
    """
