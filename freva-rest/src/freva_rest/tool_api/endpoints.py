"""Define all public endpoints."""

from datetime import datetime
from typing import Annotated, Dict, List, Literal, Optional, Union
from uuid import NAMESPACE_URL, UUID, uuid5

from fastapi import Body, Depends, Path, status
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from pydantic import BaseModel, Field

from freva_rest.auth import TokenPayload, auth
from freva_rest.rest import app

from .internal_endpoints import *  # noqa: F401
from .tool_abstract import ToolAbstract

ParameterItems = Union[str, float, int, datetime, bool, None]


class ToolParameterPayload(BaseModel):
    """This class represents the payload for the tool import parameters."""

    parameters: Annotated[
        Optional[Dict[str, Union[ParameterItems, List[ParameterItems]]]],
        Field(
            title="Parameters",
            description="The parameters you pass to the tool.",
            examples=[
                {
                    "variable": "tas",
                    "product": "EUR-11",
                    "project": "cordex",
                    "time-frequency": "day",
                    "experiment": "historical",
                    "time-period": ["1970", "2000"],
                }
            ],
        ),
    ] = None
    version: Annotated[
        Optional[str],
        Field(
            title="Version",
            description=(
                "The version of the tool, if not given the latest "
                "version will be applied."
            ),
            examples=["v0.0.1"],
        ),
    ] = None


class AddModelStatus(BaseModel):
    """Response model for successful addition of a tool."""

    uuid: Annotated[
        UUID,
        Field(
            title="Tool id.",
            description="Unique identifyer for the tool.",
        ),
    ]
    status_code: Annotated[
        int,
        Field(
            title="Status code",
            description="The status code of the tool submission.",
        ),
    ]

class RemoveModelStatus(BaseModel):
    """Response model for successful removal of a tool from the database."""

    message: Annotated[
        str,
        Field(
            title="Human readable status of tool removal.",
        )
    ]
    status_code: Annotated[
        str,
        Field(
            title="Status code",
            description=(
                "Status code of tool removal" "0: success, 1: could not remove"
            ),
        ),
    ]


class CancelStatus(BaseModel):
    """Response model for canceling jobs."""

    message: Annotated[
        str,
        Field(
            description="Human readable status of the cancelation.",
            examples=[
                "Job with id 095be615-a8ad-4c33-8e9c-c7612fbf6c9f was canceled",
                "Job with id 095be615-a8ad-4c33-8e9c-c7612fbf6c9f has already been cancelled",
            ],
        ),
    ]
    status_code: Annotated[
        int,
        Field(
            description=(
                "Machine readable status of the cancelation"
                "0: success, 1: could not cancel"
            ),
            examples=[0, 1],
        ),
    ]


class SubmitStatus(BaseModel):
    """Response model for job successful job submission."""

    uuid: Annotated[
        UUID,
        Field(
            title="Tool id.",
            description="Unique identifyer for the tool.",
        ),
    ]
    status_code: Annotated[
        int,
        Field(
            title="Status code",
            description="The status code of the job submission.",
        ),
    ]
    hostname: Annotated[
        str,
        Field(
            title="Hostname",
            description="The remote host the job is running on.",
        ),
    ]
    time_submitted: Annotated[
        str,
        Field(
            title="Submit time",
            alias="time-submitted",
            description="The time the job was submitted",
        ),
    ]
    batch_mode: Annotated[
        bool,
        Field(
            title="Batch mode",
            alias="batch-mode",
            description=(
                "Boolean indicating wether the job is a batch mode job."
            ),
        ),
    ]


@app.get(
    "/api/freva-nextgen/tool/overview",
    tags=["Analysis tools"],
    response_model=Dict[str, ToolAbstract],
    status_code=status.HTTP_200_OK,
    responses={
        401: {"description": "Unauthorised / not a valid token."},
        503: {"description": "Service is currently unavailable."},
        status.HTTP_200_OK: {
            "description": ("List of tools that are currently" " available"),
        },
    },
)
async def tool_overview(
    current_user: TokenPayload = Depends(auth.required)
):
    """Get all available tools and their attributes."""
    tool = ToolAbstract(
        name="foo",
        author="bar",
        version="0.0",
        summary="",
        parameters=[],
        added=datetime.now(),
        command="fakecommand",
        binary="fakebinary",
        output_type="data",
    ).model_dump()
    return JSONResponse(status_code=status.HTTP_200_OK, content={"foo": tool})


@app.post(
    "/api/freva-nextgen/tool/add",
    tags=["Analysis tools"],
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"description": "Invalid tool configuration."},
        401: {"description": "Unauthorised / not a valid token."},
        500: {"description": "Adding the tool was unsuccessful."},
        503: {"description": "Service is currently unavailable."},
    },
    response_model=AddModelStatus,
)
async def add_tool(
    tool: Annotated[ToolAbstract, Body(...)],
    current_user: TokenPayload = Depends(auth.required),
):
    """Add a tool to the MongoDB or update it if it already exists."""
    return AddModelStatus(uuid=uuid5(NAMESPACE_URL, "bar"), status_code=202)


@app.delete(
    "/api/freva-nextgen/tool/del/{tool}",
    tags=["Analysis tools"],
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        401: {"description": "Unauthorised / not a valid token."},
        404: {"description": "The requested tool cannot be found."},
        500: {"description": "Removing tool was unsuccessful."},
        503: {"description": "Service is currently unavailable."},
    },
    response_model=RemoveModelStatus,
)
async def delete_tool(
    tool_uuid: Annotated[UUID, Body(...)],
    current_user: TokenPayload = Depends(auth.required),
):
    """Remove a tool and its metadata from the database."""
    return RemoveModelStatus(
        message="Foo bar! Tool was deleted.", status_code="0"
    )


@app.post(
    "/api/freva-nextgen/tool/submit/{tool}",
    tags=["Analysis tools"],
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"description": "Invalid input parameters were given."},
        401: {"description": "Unauthorised / not a valid token."},
        404: {"description": "If the tool is not known to the system."},
        500: {"description": "If the job submission failed."},
        503: {"description": "If the service is currently unavailable."},
    },
    response_model=SubmitStatus,
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
) -> SubmitStatus:
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
    return SubmitStatus(
        uuid=uuid5(NAMESPACE_URL, "foo"),
        status_code=0,
        time_submitted=datetime.now().isoformat(),
        hostname="foo.bar",
        batch_mode=True,
    )


@app.get(
    "/api/freva-nextgen/tool/log/{tool}",
    tags=["Analysis tools"],
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
) -> StreamingResponse:
    """Read the logs from a tool.

    This endpoint streams the logs (stdout & stderr) of a tool.

    Business Logic:
        - Check if job exists -> if not: 404
        - Check if output exists and can be read -> if not: 500
    """
    return StreamingResponse(
        "", status_code=status.HTTP_200_OK, media_type="text/plain"
    )


@app.post(
    "/api/freva-nextgen/tool/cancel/{tool}",
    tags=["Analysis tools"],
    status_code=status.HTTP_200_OK,
    responses={
        401: {"description": "Unauthorised / not a valid token."},
        404: {"description": "The job id is not known to the system."},
        500: {"description": "If the job cancelation failed."},
    },
    response_model=CancelStatus,
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
) -> CancelStatus:
    """Cancel a running job of a given uuid.

    This endpoint cancels a *running* of a given uuid that belongs to a user.

    Business Logic:
        - Check if job exists -> if not: 404
        - Check if job has already bin cacneled -> if so: 200 status: 1
        - Check if job belongs to user: if not: 401
        - Cancel job: if failed: 500
    """
    return CancelStatus(message="foo", status_code=0)
