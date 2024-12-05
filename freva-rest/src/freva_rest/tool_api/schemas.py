"""API schema definitions."""

from datetime import datetime
from typing import Annotated, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field

ParameterItems = Union[str, float, int, datetime, bool, None]


class ToolChangePayload(BaseModel):
    """Define the payload for disabling the visibility of a tool."""

    tool: Annotated[
        str,
        Field(
            title="Tool",
            description="The name of the tool that needs to be changed.",
        ),
    ]
    versions: Annotated[
        Optional[List[str]],
        Field(
            title="Versions",
            description=(
                "List of tool versions that are effected by the "
                "change of visibility, by default all versions are "
                "affected."
            ),
            examples=[["v0.0.1", "v0.0.2"]],
        ),
    ] = None
    visible: Annotated[
        Optional[bool],
        Field(
            description=(
                "If this flag is set to true/false the tool will be made"
                "visible / invisible. You can use this flag to disable tools."
            )
        ),
    ] = None
    make_global: Annotated[
        Optional[bool],
        Field(
            alias="make-global",
            description=(
                "If this flag is set to true you will make this,"
                "tool visible to anyone - ⚠️ You cannot undo this operation."
            ),
        ),
    ] = None


class ToolAddPayload(BaseModel):
    """Define how a tool is added to the system.

    Your tool needs to be stored on a *public* git repository.
    If you want to deploy a development version you can check out any branch
    other than *main*, which is the default branch.
    """

    tool_name: Annotated[
        Optional[str],
        Field(
            title="Tool name",
            description=(
                "If you want to add a tool from the community tool definitions"
                " you can simply give the name of the tool."
            ),
            alias="tool-name",
        ),
    ] = None
    url: Annotated[
        Optional[str],
        Field(
            title="Git URL",
            description=(
                "Use the url of a *public* git repository to add the tool."
            ),
        ),
    ] = None
    branch: Annotated[
        Optional[str],
        Field(
            title="Git branch",
            description=(
                "A specific branch of the tool. This can be useful if you"
                " want to add a development branch for yourself."
            ),
        ),
    ] = None
    tool_path: Annotated[
        Optional[str],
        Field(
            title="Path to tool",
            alias="tool-path",
            description=(
                "This is the path to the parent directroy where the tool is "
                "stored within the git repository. For example tools/animator."
            ),
        ),
    ] = None
    force: Annotated[
        bool,
        Field(
            title="Force",
            description="Force recreation of the environment, admin only.",
        ),
    ] = False


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


class AddToolStatus(BaseModel):
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


class ChangeToolStatus(BaseModel):
    """Response model for successful disabling of a tool from the database."""

    found_items: Annotated[
        int,
        Field(
            title="Found items",
            serialization_alias="found-items",
            description="Number of items that matched the search criteria.",
            examples=[2],
        ),
    ]
    modified_items: Annotated[
        int,
        Field(
            title="Modefied items",
            description="Number of items that have been modified.",
            examples=[1],
            serialization_alias="modified-items",
        ),
    ]

    class Config:
        allow_population_by_field_name = True


class CancelJobStatus(BaseModel):
    """Response model for canceling jobs."""

    message: Annotated[
        str,
        Field(
            description="Human readable status of the cancelation.",
            examples=[
                "Job with id 095be615-a8ad-4c33-8e9c-c7612fbf6c9f was canceled",
                (
                    "Job with id 095be615-a8ad-4c33-8e9c-c7612fbf6c9f has already "
                    "been cancelled"
                ),
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


class SubmitJobStatus(BaseModel):
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
