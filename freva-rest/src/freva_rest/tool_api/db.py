"""Definitions of Tool parameters and types."""

import enum
from datetime import datetime
from pathlib import Path
from typing import Annotated, Dict, List, Literal, Optional, Union

from fastapi import HTTPException, status
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ConfigDict, Field
from pydantic.functional_validators import field_validator, model_validator

from freva_rest.config import ServerConfig

BaseParamType = Union[
    str,
    int,
    float,
    bool,
    datetime,
    Path,
    List[str],
    List[int],
    List[float],
    List[bool],
    List[datetime],
    List[Path],
]


@enum.unique
class ToolState(enum.IntEnum):
    """Define the states a tool could be in."""

    UNKOWN = -1  # The tool seems to be lost.
    SUBMITTED = 0
    IN_PROGRESS = 1
    FINISHED = 2
    CANCEL = 3  # The tool should will be cancled
    CANCELED = 4
    FAILED = 5

    @classmethod
    def get_status_from_value(cls, value: int) -> "ToolState":
        """Get the status name from status value."""
        for state in cls:
            if value == state.value:
                return state
        raise ValueError(f"Unknown ToolState value: {value}")


class BaseParam(BaseModel):
    """
    Base class for all parameter types. Inherits from pydantic.BaseModel.

    All other available parameters inherit from this class.
    A parameter object created by this class holds the following user-provided
    information. Values taken by the parameter can be any of type
    BaseParamType, which include str, int, float, bool, datetime, Sequence.
    This base class does not allow for any user-defined attributes, other than
    those described below.

    By being based on the pydantic.BaseModel, parameter values are
    automatically validated by the framework. Conversions are performed
    according to the conversion table found under
    https://docs.pydantic.dev/latest/concepts/conversion_table .

    Parameters
    ----------
    name (str):
        Name of the parameter. Must be at least on character.
    title (str):
        Title under which the parameter will appear. If none is given,
        defaults to name.
    help (str, optional):
        Description of the parameter's function. Defaults to None.
    mandatory(bool, optional):
        Boolean value determining if setting this parameter is required.
        Defaults to False.
    default(BaseParamType, optional):
        A default value for the parameter, in case non is set by the user.
        Defaults to None.
    type(Literal["BaseParam"]):
        A description of the parameter's type. Defaults to "BaseParam".

    Attributes
    ----------
    name (str):
        Name of the parameter.
    help (str):
        Description of the parameter's function.
    mandatory(bool):
        Boolean value determining if setting this parameter is required.
    default(BaseParamType):
        A default value for the parameter, in case non is set by the user.
    type(Literal["BaseParam"]):
        A description of the parameter's type.
    """

    model_config = ConfigDict(
        extra="forbid"
    )  # forbid extra fields entered by a user

    name: Annotated[
        str, Field(min_length=1, description="Name of the parameter.")
    ]
    title: Annotated[Optional[str], Field(description="Title of parameter.")] = (
        None
    )
    help: Annotated[
        str, Field(description="Description of parameter's function.")
    ] = ""
    mandatory: Annotated[
        bool,
        Field(
            description=(
                "Boolean value determining if parameter value has to be set."
            ),
        ),
    ] = False
    default: Annotated[
        Optional[Union[BaseParamType, List[BaseParamType]]],
        Field(mandatory=False, description="Default value of parameter."),
    ] = None

    @model_validator(mode="after")
    def validate_intput(self) -> "BaseParam":
        """Assign the title."""
        if self.title is None:
            self.title = self.name
        return self


class Bool(BaseParam):
    """
    A simple boolean parameter.

    Parameters
    ----------
    default (bool, optional):
        The default value of the parameter. Defaults to None.
    type (Literal["Bool"]):
        A description of the parameter's type. Must be "Bool".

    Attributes
    ----------
    default (bool, optional):
        The default value of the parameter.
    type (Literal["Bool"]):
        A description of the parameter's type.
    """

    type: Literal["bool"]
    default: Optional[Union[bool, List[bool]]] = None


class Integer(BaseParam):
    """
    A simple integer parameter.
    Parameter values can be any that

    Parameters
    ----------
    default (int, optional):
        The default value of the parameter. Defaults to None.
    type (Literal["Integer"]):
        A description of the parameter's type. Must be "Integer".

    Attributes
    ----------
    default (int):
        The default value of the parameter.
    type (Literal["Integer"]):
        A description of the parameter's type.
    """

    type: Literal["integer"]
    default: Optional[Union[int, List[int]]] = None


class Float(BaseParam):
    """
    A simple parameter for floating-point numbers.

    Parameters
    ----------
    default (float, optional):
        The default value of the parameter. Defaults to None.
    type (Literal["Float"]):
        A description of the parameter's type. Must be "Float".

    Attributes
    ----------
    default (float):
        The default value of the parameter.
    type (Literal["Float"]):
        A description of the parameter's type.
    """

    type: Literal["float"]
    default: Optional[Union[float, List[float]]] = None


class String(BaseParam):
    """
    A simple string parameter.

    Parameters
    ----------
    default (str, optional):
        The default value of the parameter. Defaults to None.
    type (Literal["String"]):
        A description of the parameter's type. Must be "String".

    Attributes
    ----------
    default (str):
        The default value of the parameter.
    type (Literal["String"]):
        A description of the parameter's type.
    """

    type: Literal["string"]
    default: Optional[Union[str, List[str]]] = None


class Date(BaseParam):
    """
    A simple datetime parameter.

    Parameters
    ----------
    default (datetime, optional)
        The default value of the parameter. Defaults to None.
    type (Literal["Date"])
        A description of the parameter's type. Must be "Date".

    Attributes
    ----------
    default (datetime):
        The default value of the parameter.
    type (Literal["Date"]):
        A description of the parameter's type.
    """

    type: Literal["datetime"]
    default: Optional[Union[datetime, List[datetime]]] = None


class DataField(BaseParam):
    """
    A parameter for selecting valid values using the databrowser.

    Parameters
    ----------
    search_key (str):
        The databrowser facet used for this parameter.
    group (int, optional):
        The group this search facet belongs to. This can be used to group
        different search facets together, for example for comparing
        multi model ensembles. Default is 1.
    multiple(bool, optional):
        Flag indicating whether multiple values can be selected for the
        selected facet. Default is False.
    constraints(Dict[str, Union[str, Sequence[str]]], optional):
        A dict containing default values for other search facets.
        Defaults to None.
    default (str, optional):
        The default value of the parameter. Defaults to None.
    type (Literal["DataField"]):
        A description of the parameter's type. Must be "DataField".

    Attributes
    ----------
    search_key (str):
        The databrowser facet used for this parameter.
    group (int, optional):
        The group this search facet belongs to. This can be used to group
        different search facets together, for example for comparing
        multi model ensembles.
    multiple(bool, optional):
        Flag indicating whether multiple values can be selected for the
        selected facet.
    constraints(Dict[str, Union[str, List[str]]], optional):
        A dict containing default values for other search facets.
    default (str, optional):
        The default value of the parameter.
    type (Literal["DataField"]):
        A description of the parameter's type.

    Example
    -------

    .. code-block:: python

        from input_parameters import DataField

        datafield = DataField(
                        name="variable_name",
                        default="tas",
                        facet="variable",
                        max_items=1,
                        group=2,
                        predefined_facet={"time_frequency":["1hr"]},
                        help="Select the variable name"
        )
    """

    type: Literal["databrowser"]
    default: Optional[Union[str, List[str]]] = None
    search_key: Annotated[Optional[str], Field()] = None
    constraint: Annotated[Optional[Dict[str, Union[str, List[str]]]], Field()] = (
        None
    )
    search_result: Annotated[Optional[str], Field()] = None


class Range(BaseParam):
    """
    A simple Range parameter for iterable, indexable sequences such as lists and tuples.

    Parameters
    ----------
    default (List[BaseParamType], optional):
        The default value of the parameter. Defaults to None.
    type (Literal["Range"]):
        A description of the parameter's type. Must be "Range".

    Attributes
    ----------
    default (List[BaseParamType]):
        The default value of the parameter.
    type (Literal["Range"]):
        A description of the parameter's type.
    """

    type: Literal["range"]
    default: Optional[List[BaseParamType]] = None


class File(BaseParam):
    """
    A simple parameter to define a file stored on the system.

    Parameters
    ----------
    default (Path, optional):
        The default value of the parameter. Defaults to None.
    type (Literal["File"]):
        A description of the parameter's type. Must be "File".

    Attributes
    ----------
    default (Path):
        The default value of the parameter.
    type (Literal["File"]):
        A description of the parameter's type.
    """

    type: Literal["path"]
    default: Optional[Union[Path, str, List[Path], List[str]]] = None


ParameterType = Annotated[
    Union[Bool, String, Float, Integer, Date, DataField, Range, File, Range],
    Field(discriminator="type"),
]


class ToolConfig(BaseModel):
    """Define the database model for the tool configuration."""

    # basic tool metadata
    name: Annotated[str, Field(description="Name of the tool", min_length=1)]
    authors: Annotated[
        List[str], Field(description="Tool Author(s)", min_length=1)
    ]
    version: Annotated[str, Field(description="Tool version")]
    summary: Annotated[
        str,
        Field(
            min_length=1,
            description="Short description of what the tool is supposed to do.",
        ),
    ]
    command: Annotated[
        str, Field(min_length=1, description="The command that is executed.")
    ]
    username: Annotated[
        str, Field(min_length=1, description="The user who add's the tool.")
    ]
    conda_env: Annotated[
        Optional[str],
        Field(description="Binary used to execute tool command."),
    ] = None
    added: Annotated[datetime, Field(description="When the tool was added.")]
    description: Annotated[
        Optional[str],
        Field(
            description="A more detailed description of the tool",
        ),
    ] = None
    title: Annotated[
        Optional[str],
        Field(description="An optional title of the tool."),
    ] = None
    visible: Annotated[
        bool, Field(description="If this particular version is to be displayed.")
    ] = True
    # input parameters
    parameters: Annotated[
        Optional[List[ParameterType]],
        Field(description="The input parameters of the tool."),
    ] = None
    # tool command and binary used to execute tool

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        """
        Validate that the version string conforms to PEP 440.
        """
        try:
            return str(Version(value))  # Validate the version
        except InvalidVersion:
            raise ValueError(
                f"Invalid version: '{value}'. "
                "Ensure it conforms to PEP 440 (e.g., 1.0.0, 1.0.0b1)."
            ) from None

    async def dump_to_db(self) -> None:
        """Add this specific tool to the tools db."""
        cfg = ServerConfig()
        collection = cfg.mongo_client[cfg.mongo_db]["tool_definitions"]
        await collection.create_index(
            [("name", 1), ("version", 1), ("username", 1)], unique=True
        )
        await collection.replace_one(
            {
                "name": self.name,
                "version": self.version,
                "username": self.username,
                "visible": self.visible,
            },
            self.dict(by_alias=True),
            upsert=True,
        )

    async def check_for_version(self) -> None:
        """Check if a tool of the given version already exists in the DB."""
        cfg = ServerConfig()
        collection = cfg.mongo_client[cfg.mongo_db]["tool_definitions"]
        query = {
            "$and": [
                {"name": self.name},
                {"version": self.version},
                {"$or": [{"username": self.username}, {"username": "admin"}]},
            ]
        }
        if await collection.find_one(query):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Version {self.version} has already been added, "
                    "please bump the version of your tool first."
                ),
            )
