"""Command line interface for the freva databrowser.

Search quickly and intuitively for many different climate datasets.
"""

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Literal, Optional, Union, cast

import typer
from freva_client import databrowser
from freva_client.auth import Auth, Token
from freva_client.utils import exception_handler, logger

from .cli_utils import parse_cli_args, version_callback


def _auth(url: str, token: Optional[str]) -> None:
    if token:
        auth = Auth()
        exp = auth.token_expiration_time(url, token)
        now = datetime.now().timestamp()
        auth.auth_token = cast(
            Token,
            {
                "access_token": token or "",
                "refresh_token": "",
                "refresh_expires_in": 20,
                "expires_in": exp - now,
                "refresh_expires": now + 20,
                "expires": exp,
            },
        )


class UniqKeys(str, Enum):
    """Literal implementation for the cli."""

    file: str = "file"
    uri: str = "uri"


class Flavours(str, Enum):
    """Literal implementation for the cli."""

    freva: str = "freva"
    cmip6: str = "cmip6"
    cmip5: str = "cmip5"
    cordex: str = "cordex"
    nextgems: str = "nextgems"


class TimeSelect(str, Enum):
    """Literal implementation for the cli."""

    strict: str = "strict"
    flexible: str = "flexible"
    file: str = "file"

    @staticmethod
    def get_help() -> str:
        """Generate the help string."""
        return (
            "Operator that specifies how the time period is selected. "
            "Choose from flexible (default), strict or file. "
            "``strict`` returns only those files that have the *entire* "
            "time period covered. The time search ``2000 to 2012`` will "
            "not select files containing data from 2010 to 2020 with "
            "the ``strict`` method. ``flexible`` will select those files "
            "as  ``flexible`` returns those files that have either start "
            "or end period covered. ``file`` will only return files where "
            "the entire time period is contained within *one single* file."
        )


databrowser_app = typer.Typer(
    help="Data search related commands", callback=logger.set_cli
)


@databrowser_app.command(
    name="data-overview",
    help="Get an overview over what is available in the databrowser.",
)
@exception_handler
def overview(
    host: Optional[str] = typer.Option(
        None,
        "--host",
        help=(
            "Set the hostname of the databrowser, if not set (default) "
            "the hostname is read from a config file"
        ),
    ),
) -> None:
    """Get a general overview of the databrowser's search capabilities."""
    print(databrowser.overview(host=host))


@databrowser_app.command(
    name="metadata-search", help="Search databrowser for metadata (facets)."
)
@exception_handler
def metadata_search(
    search_keys: Optional[List[str]] = typer.Argument(
        default=None,
        help="Refine your data search with this `key=value` pair search "
        "parameters. The parameters could be, depending on the DRS standard, "
        "flavour product, project model etc.",
    ),
    facets: Optional[List[str]] = typer.Option(
        None,
        "--facet",
        help=(
            "If you are not sure about the correct search key's you can use"
            " the ``--facet`` flag to search of any matching entries. For "
            "example --facet 'era5' would allow you to search for any entries"
            " containing era5, regardless of project, product etc."
        ),
    ),
    flavour: Flavours = typer.Option(
        "freva",
        "--flavour",
        "-f",
        help=(
            "The Data Reference Syntax (DRS) standard specifying the type "
            "of climate datasets to query."
        ),
    ),
    time_select: TimeSelect = typer.Option(
        "flexible",
        "-ts",
        "--time-select",
        help=TimeSelect.get_help(),
    ),
    time: Optional[str] = typer.Option(
        None,
        "-t",
        "--time",
        help=(
            "Special search facet to refine/subset search results by time. "
            "This can be a string representation of a time range or a single "
            "time step. The time steps have to follow ISO-8601. Valid strings "
            "are ``%Y-%m-%dT%H:%M`` to ``%Y-%m-%dT%H:%M`` for time ranges and "
            "``%Y-%m-%dT%H:%M``. **Note**: You don't have to give the full "
            "string format to subset time steps ``%Y``, ``%Y-%m`` etc are also"
            " valid."
        ),
    ),
    extended_search: bool = typer.Option(
        False,
        "-e",
        "--extended-search",
        help="Retrieve information on additional search keys.",
    ),
    multiversion: bool = typer.Option(
        False,
        "--multi-version",
        help="Select all versions and not just the latest version (default).",
    ),
    host: Optional[str] = typer.Option(
        None,
        "--host",
        help=(
            "Set the hostname of the databrowser, if not set (default) "
            "the hostname is read from a config file"
        ),
    ),
    parse_json: bool = typer.Option(
        False, "-j", "--json", help="Parse output in json format."
    ),
    verbose: int = typer.Option(0, "-v", help="Increase verbosity", count=True),
    version: Optional[bool] = typer.Option(
        False,
        "-V",
        "--version",
        help="Show version an exit",
        callback=version_callback,
    ),
) -> None:
    """Search metadata (facets) based on the specified Data Reference Syntax
    (DRS) standard (flavour) and the type of search result (uniq_key), which
    can be either file or uri. Facets represent the metadata categories
    associated with the climate datasets, such as experiment, model,
    institute, and more. This method provides a comprehensive view of the
    available facets and their corresponding counts based on the provided
    search criteria.
    """
    logger.set_verbosity(verbose)
    logger.debug("Search the databrowser")
    result = databrowser.metadata_search(
        *(facets or []),
        time=time or "",
        time_select=cast(Literal["file", "flexible", "strict"], time_select.value),
        flavour=cast(
            Literal["freva", "cmip6", "cmip5", "cordex", "nextgems"],
            flavour.value,
        ),
        host=host,
        extended_search=extended_search,
        multiversion=multiversion,
        fail_on_error=False,
        **(parse_cli_args(search_keys or [])),
    )
    if parse_json:
        print(json.dumps(result))
        return
    for key, values in result.items():
        print(f"{key}: {', '.join(values)}")


@databrowser_app.command(
    name="data-search", help="Search the databrowser for datasets."
)
@exception_handler
def data_search(
    search_keys: Optional[List[str]] = typer.Argument(
        default=None,
        help="Refine your data search with this `key=value` pair search "
        "parameters. The parameters could be, depending on the DRS standard, "
        "flavour product, project model etc.",
    ),
    facets: Optional[List[str]] = typer.Option(
        None,
        "--facet",
        help=(
            "If you are not sure about the correct search key's you can use"
            " the ``--facet`` flag to search of any matching entries. For "
            "example --facet 'era5' would allow you to search for any entries"
            " containing era5, regardless of project, product etc."
        ),
    ),
    uniq_key: UniqKeys = typer.Option(
        "file",
        "--uniq-key",
        "-u",
        help=(
            "The type of search result, which can be either “file” "
            "or “uri”. This parameter determines whether the search will be "
            "based on file paths or Uniform Resource Identifiers"
        ),
    ),
    flavour: Flavours = typer.Option(
        "freva",
        "--flavour",
        "-f",
        help=(
            "The Data Reference Syntax (DRS) standard specifying the type "
            "of climate datasets to query."
        ),
    ),
    time_select: TimeSelect = typer.Option(
        "flexible",
        "-ts",
        "--time-select",
        help=TimeSelect.get_help(),
    ),
    zarr: bool = typer.Option(False, "--zarr", help="Create zarr stream files."),
    access_token: Optional[str] = typer.Option(
        None,
        "--access-token",
        help=(
            "Use this access token for authentication"
            " when creating a zarr stream files."
        ),
    ),
    time: Optional[str] = typer.Option(
        None,
        "-t",
        "--time",
        help=(
            "Special search facet to refine/subset search results by time. "
            "This can be a string representation of a time range or a single "
            "time step. The time steps have to follow ISO-8601. Valid strings "
            "are ``%Y-%m-%dT%H:%M`` to ``%Y-%m-%dT%H:%M`` for time ranges and "
            "``%Y-%m-%dT%H:%M``. **Note**: You don't have to give the full "
            "string format to subset time steps ``%Y``, ``%Y-%m`` etc are also"
            " valid."
        ),
    ),
    parse_json: bool = typer.Option(
        False, "-j", "--json", help="Parse output in json format."
    ),
    host: Optional[str] = typer.Option(
        None,
        "--host",
        help=(
            "Set the hostname of the databrowser, if not set (default) "
            "the hostname is read from a config file"
        ),
    ),
    verbose: int = typer.Option(0, "-v", help="Increase verbosity", count=True),
    multiversion: bool = typer.Option(
        False,
        "--multi-version",
        help="Select all versions and not just the latest version (default).",
    ),
    version: Optional[bool] = typer.Option(
        False,
        "-V",
        "--version",
        help="Show verion an exit",
        callback=version_callback,
    ),
) -> None:
    """Search for climate datasets based on the specified Data Reference Syntax
    (DRS) standard (flavour) and the type of search result (uniq_key), which
    can be either “file” or “uri”. The databrowser method provides a flexible
    and efficient way to query datasets matching specific search criteria and
    retrieve a list of data files or locations that meet the query parameters.
    """
    logger.set_verbosity(verbose)
    logger.debug("Search the databrowser")
    result = databrowser(
        *(facets or []),
        time=time or "",
        time_select=cast(Literal["file", "flexible", "strict"], time_select),
        flavour=cast(
            Literal["freva", "cmip6", "cmip5", "cordex", "nextgems"],
            flavour.value,
        ),
        uniq_key=cast(Literal["uri", "file"], uniq_key.value),
        host=host,
        fail_on_error=False,
        multiversion=multiversion,
        stream_zarr=zarr,
        **(parse_cli_args(search_keys or [])),
    )
    if zarr:
        _auth(result._cfg.auth_url, access_token)
    if parse_json:
        print(json.dumps(sorted(result)))
    else:
        for res in result:
            print(res)


@databrowser_app.command(
    name="intake-catalogue", help="Create an intake catalogue from the search."
)
@exception_handler
def intake_catalogue(
    search_keys: Optional[List[str]] = typer.Argument(
        default=None,
        help="Refine your data search with this `key=value` pair search "
        "parameters. The parameters could be, depending on the DRS standard, "
        "flavour product, project model etc.",
    ),
    facets: Optional[List[str]] = typer.Option(
        None,
        "--facet",
        help=(
            "If you are not sure about the correct search key's you can use"
            " the ``--facet`` flag to search of any matching entries. For "
            "example --facet 'era5' would allow you to search for any entries"
            " containing era5, regardless of project, product etc."
        ),
    ),
    uniq_key: UniqKeys = typer.Option(
        "file",
        "--uniq-key",
        "-u",
        help=(
            "The type of search result, which can be either “file” "
            "or “uri”. This parameter determines whether the search will be "
            "based on file paths or Uniform Resource Identifiers"
        ),
    ),
    flavour: Flavours = typer.Option(
        "freva",
        "--flavour",
        "-f",
        help=(
            "The Data Reference Syntax (DRS) standard specifying the type "
            "of climate datasets to query."
        ),
    ),
    time_select: TimeSelect = typer.Option(
        "flexible",
        "-ts",
        "--time-select",
        help=TimeSelect.get_help(),
    ),
    time: Optional[str] = typer.Option(
        None,
        "-t",
        "--time",
        help=(
            "Special search facet to refine/subset search results by time. "
            "This can be a string representation of a time range or a single "
            "time step. The time steps have to follow ISO-8601. Valid strings "
            "are ``%Y-%m-%dT%H:%M`` to ``%Y-%m-%dT%H:%M`` for time ranges and "
            "``%Y-%m-%dT%H:%M``. **Note**: You don't have to give the full "
            "string format to subset time steps ``%Y``, ``%Y-%m`` etc are also"
            " valid."
        ),
    ),
    zarr: bool = typer.Option(
        False, "--zarr", help="Create zarr stream files, as catalogue targets."
    ),
    access_token: Optional[str] = typer.Option(
        None,
        "--access-token",
        help=(
            "Use this access token for authentication"
            " when creating a zarr based intake catalogue."
        ),
    ),
    filename: Optional[Path] = typer.Option(
        None,
        "-f",
        "--filename",
        help=(
            "Path to the file where the catalogue, should be written to. "
            "if None given (default) the catalogue is parsed to stdout."
        ),
    ),
    host: Optional[str] = typer.Option(
        None,
        "--host",
        help=(
            "Set the hostname of the databrowser, if not set (default) "
            "the hostname is read from a config file"
        ),
    ),
    verbose: int = typer.Option(0, "-v", help="Increase verbosity", count=True),
    multiversion: bool = typer.Option(
        False,
        "--multi-version",
        help="Select all versions and not just the latest version (default).",
    ),
    version: Optional[bool] = typer.Option(
        False,
        "-V",
        "--version",
        help="Show verion an exit",
        callback=version_callback,
    ),
) -> None:
    """Create an intake catalogue for climate datasets based on the specified "
    "Data Reference Syntax (DRS) standard (flavour) and the type of search "
    result (uniq_key), which can be either “file” or “uri”."""
    logger.set_verbosity(verbose)
    logger.debug("Search the databrowser")
    result = databrowser(
        *(facets or []),
        time=time or "",
        time_select=cast(Literal["file", "flexible", "strict"], time_select),
        flavour=cast(
            Literal["freva", "cmip6", "cmip5", "cordex", "nextgems"],
            flavour.value,
        ),
        uniq_key=cast(Literal["uri", "file"], uniq_key.value),
        host=host,
        fail_on_error=False,
        multiversion=multiversion,
        stream_zarr=zarr,
        **(parse_cli_args(search_keys or [])),
    )
    if zarr:
        _auth(result._cfg.auth_url, access_token)
    with NamedTemporaryFile(suffix=".json") as temp_f:
        result._create_intake_catalogue_file(str(filename or temp_f.name))
        if not filename:
            print(Path(temp_f.name).read_text())


@databrowser_app.command(name="data-count", help="Count the databrowser search results")
@exception_handler
def count_values(
    search_keys: Optional[List[str]] = typer.Argument(
        default=None,
        help="Refine your data search with this `key=value` pair search "
        "parameters. The parameters could be, depending on the DRS standard, "
        "flavour product, project model etc.",
    ),
    facets: Optional[List[str]] = typer.Option(
        None,
        "--facet",
        help=(
            "If you are not sure about the correct search key's you can use"
            " the ``--facet`` flag to search of any matching entries. For "
            "example --facet 'era5' would allow you to search for any entries"
            " containing era5, regardless of project, product etc."
        ),
    ),
    detail: bool = typer.Option(
        False,
        "--detail",
        "-d",
        help=("Separate the count by search facets."),
    ),
    flavour: Flavours = typer.Option(
        "freva",
        "--flavour",
        "-f",
        help=(
            "The Data Reference Syntax (DRS) standard specifying the type "
            "of climate datasets to query."
        ),
    ),
    time_select: TimeSelect = typer.Option(
        "flexible",
        "-ts",
        "--time-select",
        help=TimeSelect.get_help(),
    ),
    time: Optional[str] = typer.Option(
        None,
        "-t",
        "--time",
        help=(
            "Special search facet to refine/subset search results by time. "
            "This can be a string representation of a time range or a single "
            "time step. The time steps have to follow ISO-8601. Valid strings "
            "are ``%Y-%m-%dT%H:%M`` to ``%Y-%m-%dT%H:%M`` for time ranges and "
            "``%Y-%m-%dT%H:%M``. **Note**: You don't have to give the full "
            "string format to subset time steps ``%Y``, ``%Y-%m`` etc are also"
            " valid."
        ),
    ),
    extended_search: bool = typer.Option(
        False,
        "-e",
        "--extended-search",
        help="Retrieve information on additional search keys.",
    ),
    multiversion: bool = typer.Option(
        False,
        "--multi-version",
        help="Select all versions and not just the latest version (default).",
    ),
    host: Optional[str] = typer.Option(
        None,
        "--host",
        help=(
            "Set the hostname of the databrowser, if not set (default) "
            "the hostname is read from a config file"
        ),
    ),
    parse_json: bool = typer.Option(
        False, "-j", "--json", help="Parse output in json format."
    ),
    verbose: int = typer.Option(0, "-v", help="Increase verbosity", count=True),
    version: Optional[bool] = typer.Option(
        False,
        "-V",
        "--version",
        help="Show verion an exit",
        callback=version_callback,
    ),
) -> None:
    """Search metadata (facets) based on the specified Data Reference Syntax
    (DRS) standard (flavour) and the type of search result (uniq_key), which
    can be either file or uri. Facets represent the metadata categories
    associated with the climate datasets, such as experiment, model,
    institute, and more. This method provides a comprehensive view of the
    available facets and their corresponding counts based on the provided
    search criteria.
    """
    logger.set_verbosity(verbose)
    logger.debug("Search the databrowser")
    result: Union[int, Dict[str, Dict[str, int]]] = 0
    search_kws = parse_cli_args(search_keys or [])
    time = cast(str, time or search_kws.pop("time", ""))
    facets = facets or []
    if detail:
        result = databrowser.count_values(
            *facets,
            time=time or "",
            time_select=cast(Literal["file", "flexible", "strict"], time_select),
            flavour=cast(
                Literal["freva", "cmip6", "cmip5", "cordex", "nextgems"],
                flavour.value,
            ),
            host=host,
            extended_search=extended_search,
            multiversion=multiversion,
            fail_on_error=False,
            **search_kws,
        )
    else:
        result = len(
            databrowser(
                *facets,
                time=time or "",
                time_select=cast(Literal["file", "flexible", "strict"], time_select),
                flavour=cast(
                    Literal["freva", "cmip6", "cmip5", "cordex", "nextgems"],
                    flavour.value,
                ),
                host=host,
                multiversion=multiversion,
                fail_on_error=False,
                uniq_key="file",
                stream_zarr=False,
                **search_kws,
            )
        )
    if parse_json:
        print(json.dumps(result))
        return
    if isinstance(result, dict):
        for key, values in result.items():
            counts = []
            for facet, count in values.items():
                counts.append(f"{facet}[{count}]")
            print(f"{key}: {', '.join(counts)}")
    else:
        print(result)
