"""Query climate data sets by using-key value pair search queries."""

import sys
from collections import defaultdict
from fnmatch import fnmatch
from functools import cached_property
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import (
    Any,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
)

import intake
import requests
import xarray as xr
import yaml
from rich import print as pprint

from .auth import Auth
from .utils import logger
from .utils.databrowser_utils import Config, UserDataHandler

__all__ = ["databrowser"]


class databrowser:
    """Find data in the system.

    You can either search for files or uri's. Uri's give you an information
    on the storage system where the files or objects you are looking for are
    located. The query is of the form ``key=value``. For ``value`` you might
    use wild cards such as \\*, ? or any regular expression.

    Parameters
    ~~~~~~~~~~

    *facets: str
        If you are not sure about the correct search key's you can use
        positional arguments to search of any matching entries. For example
        'era5' would allow you to search for any entries
        containing era5, regardless of project, product etc.
    **search_keys: str
        The search constraints applied in the data search. If not given
        the whole dataset will be queried.
    flavour: str, default: freva
        The Data Reference Syntax (DRS) standard specifying the type of climate
        datasets to query. You can get an overview by using the
        :py:meth:databrowser.overview class method to retrieve information
        on the available search flavours and their different search keys.
    time: str, default: ""
        Special search key to refine/subset search results by time.
        This can be a string representation of a time range or a single
        timestamp. The timestamps has to follow ISO-8601. Valid strings are
        ``%Y-%m-%dT%H:%M to %Y-%m-%dT%H:%M`` for time ranges or
        ``%Y-%m-%dT%H:%M`` for single time stamps.

        .. note:: You don't have to give the full string format to subset time
                steps ``%Y``, ``%Y-%m`` etc are also valid.

    time_select: str, default: flexible
        Operator that specifies how the time period is selected. Choose from
        flexible (default), strict or file. ``strict`` returns only those files
        that have the `entire` time period covered. The time search ``2000 to
        2012`` will not select files containing data from 2010 to 2020 with
        the ``strict`` method. ``flexible`` will select those files as
        ``flexible`` returns those files that have either start or end period
        covered. ``file`` will only return files where the entire time
        period is contained within `one single` file.
    bbox: str, default: ""
        Special search facet to refine/subset search results by spatial extent.
        This can be a list representation of a bounding box or a WKT polygon.
        Valid lists are ``min_lon max_lon min_lat max_lat`` for bounding
        boxes and Well-Known Text (WKT) format for polygons.

    bbox_select: str, default: flexible
        Operator that specifies how the spatial extent is selected. Choose from
        flexible (default), strict or file. ``strict`` returns only those files
        that fully contain the query extent. The bbox search ``-10 10 -10 10``
        will not select files covering only ``0 5 0 5`` with the ``strict``
        method. ``flexible`` will select those files as it returns files that
        have any overlap with the query extent. ``file`` will only return files
        where the entire spatial extent is contained by the query geometry.
    uniq_key: str, default: file
        Chose if the solr search query should return paths to files or
        uris, uris will have the file path along with protocol of the storage
        system. URIs are useful when working with libraries like fsspec, which
        require protocol information.
    host: str, default: None
        Override the host name of the databrowser server. This is usually the
        url where the freva web site can be found. Such as www.freva.dkrz.de.
        By default no host name is given and the host name will be taken from
        the freva config file.
    stream_zarr: bool, default: False
        Create a zarr stream for all search results. When set to true the
        files are served in zarr format and can be opened from anywhere.
    multiversion: bool, default: False
        Select all versions and not just the latest version (default).
    fail_on_error: bool, default: False
        Make the call fail if the connection to the databrowser could not
        be established.


    Attributes
    ~~~~~~~~~~

    url: str
        the url of the currently selected databrowser api server
    metadata: dict[str, str]
        The available search keys, or metadata, found for the applied search
        constraints. This can be useful for reverse searches.


    Example
    ~~~~~~~

    Search for the cmorph datasets. Suppose we know that the experiment name
    of this dataset is cmorph therefore we can create in instance of the
    databrowser class using the ``experiment`` search constraint.
    If you just 'print' the created object you will get a quick overview:

    .. execute_code::

        from freva_client import databrowser
        db = databrowser(experiment="cmorph", uniq_key="uri")
        print(db)

    After having created the search object you can acquire different kinds of
    information like the number of found objects:

    .. execute_code::

        from freva_client import databrowser
        db = databrowser(experiment="cmorph", uniq_key="uri")
        print(len(db))
        # Get all the search keys associated with this search

    Or you can retrieve the combined metadata of the search objects.

    .. execute_code::

        from freva_client import databrowser
        db = databrowser(experiment="cmorph", uniq_key="uri")
        print(db.metadata)

    Most importantly you can retrieve the locations of all encountered objects

    .. execute_code::

        from freva_client import databrowser
        db = databrowser(experiment="cmorph", uniq_key="uri")
        for file in db:
            pass
        all_files = sorted(db)
        print(all_files[0])


    You can also set a different flavour, for example according to cmip6
    standard:

    .. execute_code::

        from freva_client import databrowser
        db = databrowser(flavour="cmip6", experiment_id="cmorph")
        print(db.metadata)


    Sometimes you don't exactly know the exact names of the search keys and
    want retrieve all file objects that match a certain category. For example
    for getting all ocean reanalysis datasets you can apply the 'reana*'
    search key as a positional argument:

    .. execute_code::

        from freva_client import databrowser
        db = databrowser("reana*", realm="ocean", flavour="cmip6")
        for file in db:
            print(file)

    If you don't have direct access to the data, for example because you are
    not directly logged in to the computer where the data is stored you can
    set ``stream_zarr=True``. The data will then be
    provisioned in zarr format and can be opened from anywhere. But bear in
    mind that zarr streams if not accessed in time will expire. Since the
    data can be accessed from anywhere you will also have to authenticate
    before you are able to access the data. Refer also to the
    :py:meth:`freva_client.authenticate` method.

    .. execute_code::

        from freva_client import authenticate, databrowser
        token_info = authenticate(username="janedoe")
        db = databrowser(dataset="cmip6-fs", stream_zarr=True)
        zarr_files = list(db)
        print(zarr_files)

    After you have created the paths to the zarr files you can open them

    ::

        import xarray as xr
        dset = xr.open_dataset(
           zarr_files[0],
           chunks="auto",
           engine="zarr",
           storage_options={"header":
                {"Authorization": f"Bearer {token_info['access_token']}"}
           }
        )


    """

    def __init__(
        self,
        *facets: str,
        uniq_key: Literal["file", "uri"] = "file",
        flavour: Literal[
            "freva", "cmip6", "cmip5", "cordex", "nextgems", "user"
        ] = "freva",
        time: Optional[str] = None,
        host: Optional[str] = None,
        time_select: Literal["flexible", "strict", "file"] = "flexible",
        bbox: Optional[Tuple[float, float, float, float]] = None,
        bbox_select: Literal["flexible", "strict", "file"] = "flexible",
        stream_zarr: bool = False,
        multiversion: bool = False,
        fail_on_error: bool = False,
        **search_keys: Union[str, List[str]],
    ) -> None:
        self._auth = Auth()
        self._fail_on_error = fail_on_error
        self._cfg = Config(host, uniq_key=uniq_key, flavour=flavour)
        self._flavour = flavour
        self._stream_zarr = stream_zarr
        facet_search: Dict[str, List[str]] = defaultdict(list)
        for key, value in search_keys.items():
            if isinstance(value, str):
                facet_search[key] = [value]
            else:
                facet_search[key] = value
        self._params: Dict[str, Union[str, bool, List[str]]] = {
            **{"multi-version": multiversion},
            **search_keys,
        }

        if time:
            self._params["time"] = time
            self._params["time_select"] = time_select
        if bbox:
            bbox_str = ",".join(map(str, bbox))
            self._params["bbox"] = bbox_str
            self._params["bbox_select"] = bbox_select
        if facets:
            self._add_search_keyword_args_from_facet(facets, facet_search)

    def _add_search_keyword_args_from_facet(
        self, facets: Tuple[str, ...], search_kw: Dict[str, List[str]]
    ) -> None:
        metadata = {
            k: v[::2]
            for (k, v) in self._facet_search(extended_search=True).items()
        }
        primary_key = list(metadata.keys() or ["project"])[0]
        num_facets = 0
        for facet in facets:
            for key, values in metadata.items():
                for value in values:
                    if fnmatch(value, facet):
                        num_facets += 1
                        search_kw[key].append(value)

        if facets and num_facets == 0:
            # TODO: This isn't pretty, but if a user requested a search
            # string that doesn't exist than we have to somehow make the search
            # return nothing.
            search_kw = {primary_key: ["NotAvailable"]}
        self._params.update(search_kw)

    def __iter__(self) -> Iterator[str]:
        query_url = self._cfg.search_url
        headers = {}
        if self._stream_zarr:
            query_url = self._cfg.zarr_loader_url
            token = self._auth.check_authentication(auth_url=self._cfg.auth_url)
            headers = {"Authorization": f"Bearer {token['access_token']}"}
        result = self._request("GET", query_url, headers=headers, stream=True)
        if result is not None:
            try:
                for res in result.iter_lines():
                    yield res.decode("utf-8")
            except KeyboardInterrupt:
                pprint("[red][b]User interrupt: Exit[/red][/b]", file=sys.stderr)

    def __repr__(self) -> str:
        params = ", ".join(
            [f"{k.replace('-', '_')}={v}" for (k, v) in self._params.items()]
        )
        return (
            f"{self.__class__.__name__}(flavour={self._flavour}, "
            f"host={self.url}, {params})"
        )

    def _repr_html_(self) -> str:
        params = ", ".join(
            [f"{k.replace('-', '_')}={v}" for (k, v) in self._params.items()]
        )

        found_objects_count = len(self)

        available_flavours = ", ".join(
            flavour for flavour in self._cfg.overview["flavours"]
        )
        available_search_facets = ", ".join(
            facet for facet in self._cfg.overview["attributes"][self._flavour]
        )

        # Create a table-like structure for available flavors and search facets
        style = 'style="text-align: left"'
        facet_heading = (
            f"Available search facets for <em>{self._flavour}</em> flavour"
        )
        html_repr = (
            "<table>"
            f"<tr><th colspan='2' {style}>{self.__class__.__name__}"
            f"(flavour={self._flavour}, host={self.url}, "
            f"{params})</th></tr>"
            f"<tr><td><b># objects</b></td><td {style}>{found_objects_count}"
            "</td></tr>"
            f"<tr><td valign='top'><b>{facet_heading}</b></td>"
            f"<td {style}>{available_search_facets}</td></tr>"
            "<tr><td valign='top'><b>Available flavours</b></td>"
            f"<td {style}>{available_flavours}</td></tr>"
            "</table>"
        )

        return html_repr

    def __len__(self) -> int:
        """Query the total number of found objects.

        Example
        ~~~~~~~
        .. execute_code::

            from freva_client import databrowser
            print(len(databrowser(experiment="cmorph")))


        """
        result = self._request("GET", self._cfg.metadata_url)
        if result:
            return cast(int, result.json().get("total_count", 0))
        return 0

    def _create_intake_catalogue_file(self, filename: str) -> None:
        """Create an intake catalogue file."""
        kwargs: Dict[str, Any] = {"stream": True}
        url = self._cfg.intake_url
        if self._stream_zarr:
            token = self._auth.check_authentication(auth_url=self._cfg.auth_url)
            url = self._cfg.zarr_loader_url
            kwargs["headers"] = {
                "Authorization": f"Bearer {token['access_token']}"
            }
            kwargs["params"] = {"catalogue-type": "intake"}
        result = self._request("GET", url, **kwargs)
        if result is None:
            raise ValueError("No results found")

        try:
            Path(filename).parent.mkdir(exist_ok=True, parents=True)
            with open(filename, "bw") as stream:
                for content in result.iter_content(decode_unicode=False):
                    stream.write(content)
        except Exception as error:
            raise ValueError(
                f"Couldn't write catalogue content: {error}"
            ) from None

    def intake_catalogue(self) -> Any:
        """Create an intake esm catalogue object from the search.

        This method creates a intake-esm catalogue from the current object
        search. Instead of having the original files as target objects you can
        also choose to stream the files via zarr.

        Returns
        ~~~~~~~
        intake_esm.core.esm_datastore: intake-esm catalogue.

        Raises
        ~~~~~~
        ValueError: If user is not authenticated or catalogue creation failed.

        Example
        ~~~~~~~
        Let's create an intake-esm catalogue that points points allows for
        streaming the target data as zarr:

        .. execute_code::

            from freva_client import databrowser
            db = databrowser(dataset="cmip6-hsm", stream_zarr=True)
            cat = db.intake_catalogue()
            print(cat.df)
        """
        with NamedTemporaryFile(suffix=".json") as temp_f:
            self._create_intake_catalogue_file(temp_f.name)
            return intake.open_esm_datastore(temp_f.name)

    def stac_catalogue(
        self,
        filename: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> str:
        """Create an static or dynamic STAC API catalogue endpoint
        from the search. This method creates a STAC API collection
        endpoint from the current object search.

        Parameters
        ~~~~~~~~~~
        stac_dynamic: bool, default: False
            Create a dynamic STAC API collection endpoint. If set to False
            a static STAC catalogue will be created.
        filename: str, default: None
            The filename of the STAC catalogue. If not given the STAC catalogue
            will be saved to the current working directory.
        **kwargs: Any
            Additional keyword arguments to be passed to the request.

        Returns
        ~~~~~~~
        Union[Dict[str, Any], BinaryIO]
        Either a STAC API collection endpoint as a dictionary (for dynamic catalogs),
        or a tar.gz file stream (for static catalogs).

        Raises
        ~~~~~~
        ValueError: If stac-catalogue creation failed.

        Example
        ~~~~~~~
        Let's create an STAC API collection endpoint to get access
        to the metadata and data:

        .. execute_code::

            from freva_client import databrowser
            db = databrowser(dataset="cmip6-hsm")
            print(db.stac_catalogue())

        Now let's create a static STAC catalogue:

        .. execute_code::

            from tempfile import mktemp
            temp_path = mktemp(suffix=".tar.gz")

            from freva_client import databrowser
            db = databrowser(dataset="cmip6-hsm")
            db.stac_catalogue(filename=temp_path)
            print(f"STAC catalog saved to: {temp_path}")

        """
        params: Dict[str, Any] = {
            "stac_dynamic": False if filename is not None else True
        }
        kwargs.update({"stream": True,
                       "params": params})
        stac_url = self._cfg.stac_url
        result = self._request("GET", stac_url, **kwargs)
        if result is None or result.status_code == 404:
            raise ValueError(
                "No STAC catalog found. Please check if you have any search results."
            )
        if params.get("stac_dynamic") is False:
            default_filename = result.headers.get('Content-Disposition', '').split(
                'filename=')[-1].strip('"')

            save_path = Path(cast(str, filename))

            if save_path.is_dir() and save_path.exists():
                save_path = save_path / default_filename

            save_path.parent.mkdir(parents=True, exist_ok=True)

            total_size = 0
            with open(save_path, 'wb') as f:
                for chunk in result.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        total_size += len(chunk)

            return (
                f"STAC catalog saved to: {save_path} "
                f"(size: {total_size / 1024 / 1024:.2f} MB). "
                f"Or simply download from: {result.url}"
            )
        else:
            return f"STAC catalog available at: {result.url}"

    @classmethod
    def count_values(
        cls,
        *facets: str,
        flavour: Literal[
            "freva", "cmip6", "cmip5", "cordex", "nextgems", "user"
        ] = "freva",
        time: Optional[str] = None,
        host: Optional[str] = None,
        time_select: Literal["flexible", "strict", "file"] = "flexible",
        bbox: Optional[Tuple[float, float, float, float]] = None,
        bbox_select: Literal["flexible", "strict", "file"] = "flexible",
        multiversion: bool = False,
        fail_on_error: bool = False,
        extended_search: bool = False,
        **search_keys: Union[str, List[str]],
    ) -> Dict[str, Dict[str, int]]:
        """Count the number of objects in the databrowser.

        Parameters
        ~~~~~~~~~~

        *facets: str
            If you are not sure about the correct search key's you can use
            positional arguments to search of any matching entries. For example
            'era5' would allow you to search for any entries
            containing era5, regardless of project, product etc.
        flavour: str, default: freva
            The Data Reference Syntax (DRS) standard specifying the type of climate
            datasets to query.
        time: str, default: ""
            Special search facet to refine/subset search results by time.
            This can be a string representation of a time range or a single
            timestamp. The timestamp has to follow ISO-8601. Valid strings are
            ``%Y-%m-%dT%H:%M`` to ``%Y-%m-%dT%H:%M`` for time ranges and
            ``%Y-%m-%dT%H:%M``.

            .. note:: You don't have to give the full string format to subset time
                    steps ``%Y``, ``%Y-%m`` etc are also valid.

        time_select: str, default: flexible
            Operator that specifies how the time period is selected. Choose from
            flexible (default), strict or file. ``strict`` returns only those files
            that have the *entire* time period covered. The time search ``2000 to
            2012`` will not select files containing data from 2010 to 2020 with
            the ``strict`` method. ``flexible`` will select those files as
            ``flexible`` returns those files that have either start or end period
            covered. ``file`` will only return files where the entire time
            period is contained within `one single` file.
        bbox: str, default: ""
            Special search facet to refine/subset search results by spatial extent.
            This can be a list representation of a bounding box or a WKT polygon.
            Valid lists are ``min_lon max_lon min_lat max_lat`` for bounding
            boxes and Well-Known Text (WKT) format for polygons.

        bbox_select: str, default: flexible
            Operator that specifies how the spatial extent is selected. Choose from
            flexible (default), strict or file. ``strict`` returns only those files
            that fully contain the query extent. The bbox search ``-10 10 -10 10``
            will not select files covering only ``0 5 0 5`` with the ``strict``
            method. ``flexible`` will select those files as it returns files that
            have any overlap with the query extent. ``file`` will only return files
            where the entire spatial extent is contained by the query geometry.
        extended_search: bool, default: False
            Retrieve information on additional search keys.
        host: str, default: None
            Override the host name of the databrowser server. This is usually
            the url where the freva web site can be found. Such as
            www.freva.dkrz.de. By default no host name is given and the host
            name will be taken from the freva config file.
        multiversion: bool, default: False
            Select all versions and not just the latest version (default).
        fail_on_error: bool, default: False
            Make the call fail if the connection to the databrowser could not
            be established.
        **search_keys: str
            The search constraints to be applied in the data search. If not given
            the whole dataset will be queried.

        Returns
        ~~~~~~~
        dict[str, int]:
            Dictionary with the number of objects for each search facet/key
            is given.

        Example
        ~~~~~~~

        .. execute_code::

            from freva_client import databrowser
            print(databrowser.count_values(experiment="cmorph"))

        .. execute_code::

            from freva_client import databrowser
            print(databrowser.count_values("model"))

        Sometimes you don't exactly know the exact names of the search keys and
        want retrieve all file objects that match a certain category. For
        example for getting all ocean reanalysis datasets you can apply the
        'reana*' search key as a positional argument:

        .. execute_code::

            from freva_client import databrowser
            print(databrowser.count_values("reana*", realm="ocean", flavour="cmip6"))

        """
        this = cls(
            *facets,
            flavour=flavour,
            time=time,
            time_select=time_select,
            bbox=bbox,
            bbox_select=bbox_select,
            host=host,
            multiversion=multiversion,
            fail_on_error=fail_on_error,
            uniq_key="file",
            stream_zarr=False,
            **search_keys,
        )
        result = this._facet_search(extended_search=extended_search)
        counts = {}
        for facet, value_counts in result.items():
            counts[facet] = dict(
                zip(value_counts[::2], map(int, value_counts[1::2]))
            )
        return counts

    @cached_property
    def metadata(self) -> Dict[str, List[str]]:
        """Get the metadata (facets) for the current databrowser query.

        You can retrieve all information that is associated with your current
        databrowser search. This can be useful for reverse searches for example
        for retrieving metadata of object stores or file/directory names.

        Example
        ~~~~~~~

        Reverse search: retrieving meta data from a known file

        .. execute_code::

            from freva_client import databrowser
            db = databrowser(uri="slk:///arch/*/CPC/*")
            print(db.metadata)


        """
        return {
            k: v[::2]
            for (k, v) in self._facet_search(extended_search=True).items()
        }

    @classmethod
    def metadata_search(
        cls,
        *facets: str,
        flavour: Literal[
            "freva", "cmip6", "cmip5", "cordex", "nextgems", "user"
        ] = "freva",
        time: Optional[str] = None,
        host: Optional[str] = None,
        time_select: Literal["flexible", "strict", "file"] = "flexible",
        bbox: Optional[Tuple[float, float, float, float]] = None,
        bbox_select: Literal["flexible", "strict", "file"] = "flexible",
        multiversion: bool = False,
        fail_on_error: bool = False,
        extended_search: bool = False,
        **search_keys: Union[str, List[str]],
    ) -> Dict[str, List[str]]:
        """Search for data attributes (facets) in the databrowser.

        The method queries the databrowser for available search facets (keys)
        like model, experiment etc.

        Parameters
        ~~~~~~~~~~

        *facets: str
            If you are not sure about the correct search key's you can use
            positional arguments to search of any matching entries. For example
            'era5' would allow you to search for any entries
            containing era5, regardless of project, product etc.
        flavour: str, default: freva
            The Data Reference Syntax (DRS) standard specifying the type of climate
            datasets to query.
        time: str, default: ""
            Special search facet to refine/subset search results by time.
            This can be a string representation of a time range or a single
            timestamp. The timestamp has to follow ISO-8601. Valid strings are
            ``%Y-%m-%dT%H:%M`` to ``%Y-%m-%dT%H:%M`` for time ranges and
            ``%Y-%m-%dT%H:%M``.

            .. note:: You don't have to give the full string format to subset time
                    steps ``%Y``, ``%Y-%m`` etc are also valid.

        time_select: str, default: flexible
            Operator that specifies how the time period is selected. Choose from
            flexible (default), strict or file. ``strict`` returns only those files
            that have the *entire* time period covered. The time search ``2000 to
            2012`` will not select files containing data from 2010 to 2020 with
            the ``strict`` method. ``flexible`` will select those files as
            ``flexible`` returns those files that have either start or end period
            covered. ``file`` will only return files where the entire time
            period is contained within *one single* file.
        bbox: str, default: ""
            Special search facet to refine/subset search results by spatial extent.
            This can be a list representation of a bounding box or a WKT polygon.
            Valid lists are ``min_lon max_lon min_lat max_lat`` for bounding
            boxes and Well-Known Text (WKT) format for polygons.

        bbox_select: str, default: flexible
            Operator that specifies how the spatial extent is selected. Choose from
            flexible (default), strict or file. ``strict`` returns only those files
            that fully contain the query extent. The bbox search ``-10 10 -10 10``
            will not select files covering only ``0 5 0 5`` with the ``strict``
            method. ``flexible`` will select those files as it returns files that
            have any overlap with the query extent. ``file`` will only return files
            where the entire spatial extent is contained by the query geometry.
        extended_search: bool, default: False
            Retrieve information on additional search keys.
        multiversion: bool, default: False
            Select all versions and not just the latest version (default).
        host: str, default: None
            Override the host name of the databrowser server. This is usually
            the url where the freva web site can be found. Such as
            www.freva.dkrz.de. By default no host name is given and the host
            name will be taken from the freva config file.
        fail_on_error: bool, default: False
            Make the call fail if the connection to the databrowser could not
            be established.
        **search_keys: str, list[str]
            The facets to be applied in the data search. If not given
            the whole dataset will be queried.

        Returns
        ~~~~~~~
        dict[str, list[str]]:
            Dictionary with a list search facet values for each search facet key


        Example
        ~~~~~~~

        .. execute_code::

            from freva_client import databrowser
            all_facets = databrowser.metadata_search(project='obs*')
            print(all_facets)

        You can also search for all metadata matching a search string:

        .. execute_code::

            from freva_client import databrowser
            spec_facets = databrowser.metadata_search("obs*")
            print(spec_facets)

        Get all models that have a given time step:

        .. execute_code::

            from freva_client import databrowser
            model = databrowser.metadata_search(
                project="obs*",
                time="2016-09-02T22:10"
            )
            print(model)

        Reverse search: retrieving meta data from a known file

        .. execute_code::

            from freva_client import databrowser
            res = databrowser.metadata_search(file="/arch/*CPC/*")
            print(res)

        Sometimes you don't exactly know the exact names of the search keys and
        want retrieve all file objects that match a certain category. For
        example for getting all ocean reanalysis datasets you can apply the
        'reana*' search key as a positional argument:

        .. execute_code::

            from freva_client import databrowser
            print(databrowser.metadata_search("reana*", realm="ocean", flavour="cmip6"))

        In datasets with multiple versions only the `latest` version (i.e.
        `highest` version number) is returned by default. Querying a specific
        version from a multi versioned datasets requires the ``multiversion``
        flag in combination with the ``version`` special attribute:

        .. execute_code::

            from freva_client import databrowser
            res = databrowser.metadata_search(dataset="cmip6-fs",
                model="access-cm2", version="v20191108", extended_search=True,
                multiversion=True)
            print(res)

        If no particular ``version`` is requested, information of all versions
        will be returned.

        """
        this = cls(
            *facets,
            flavour=flavour,
            time=time,
            time_select=time_select,
            bbox=bbox,
            bbox_select=bbox_select,
            host=host,
            multiversion=multiversion,
            fail_on_error=fail_on_error,
            uniq_key="file",
            stream_zarr=False,
            **search_keys,
        )
        return {
            k: v[::2]
            for (k, v) in this._facet_search(
                extended_search=extended_search
            ).items()
        }

    @classmethod
    def overview(cls, host: Optional[str] = None) -> str:
        """Get an overview over the available search options.

        If you don't know what search flavours or search keys you can use
        for searching the data you can use this method to get an overview
        over what is available.

        Parameters
        ~~~~~~~~~~

        host: str, default None
            Override the host name of the databrowser server. This is usually
            the url where the freva web site can be found. Such as
            www.freva.dkrz.de. By default no host name is given and the host
            name will be taken from the freva config file.

        Returns
        ~~~~~~~
        str: A string representation over what is available.

        Example
        ~~~~~~~

        .. execute_code::

            from freva_client import databrowser
            print(databrowser.overview())
        """
        overview = Config(host).overview.copy()
        overview["Available search flavours"] = overview.pop("flavours")
        overview["Search attributes by flavour"] = overview.pop("attributes")
        return yaml.safe_dump(overview)

    @property
    def url(self) -> str:
        """Get the url of the databrowser API.

        Example
        ~~~~~~~

        .. execute_code::

            from freva_client import databrowser
            db = databrowser()
            print(db.url)

        """
        return self._cfg.databrowser_url

    def _facet_search(
        self,
        extended_search: bool = False,
    ) -> Dict[str, List[str]]:
        result = self._request("GET", self._cfg.metadata_url)
        if result is None:
            return {}
        data = result.json()
        if extended_search:
            constraints = data["facets"].keys()
        else:
            constraints = data["primary_facets"]
        return {f: v for f, v in data["facets"].items() if f in constraints}

    @classmethod
    def userdata(
        cls,
        action: Literal["add", "delete"],
        userdata_items: Optional[List[Union[str, xr.Dataset]]] = None,
        metadata: Optional[Dict[str, str]] = None,
        host: Optional[str] = None,
        fail_on_error: bool = False,
    ) -> None:
        """Add or delete user data in the databrowser system.

        Manage user data in the databrowser system by adding new data or
        deleting existing data.

        For the "``add``" action, the user can provide data items (file paths
        or xarray datasets) along with metadata (key-value pairs) to
        categorize and organize the data.

        For the "``delete``" action, the user provides metadata as search
        criteria to identify and remove the existing data from the
        system.

        Parameters
        ~~~~~~~~~~
        action : Literal["add", "delete"]
            The action to perform: "add" to add new data, or "delete"
            to remove existing data.
        userdata_items : List[Union[str, xr.Dataset]], optional
            A list of user file paths or xarray datasets to add to the
            databrowser (required for "add").
        metadata : Dict[str, str], optional
            Key-value metadata pairs to categorize the data (for "add")
            or search and identify data for
            deletion (for "delete").
        host : str, optional
            Override the host name of the databrowser server. This is usually
            the url where the freva web site can be found. Such as
            www.freva.dkrz.de. By default no host name is given and the host
            name will be taken from the freva config file.
        fail_on_error : bool, optional
            Make the call fail if the connection to the databrowser could not
            be established.

        Raises
        ~~~~~~
        ValueError
            If the operation fails or required parameters are missing
            for the specified action.
        FileNotFoundError
            If no user data is provided for the "add" action.

        Example
        ~~~~~~~

        Adding user data:

        .. execute_code::

            from freva_client import authenticate, databrowser
            import xarray as xr
            token_info = authenticate(username="janedoe")
            filenames = (
                "../freva-rest/src/freva_rest/databrowser_api/mock/data/model/regional/cordex/output/EUR-11/"
                "GERICS/NCC-NorESM1-M/rcp85/r1i1p1/GERICS-REMO2015/v1/3hr/pr/v20181212/*.nc"
            )
            filename1 = (
                "../freva-rest/src/freva_rest/databrowser_api/mock/data/model/regional/cordex/output/EUR-11/"
                "CLMcom/MPI-M-MPI-ESM-LR/historical/r0i0p0/CLMcom-CCLM4-8-17/v1/fx/orog/v20140515/"
                "orog_EUR-11_MPI-M-MPI-ESM-LR_historical_r1i1p1_CLMcom-CCLM4-8-17_v1_fx.nc"
            )
            xarray_data = xr.open_dataset(filename1)
            databrowser.userdata(
                action="add",
                userdata_items=[xarray_data, filenames],
                metadata={"project": "cmip5", "experiment": "myFavExp"}
            )

        Deleting user data:

        .. execute_code::

            from freva_client import authenticate, databrowser
            token_info = authenticate(username="janedoe")
            databrowser.userdata(
                action="delete",
                metadata={"project": "cmip5", "experiment": "myFavExp"}
            )
        """
        this = cls(
            host=host,
            fail_on_error=fail_on_error,
        )
        userdata_items = userdata_items or []
        metadata = metadata or {}
        url = f"{this._cfg.userdata_url}"
        token = this._auth.check_authentication(auth_url=this._cfg.auth_url)
        headers = {"Authorization": f"Bearer {token['access_token']}"}
        payload_metadata: dict[str, Collection[Collection[str]]] = {}

        if action == "add":
            user_data_handler = UserDataHandler(userdata_items)
            if user_data_handler.user_metadata:
                payload_metadata = {
                    "user_metadata": user_data_handler.user_metadata,
                    "facets": metadata,
                }
                result = this._request(
                    "POST", url, data=payload_metadata, headers=headers
                )
                if result is not None:
                    response_data = result.json()
                    status_message = response_data.get("status")
                else:
                    raise ValueError("Failed to add user data")
                pprint(f"[b][green]{status_message}[green][b]")
            else:
                raise ValueError("No metadata generated from the input data.")

        if action == "delete":
            if userdata_items:
                logger.info(
                    "'userdata_items' are not needed for the 'delete'"
                    "action and will be ignored."
                )

            result = this._request("DELETE", url, data=metadata, headers=headers)

            if result is None:
                raise ValueError("Failed to delete user data")
            pprint("[b][green]User data deleted successfully[green][b]")

    def _request(
        self,
        method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"],
        url: str,
        data: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[requests.models.Response]:
        """Request method to handle CRUD operations (GET, POST, PUT, PATCH, DELETE)."""
        method_upper = method.upper()
        timeout = kwargs.pop("timeout", 30)
        params = kwargs.pop("params", {})
        stream = kwargs.pop("stream", False)

        logger.debug(
            "%s request to %s with data: %s and parameters: %s",
            method_upper,
            url,
            data,
            {**self._params, **params},
        )

        try:
            req = requests.Request(
                method=method_upper,
                url=url,
                params={**self._params, **params},
                json=None if method_upper in "GET" else data,
                **kwargs,
            )
            with requests.Session() as session:
                prepared = session.prepare_request(req)
                res = session.send(prepared, timeout=timeout, stream=stream)
                res.raise_for_status()
                return res

        except KeyboardInterrupt:
            pprint("[red][b]User interrupt: Exit[/red][/b]", file=sys.stderr)
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
        ) as error:
            msg = f"{method_upper} request failed with {error}"
            if self._fail_on_error:
                raise ValueError(msg) from None
            logger.warning(msg)
        return None
