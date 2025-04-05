"""STAC class for generating the STATIC STAC catalog."""

import ast
import io
import json
from datetime import datetime
from textwrap import dedent
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from dateutil import parser
from fastapi import Request

from freva_rest.config import ServerConfig
from freva_rest.logger import logger

from .core import FlavourType, Solr, Translator


class ZipStream(io.RawIOBase):
    """Custom unseekable stream that buffers writes and
    flushes its content after writing.

    In spite of having simillar libraries for this purpose,
    we designed this lean class to have a better control
    on memory usage and to have a better understanding
    of the underlying mechanism. Most of existing libraries
    due to the nature of their design, they consume a certain
    amount of memory which doesn't make them suitable for
    Freva case and generally doesn't make any sense to use.

    Another benefit of streaming `zip` over `tar.gz` is that it
    allows the JSON file to be streamed into the ZIP archive as it's
    being generated. This means we don't have to wait for the
    entire file to be created beforehand, which improves memory
    efficiency.

    """
    def __init__(self) -> None:
        self._buffer = bytearray()
        self._closed = False

    def close(self) -> None:
        self._closed = True

    def write(self, b) -> int:  # type: ignore
        """ Write the buffer into the stream """
        if self._closed:
            raise ValueError("Can't write to a closed stream")  # pragma: no cover
        self._buffer += b
        return len(b)

    def flush_and_read(self) -> bytes:
        """ Flush the buffer and return the chunk """
        chunk = bytes(self._buffer)
        self._buffer.clear()
        return chunk


class STAC(Solr):
    """STAC class to create static STAC catalogues."""
    def __init__(
        self,
        config: ServerConfig,
        *,
        uniq_key: Literal["file", "uri"] = "file",
        flavour: FlavourType = "freva",
        start: int = 0,
        multi_version: bool = True,
        translate: bool = True,
        _translator: Union[None, Translator] = None,
        **query: list[str],
    ):
        super().__init__(
            config,
            uniq_key=uniq_key,
            flavour=flavour,
            start=start,
            multi_version=multi_version,
            translate=translate,
            _translator=_translator,
            **query
        )
        self.config = config
        self._zip_stream = ZipStream()
        self._zip = ZipFile(self._zip_stream, mode="w", compression=ZIP_DEFLATED)
        self.spatial_extent = {
            "minx": float("-180"),
            "miny": float("-90"),
            "maxx": float("180"),
            "maxy": float("90"),
        }
        self.temporal_extent: dict[str, Optional[datetime]] = {
            "start": None,
            "end": None,
        }
        self.assets_prereqs: Dict[str, Union[str, int]] = {}

    @classmethod
    async def validate_parameters(
        cls,
        config: ServerConfig,
        *,
        uniq_key: Literal["file", "uri"] = "file",
        flavour: FlavourType = "freva",
        start: int = 0,
        multi_version: bool = False,
        translate: bool = True,
        **query: list[str],
    ) -> "STAC":
        """Use Solr validate_parameters and return a STAC validated_parameters
        for validating the search params through Solr in STAC inheritated cls."""
        solr_instance = await super().validate_parameters(
            config,
            uniq_key=uniq_key,
            flavour=flavour,
            start=start,
            multi_version=multi_version,
            translate=translate,
            **query
        )

        return cls(
            config,
            uniq_key=uniq_key,
            flavour=flavour,
            start=start,
            multi_version=multi_version,
            translate=translate,
            _translator=solr_instance.translator,
            **query
        )

    async def _iter_stac_items(self) -> "AsyncIterator[List[Item]]":
        self.query["cursorMark"] = "*"
        items_batch = []
        while True:
            async with self._session_get() as res:
                _, results = res
            for result in results.get("response", {}).get("docs", [{}]):
                item = await self._create_stac_item(result)
                self.count_item += 1
                items_batch.append(item)

                if len(items_batch) >= self.batch_size:
                    yield items_batch
                    items_batch = []

            if items_batch:
                yield items_batch
            next_cursor_mark = results.get("nextCursorMark", None)
            if next_cursor_mark == self.query["cursorMark"] or not results:
                break
            self.query["cursorMark"] = next_cursor_mark

    async def validate_stac(self) -> Tuple[int, int]:
        """Validate search and get result counts."""
        self._set_catalogue_queries()
        self.query["facet.field"] = self._config.solr_fields + ["time", "bbox"]
        self.query["fl"] = [self.uniq_key] + self._config.solr_fields + ["time", "bbox"]
        async with self._session_get() as res:
            search_status, search = res
        total_count = int(search.get("response", {}).get("numFound", 0))
        return search_status, total_count

    async def init_stac_catalogue(
        self,
        request: Request,
    ) -> None:
        filtered_params = dict(request.query_params)
        filtered_params.pop('translate', None)
        filtered_params.pop('start', None)

        self.assets_prereqs = {
            "base_url": str(self.config.proxy) + "/",
            "full_endpoint": (
                f"{self.config.proxy}/"
                f"{str(request.url).split(str(request.base_url))[-1]}"
            ),
            "only_params": str(filtered_params) if filtered_params != {} else "",
        }

    def parse_datetime(self, time_str: str) -> Tuple[datetime, datetime]:
        """
        Parse a time range string into start and end datetimes.

        Parameters
        ----------
        time_str : str
            Time range string in rdate format '[start_time TO end_time]'

        Returns
        -------
        Tuple[datetime, datetime]
            Start and end datetime objects
        """
        clean_start_time = time_str.replace("[", "").split(" TO ")[0]
        clean_end_time = time_str.replace("]", "").split(" TO ")[1]
        return parser.parse(clean_start_time), parser.parse(clean_end_time)

    def parse_bbox(self, bbox_str: Union[str, List[str]]) -> List[float]:
        """
        Parse a bounding box string into coordinates.

        Parameters
        ----------
        bbox_str : Union[str, List[str]]
            Bounding box in ENVELOPE format: 'ENVELOPE(west,east,north,south)'
            or as a list with one element: ['ENVELOPE(west,east,north,south)']

        Returns
        -------
        List[float]
            Coordinates as [minx, miny, maxx, maxy]
        """
        bbox = bbox_str[0] if isinstance(bbox_str, list) else bbox_str

        nums = [
            float(x)
            for x in bbox.replace("ENVELOPE(", "").replace(")", "").split(",")
        ]
        return [nums[0], nums[3], nums[1], nums[2]]

    def _update_spatial_extent(self, bbox: List[float]) -> None:
        """
        Update collection's spatial extent based on item bbox.

        Parameters
        ----------
        bbox : List[float]
            Bounding box coordinates [minx, miny, maxx, maxy]
        """
        self.spatial_extent["minx"] = min(self.spatial_extent["minx"], bbox[0])
        self.spatial_extent["miny"] = min(self.spatial_extent["miny"], bbox[1])
        self.spatial_extent["maxx"] = max(self.spatial_extent["maxx"], bbox[2])
        self.spatial_extent["maxy"] = max(self.spatial_extent["maxy"], bbox[3])

    def _update_temporal_extent(self, start_time: datetime, end_time: datetime) -> None:
        """Update collection's temporal extent based on item timerange.

        Parameters
        ----------
        start_time : datetime
            Item's start datetime
        end_time : datetime
            Item's end datetime
        """
        if self.temporal_extent["start"] is None:
            self.temporal_extent["start"] = start_time
        elif start_time < self.temporal_extent["start"]:
            self.temporal_extent["start"] = start_time

        if self.temporal_extent["end"] is None:
            self.temporal_extent["end"] = end_time
        elif end_time > self.temporal_extent["end"]:
            self.temporal_extent["end"] = end_time

    async def _create_stac_item(self, result: Dict[str, Any]) -> "Item":
        id = result.get(self.uniq_key, "")
        params_dict = (
            ast.literal_eval(str(self.assets_prereqs.get("only_params")))
            if self.assets_prereqs.get("only_params", "")
            else {}
        )
        python_params = " ".join(
            f"{k}='{v}',"
            for k, v in params_dict.items()
            if k not in ("translate", "start")
        )

        cli_params = " ".join(
            f"{k}={v}" for k, v in params_dict.items()
            if k not in ("translate", "start")
        )

        api_params = "&".join(
            f"{k}={v}" for k, v in params_dict.items()
            if k not in ("translate", "start")
        )
        intake_desc = dedent(
            f"""
            # Installing Intake-ESM
            ```bash
            pip install intake-esm
            conda install -c conda-forge intake-esm
            ```
            # Quick Guide: INTAKE-ESM Catalog (Python)
            ```python
            import intake
            cat = intake.open_esm_datastore(
            '{str(self.assets_prereqs.get('full_endpoint')).replace(
                "stac-catalogue", "intake-catalogue")}')
            ```
            """
        )
        zarr_desc = dedent(
            f"""
            # Accessing Zarr Data
            1. Install freva-client
            ```bash
            pip install freva-client
            ```
            2. (Python) Get the auth token and access the zarr data - recommended
            ```python
            from freva_client import authenticate, databrowser
            import xarray as xr
            token_info = authenticate(username=<your_username>, \\
                host='{self.config.proxy}')
            db = databrowser({python_params} {self.uniq_key}='{id}', \\
                            stream_zarr=True, host='{self.config.proxy}')
            xarray_dataset = xr.open_mfdataset(list(db))
            ```
            3. (CLI) Get token then access:
            ```bash
            # Attention: jq has to be installed beforehand
            token=$(freva-client auth -u <username> --host {self.config.proxy}\\
                                                        |jq -r .access_token)
            freva-client databrowser data-search {cli_params} {self.uniq_key}={id} \\
                --zarr --host {self.config.proxy} --access-token $token
            ```
            4. Access the zarr data directly (API - language agnostic)
            ```bash
            curl -X GET {self.assets_prereqs.get('base_url')}api/ \\
            freva-nextgen/databrowser/load/\\
            {self.translator.flavour}?{api_params}\\
            &{self.uniq_key}={id} \\
              -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
            ```
            💡: Read more about the
            [freva-client](https://freva-clint.github.io/freva-nextgen/)
            """
        )  # noqa: E501
        local_access_desc = dedent(
            f"""
            # Accessing data locally where the data is stored
            ```python
            import xarray as xr
            ds = xr.open_mfdataset('{id}')
            ```
            💡: Ensure required xarray packages are installed.
            """
        )
        item_id = "item" + str(self.count_item)
        bbox = result.get("bbox")
        if bbox:
            try:
                bbox = self.parse_bbox(bbox)
                self._update_spatial_extent(bbox)
            except ValueError as e:  # pragma: no cover
                logger.warning(f"Invalid bbox for {id}: {e}")
                bbox = None

        time = result.get("time")
        start_time = end_time = None
        if time:
            try:
                start_time, end_time = self.parse_datetime(time)
                self._update_temporal_extent(start_time, end_time)
            except ValueError as e:
                logger.warning(f"Invalid datetime for {id}: {e}")

        geometry = None
        if bbox:
            geometry = {
                "type": "Polygon",
                "coordinates": [
                    [
                        [bbox[0], bbox[1]],
                        [bbox[2], bbox[1]],
                        [bbox[2], bbox[3]],
                        [bbox[0], bbox[3]],
                        [bbox[0], bbox[1]],
                    ]
                ],
            }

        properties = {
            **{
                k: result.get(k)
                for k in self._config.solr_fields
                if k in result and result.get(k) is not None
            },
            "title": id,
        }
        item = Item(
            id=item_id,
            collection=self.collection_id,
            geometry=geometry,
            properties=properties,
            bbox=bbox,
        )
        if start_time and end_time:
            item.properties["start_datetime"] = start_time.isoformat() + "Z"
            item.properties["end_datetime"] = end_time.isoformat() + "Z"
            item.properties["datetime"] = start_time.isoformat() + "Z"

        links_to_add = [
            {"rel": "self",
             "target": f"./{item_id}.json",
             "media_type": "application/json"},
            {"rel": "root",
             "target": "../../catalog.json",
             "media_type": "application/json"},
            {"rel": "parent",
             "target": "../../catalog.json",
             "media_type": "application/json"},
            {"rel": "collection",
             "target": "../collection.json",
             "media_type": "application/json"}
        ]
        for link_info in links_to_add:
            if not any(link.rel == link_info["rel"] for link in item.links):
                link = Link(
                    rel=link_info["rel"],
                    href=link_info["target"],
                    type=link_info["media_type"],
                    extra_fields={"noresolve": True}
                )
                item.add_link(link)

        assets = {
            "freva-databrowser": Asset(
                href=(
                    f"{self.assets_prereqs.get('base_url')}databrowser/?"
                    f"{api_params}&{self.uniq_key}={id}"
                ),
                media_type="text/html",
                title="Freva Web DataBrowser",
                description=(
                    "Access the Freva web interface for data exploration and analysis"
                ),
                roles=["overview"]
            ),
            "intake-catalogue": Asset(
                href=(
                    str(self.assets_prereqs.get("full_endpoint")).replace(
                        "stac-catalogue", "intake-catalogue"
                    )
                    + f"&{self.uniq_key}={id}"
                ),
                media_type="application/json",
                title="Intake Catalogue",
                description=intake_desc,
                roles=["metadata"]
            ),
            "zarr-access": Asset(
                href=(
                    f"{self.assets_prereqs.get('base_url')}api/freva-nextgen/"
                    f"databrowser/load/{self.translator.flavour}?"
                    f"{api_params}&{self.uniq_key}={id}"
                ),
                media_type="application/vnd+zarr",
                title="Stream Zarr Data",
                description=zarr_desc,
                roles=["data"],
                extra_fields={
                    "requires": ["oauth2"],
                    "authentication": {
                        "type": "oauth2",
                        "description": (
                            "Authentication using your Freva credentials is required."
                        ),
                    },
                },
            ),
            "local-access": Asset(
                href=(
                    f"{self.assets_prereqs.get('base_url')}api/freva-nextgen/"
                    f"databrowser/data-search/{self.translator.flavour}/"
                    f"{self.uniq_key}?"
                    f"{api_params}"
                    f"&{self.uniq_key}={id}"
                ),
                title="Access data locally",
                description=local_access_desc,
                roles=["data"],
                media_type="application/netcdf"
            ),
        }
        for key, asset in assets.items():
            item.add_asset(key, asset)

        return item

    def _create_stac_collection(self) -> Generator[str, None, None]:

        intake_desc = dedent(
            f"""
            # Installing Intake-ESM
            ```bash
            # Method 1: Using pip
            pip install intake-esm
            # Method 2: Using conda (recommended)
            conda install -c conda-forge intake-esm
            ```
            # Quick Guide: INTAKE-ESM Catalog on Levante (Python)
            ```python
            import intake
            # create a catalog object from a EMS JSON file containing dataset metadata
            cat = intake.open_esm_datastore(
            '{str(self.assets_prereqs.get('full_endpoint')).replace(
                "stac-catalogue", "intake-catalogue")}')
            ```
            """
        )
        stac_static_desc = dedent(
            """
            # STAC Static Catalog Setup
            ```bash
            pip install pystac
            ```
            # Load the STAC Catalog
            ```python
            import pystac, tempfile, os, subprocess
            temp_dir = tempfile.mkdtemp()
            subprocess.run(['unzip', '-o',
                            'stac-catalog-Dataset-freva-52f66bb4-a8f-file.zip',
                            '-d', temp_dir],
                        stderr=subprocess.PIPE)
            catalog = pystac.Catalog.from_file(
                os.path.join(temp_dir, 'stac-catalog/catalog.json')
                )
            print(catalog.describe())
            ```
            Also one can setup a web server to serve the static catalog.
            💡: This has been also desigend to work with the data locally. So you
            can copy the catalog link from here and download and load the catalog
            locally via the provided script.
            """
        )

        params_dict = (
            ast.literal_eval(str(self.assets_prereqs.get("only_params")))
            if self.assets_prereqs.get("only_params", "")
            else {}
        )
        python_params = " ".join(
            f"{k}='{v}','"
            for k, v in params_dict.items()
            if k not in ("translate")
        )
        cli_params = " ".join(
            f"{k}={v}" for k, v in params_dict.items()
            if k not in ("translate")
        )
        api_params = "&".join(
            f"{k}={v}" for k, v in params_dict.items()
            if k not in ("translate")
        )

        zarr_desc = dedent(
            f"""
            # Accessing Zarr Data
            1. Install freva-client
            ```bash
            pip install freva-client
            ```
            2. Get the auth token and access the zarr data (Python) - recommended
            ```python
            from freva_client import authenticate, databrowser
            import xarray as xr
            token_info = authenticate(username=<your_username>,\\
                                    host='{self.config.proxy}')
            db = databrowser({python_params} stream_zarr=True,\\
                                host='{self.config.proxy}')
            xarray_dataset = xr.open_mfdataset(list(db))
            ```
            3. Get the auth token and access the zarr data (CLI)
            ```bash
            # Attention: jq has to be installed beforehand
            token=$(freva-client auth -u <username> --host {self.config.proxy}\\
                                                        |jq -r .access_token)
            freva-client databrowser data-search {cli_params} --zarr\\
                            --host {self.config.proxy} --access-token $token
            ```
            4. Access the zarr data directly (API - language agnostic)
            ```bash
            curl -X GET {self.assets_prereqs.get('base_url')}api/ \\
            freva-nextgen/databrowser/load/\\
            {self.translator.flavour}?{api_params} \\
            -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
            ```
            💡: Read more about the
            [freva-client](https://freva-clint.github.io/freva-nextgen/)
            """
        )

        local_access_desc = dedent(
            f"""
            # Accessing data locally where the data is stored

            ```python
            import xarray as xr
            ds = xr.open_mfdataset(list('{str(self.assets_prereqs.get("full_endpoint"))
                                          .replace("stac-catalogue", "data-search")}'))
            ```

            💡: Please make sure to have the required xarray packages installed.
            """
        )
        collection_spatial = [[
            self.spatial_extent["minx"],
            self.spatial_extent["miny"],
            self.spatial_extent["maxx"],
            self.spatial_extent["maxy"],
        ]]
        start_time = None
        if self.temporal_extent["start"]:
            start_time = self.temporal_extent["start"].isoformat() + "Z"

        end_time = None
        if self.temporal_extent["end"]:
            end_time = self.temporal_extent["end"].isoformat() + "Z"

        collection_temporal = f'[["{start_time}", "{end_time}"]]'
        collection_prefix = dedent(
            f"""
            {{
                "type": "Collection",
                "id": "{self.collection_id}",
                "description": "Static STAC collection for Freva databrowser search",
                "stac_version": "1.1.0",
                "license": "other",
                "links": [
                    {{"rel": "root",
                    "href": "../../catalog.json",
                    "type": "application/json"}},
                    {{"rel": "parent",
                    "href": "../../catalog.json",
                    "type": "application/json"}}
            """
        )
        yield collection_prefix

        for id_num in range(int(self.count)):
            link = {
                "rel": "item",
                "href": f"./items/item{str(id_num)}.json",
                "type": "application/json"
            }
            link_chunk = f', {json.dumps(link)}'
            yield link_chunk

        assets = {
            "freva-databrowser": Asset(
                href=(
                    f"{self.assets_prereqs.get('base_url')}databrowser/?"
                    f"{api_params}"
                ),
                title="Freva Data-Browser",
                description=(
                    "Interactive web interface for data exploration and analysis. "
                    "Access through any browser."
                ),
                roles=["overview"],
                media_type="text/html",
            ),
            "intake-catalogue": Asset(
                href=str(self.assets_prereqs.get("full_endpoint")).replace(
                    "stac-catalogue", "intake-catalogue"
                ),
                title="Intake-ESM Catalogue",
                description=intake_desc,
                roles=["metadata"],
                media_type="application/json",
            ),
            "stac-static-catalogue": Asset(
                href=str(self.assets_prereqs.get("full_endpoint")),
                title="STAC Static Catalogue",
                description=stac_static_desc,
                roles=["metadata"],
                media_type="application/zip",
            ),
            "local-access": Asset(
                href=str(self.assets_prereqs.get("full_endpoint")).replace(
                    "stac-catalogue", "data-search"
                ),
                title="Access data locally",
                description=local_access_desc,
                roles=["data"],
                media_type="application/netcdf",
            ),
            "zarr-access": Asset(
                href=(
                    f"{self.assets_prereqs.get('base_url')}api/freva-nextgen/"
                    f"databrowser/load/{self.translator.flavour}?"
                    f"{api_params}"
                ),
                title="Stream Zarr Dataset",
                description=zarr_desc,
                roles=["data"],
                media_type="application/vnd+zarr",
                extra_fields={
                    "requires": ["oauth2"],
                    "authentication": {
                        "type": "oauth2",
                        "description": (
                            "Authentication using your Freva credentials is required."
                        ),
                    },
                },
            ),
        }
        assets_json = json.dumps(
            {key: asset.to_dict() for key, asset in assets.items()}
        )

        extra_fields_json = "{}"
        if hasattr(self, "facets") and self.facets:
            extra_fields_json = json.dumps(
                {"search_keys": {key: values for key, values in self.facets.items()}}
            )

        providers_json = json.dumps(
            [{"name": "Freva DataBrowser", "url": self.config.proxy}]
        )

        collection_suffix = dedent(
            f"""
                ],
                "extent": {{
                    "spatial": {{
                        "bbox": {collection_spatial}
                    }},
                    "temporal": {{
                        "interval": {collection_temporal}
                    }}
                }},
                "assets": {assets_json},
                "extra_fields": {extra_fields_json},
                "providers": {providers_json}
            }}
            """
        )
        yield collection_suffix

    async def _add_to_zip(
        self,
        filename: str,
        content: Union[
            str,
            Dict[str, Any],
            bytes,
            Generator[str, None, None]
        ]
    ) -> AsyncIterator[Union[bytes, str]]:
        """
        Add content to the zip file on the fly and yield chunks as they
        are available.
        Attention: one write handle is open at a time.

        Parameters:
        -----------
        filename: str
            The filename within the zip
        content: Union[str, dict, bytes, Asyncgenerator]
            Content to write - can be string, dict, bytes or a generator
        """
        # Create zip info for the zip file
        info = ZipInfo(filename=filename)
        ts = datetime.now()
        info.date_time = ts.timetuple()[:6]
        info.compress_type = ZIP_DEFLATED

        # Open the zip file entry for writing
        with self._zip.open(info, mode="w") as fp:
            if (hasattr(content, '__iter__')
                    and not isinstance(content, (str, bytes, dict))):
                # 1. When we have a generator (streaming)
                for chunk in content:
                    if isinstance(chunk, str):
                        # !: We need to encode strings before writing
                        chunk = chunk.encode("utf-8")  # type: ignore
                    fp.write(chunk)  # type: ignore
                    out_chunk = self._zip_stream.flush_and_read()
                    if out_chunk:
                        yield out_chunk
            else:
                # 2. When we have a single write
                if isinstance(content, dict):
                    content = json.dumps(content, indent=2)
                if isinstance(content, str):
                    content = content.encode("utf-8")
                fp.write(content)  # type: ignore
                chunk = self._zip_stream.flush_and_read()  # type: ignore
                if chunk:
                    yield chunk

    async def stream_stac_catalogue(
        self,
        collection_id: str,
        count: int
    ) -> AsyncIterator[Union[str, bytes]]:
        """
        Stream a complete STAC catalogue with all components.
        """
        self.collection_id = collection_id
        self.count = count
        try:
            # 1. Create and add catalog.json
            catalog = {
                "type": "Catalog",
                "stac_version": "1.0.0",
                "id": "static-catalog",
                "description": "Static STAC catalog for Freva databrowser search",
                "links": [
                    {
                        "rel": "root",
                        "href": "./catalog.json",
                        "type": "application/json"
                    },
                    {
                        "rel": "child",
                        "href": f"./collections/{collection_id}/collection.json",
                        "type": "application/json"
                    },
                ]
            }
            async for chunk in self._add_to_zip("stac-catalog/catalog.json", catalog):
                yield chunk

            # 2. Process and add items one by one - The reason that we have
            # first items, and then streaming collection, is to ensure that
            # the collection has all the updated spatial and temporal extents
            # from the items.
            self.count_item = 0  # to be linked to the collection
            async for item_batch in self._iter_stac_items():
                for item in item_batch:
                    item_dict = item.to_dict()
                    async for chunk in self._add_to_zip(
                        f"stac-catalog/collections/{collection_id}/items/"
                        f"{item_dict['id']}.json",
                        item_dict
                    ):
                        yield chunk

            # 3. Create and add collection.json using the _create_stac_collection
            async for chunk in self._add_to_zip(
                f"stac-catalog/collections/{collection_id}/collection.json",
                self._create_stac_collection()
            ):
                yield chunk

            # 4. Close the zip file and yield any remaining data
            self._zip.close()
            final_chunk = self._zip_stream.flush_and_read()
            if final_chunk:
                yield final_chunk

        except Exception as e:  # pragma: no cover
            logger.error(f"STAC collection creation failed for {collection_id}: {e}")
            try:
                self._zip.close()
            except Exception:
                pass
            raise


class Item:
    """ Item class which is compatible with pySTAC
     Item object """
    def __init__(
        self,
        id: str,
        collection: str,
        geometry: Optional[Dict[str, Any]],
        properties: Dict[str, Any],
        bbox: Optional[List[float]] = None,
    ):
        self.id = id
        self.collection = collection
        self.geometry = geometry
        self.properties = properties
        self.bbox = bbox
        self.stac_version = "1.1.0"
        self.stac_extensions: List[Any] = []
        self.links: List[Link] = []
        self.assets: Dict[str, Asset] = {}

    def add_link(self, link: "Link") -> None:
        self.links.append(link)

    def add_asset(self, key: str, asset: "Asset") -> None:
        self.assets[key] = asset

    def to_dict(self) -> Dict[str, Any]:
        item = {
            "type": "Feature",
            "stac_version": self.stac_version,
            "stac_extensions": self.stac_extensions,
            "id": self.id,
            "geometry": self.geometry,
            "properties": self.properties,
            "links": [link.to_dict() for link in self.links],
            "assets": {k: asset.to_dict() for k, asset in self.assets.items()},
            "collection": self.collection,
        }
        if self.bbox:
            item["bbox"] = self.bbox
        return item


class Link:
    """ Link class which is compatible with pySTAC
        Link object """
    def __init__(
        self,
        rel: str,
        href: str,
        type: str,
        noresolve: bool = False,
        extra_fields: Optional[Dict[str, Any]] = None,
    ):
        self.rel = rel
        self.href = href
        self.type = type
        self.noresolve = noresolve
        self.extra_fields = extra_fields or {}

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "rel": self.rel,
            "href": self.href,
            "type": self.type,
        }
        if self.noresolve:  # pragma: no cover
            d["noresolve"] = True
        d.update(self.extra_fields)
        return d


class Asset:
    """ Asset class which is compatible with pySTAC
        Asset object """
    def __init__(
        self,
        href: str,
        title: str,
        description: str,
        roles: List[str],
        media_type: str,
        extra_fields: Optional[Dict[str, Any]] = None,
    ):
        self.href = href
        self.title = title
        self.description = description
        self.roles = roles
        self.media_type = media_type
        self.extra_fields = extra_fields or {}

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "href": self.href,
            "title": self.title,
            "description": self.description,
            "roles": self.roles,
            "type": self.media_type,
        }
        d.update(self.extra_fields)
        return d
