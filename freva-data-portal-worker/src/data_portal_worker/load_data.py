"""Load backend for reading different datasets."""

import json
import os
import ssl
import time
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple, TypedDict, cast

import cloudpickle
import redis
import xarray as xr
from dask.distributed import Client, LocalCluster
from xarray.backends.zarr import encode_zarr_variable

from .backends import load_data
from .utils import data_logger, str_to_int
from .zarr_utils import (
    create_zmetadata,
    encode_chunk,
    get_data_chunk,
    jsonify_zmetadata,
)

ZARR_CONSOLIDATED_FORMAT = 1
ZARR_FORMAT = 2
ZARRAY_JSON = ".zarray"

CLIENT: Optional[Client] = None
LoadDict = TypedDict(
    "LoadDict",
    {
        "status": Literal[0, 1, 2, 3],
        "url": str,
        "obj_path": str,
        "reason": str,
        "meta": Optional[Dict[str, Any]],
        "json_meta": Optional[Dict[str, Any]],
    },
)
RedisKw = TypedDict(
    "RedisKw",
    {
        "user": str,
        "passwd": str,
        "host": str,
        "port": int,
        "ssl_cert": str,
        "ssl_key": str,
    },
)


class RedisCacheFactory(redis.Redis):
    """Define a custom redis cache."""

    def __init__(self, db: int = 0) -> None:
        host, _, port = (
            (os.environ.get("API_REDIS_HOST") or "localhost")
            .replace("redis://", "")
            .partition(":")
        )
        port_i = int(port or "6379")
        conn = {
            "host": host,
            "port": port_i,
            "db": db,
            "username": os.getenv("API_REDIS_USER") or None,
            "password": os.getenv("API_REDIS_PASSWORD") or None,
            "ssl_certfile": os.getenv("API_REDIS_SSL_CERTFILE") or None,
            "ssl_keyfile": os.getenv("API_REDIS_SSL_KEYFILE") or None,
            "ssl_ca_certs": os.getenv("API_REDIS_SSL_CERTFILE") or None,
        }
        conn["ssl"] = conn["ssl_certfile"] is not None
        data_logger.info("Creating redis connection with args: %s", conn)
        super().__init__(
            host=conn["host"],
            port=conn["port"],
            db=conn["db"],
            username=conn["username"],
            password=conn["password"],
            ssl=conn["ssl"],
            ssl_certfile=conn["ssl_certfile"],
            ssl_keyfile=conn["ssl_keyfile"],
            ssl_ca_certs=conn["ssl_ca_certs"],
            ssl_cert_reqs=ssl.CERT_NONE,
        )


@dataclass
class LoadStatus:
    """Schema defining the status of loading dataset."""

    status: Literal[0, 1, 2, 3]
    """Status of the submitted jobs:
        0: exit success
        1: exit failed
        2: in queue (submitted)
        3: in progress
    """
    obj_path: str
    """url of the zarr dataset once finished."""
    reason: str
    """if status = 1 reasone why opening the dataset failed."""
    meta: Optional[Dict[str, Any]] = None
    """Meta data of the zarr store"""
    url: str = ""
    """Url of the machine that loads the zarr store."""
    json_meta: Optional[Dict[str, Any]] = None
    """Json representation of the zarr metadata."""

    def dict(self) -> LoadDict:
        """Convert object to dict."""
        return {
            "status": self.status,
            "obj_path": self.obj_path,
            "reason": self.reason,
            "meta": self.meta,
            "url": self.url,
            "json_meta": self.json_meta,
        }

    @classmethod
    def from_dict(cls, load_dict: Optional[LoadDict]) -> "LoadStatus":
        """Create an instance of the class from a normal python dict."""
        _dict = load_dict or {
            "status": 2,
            "reason": "",
            "url": "",
            "obj_path": "",
            "meta": None,
            "json_meta": None,
        }
        return cls(**_dict)


def get_dask_client(
    client: Optional[Client] = CLIENT, dev_mode: bool = False
) -> Optional[Client]:
    """Get or create a cached dask cluster."""
    if client is None and dev_mode is False:
        client = Client(
            LocalCluster(
                host="0.0.0.0",
                scheduler_port=int(os.getenv("DASK_PORT", "40000")),
            )
        )
        data_logger.info("Created new cluster: %s", client.dashboard_link)
    else:
        data_logger.debug("recycling dask cluster.")
    return client


class DataLoadFactory:
    """Class to load data object and convert them to zarr.

    The class defines different staticmethods that load datasets for different
    storage systems.

    Currently implemented are:
        from_posix: loads a dataset from a posix file system

    Parameters
    ----------
    scheme: str
    the url prefix of the object path that holds the
    dataset. For example the scheme of hsm://arch/foo.nc would
    indicate that the data is stored on an hsm tape archive.
    A posix file system if assumed if scheme is empty, e.g /home/bar/foo.nc
    """

    def __init__(self) -> None:
        self._cache: Optional[RedisCacheFactory] = None

    @property
    def cache(self) -> RedisCacheFactory:
        """Get or create the cache."""
        if self._cache is None:
            self._cache = RedisCacheFactory()
        return self._cache

    def from_object_path(
        self,
        input_path: str,
        path_id: str,
    ) -> None:
        """Create a zarr object from an input path."""
        status_dict = LoadStatus.from_dict(
            cast(
                Optional[LoadDict],
                cloudpickle.loads(self.cache.get(path_id)),
            )
        ).dict()
        expires_in = str_to_int(os.environ.get("API_CACHE_EXP"), 3600)
        status_dict["status"] = 3
        self.cache.setex(path_id, expires_in, cloudpickle.dumps(status_dict))
        data_logger.debug("Reading %s", input_path)
        try:
            data_logger.info(input_path)
            dset = load_data(input_path)
            metadata = create_zmetadata(dset)
            status_dict["json_meta"] = jsonify_zmetadata(dset, metadata)
            status_dict["meta"] = metadata
            status_dict["status"] = 0
            # We need to add the xarray to an extra cache entry because the
            # status_dict will be loaded by the rest-api, if the xarray dataset
            # is present the rest-api code will attempt to instanciate the
            # pickled dataset object and that might fail because we might or
            # might not have xarray and all the backends instanciated.
            # Since the xarray dataset object isn't needed anyway byt the
            # rest-api we simply add it to a cache entry of its own.
            self.cache.setex(
                f"{path_id}-dset", expires_in, cloudpickle.dumps(dset)
            )
        except Exception as error:
            data_logger.exception("Could not process %s: %s", path_id, error)
            status_dict["status"] = 1
            status_dict["reason"] = str(error)
        self.cache.setex(
            path_id,
            expires_in,
            cloudpickle.dumps(status_dict),
        )

    def get_zarr_chunk(
        self,
        key: str,
        chunk: str,
        variable: str,
    ) -> None:
        """Read the zarr metadata from the cache."""
        pickle_data, dset = self.load_object(key)
        meta = cast(Dict[str, Any], pickle_data["meta"])
        arr_meta = meta["metadata"][f"{variable}/{ZARRAY_JSON}"]
        data = encode_chunk(
            get_data_chunk(
                encode_zarr_variable(
                    dset.variables[variable], name=variable
                ).data,
                chunk,
                out_shape=arr_meta["chunks"],
            ).tobytes(),
            filters=arr_meta["filters"],
            compressor=arr_meta["compressor"],
        )
        self.cache.setex(f"{key}-{variable}-{chunk}", 360, data)

    def load_object(self, key: str) -> Tuple[LoadDict, xr.Dataset]:
        """Load a cached dataset.

        Parameters
        ----------
        key: str, The cache key.

        Returns
        -------
        The data that was stored under that key.

        Raises
        ------
        RuntimeError: If the cache key exists but the data could not be loaded,
                      which means that there is a load status != 0
        KeyError: If the key doesn't exist in the cache (anymore).
        """
        metadata_cache = self.cache.get(key)
        dset_cache = self.cache.get(f"{key}-dset")
        if metadata_cache is None or dset_cache is None:
            raise KeyError(f"{key} uuid does not exist (anymore).")
        load_dict = cast(LoadDict, cloudpickle.loads(metadata_cache))
        dset = cast(xr.Dataset, cloudpickle.loads(dset_cache))
        return load_dict, dset


class ProcessQueue(DataLoadFactory):
    """Class that can load datasets on different object stores."""

    def __init__(self, dev_mode: bool = False) -> None:
        super().__init__()
        self.client = get_dask_client(dev_mode=dev_mode)
        self.dev_mode = dev_mode

    def run_for_ever(self, channel: str) -> None:
        """Start the listener daemon."""
        data_logger.info("Starting data-loading daemon")
        pubsub = self.cache.pubsub()
        pubsub.subscribe(channel)
        data_logger.info("Broker will listen for messages now")
        while True:
            message = pubsub.get_message()
            if message and message["type"] == "message":
                try:
                    self.redis_callback(message["data"])
                except KeyboardInterrupt:
                    raise
                except Exception as error:
                    data_logger.exception(error)
            else:
                time.sleep(0.1)

    def redis_callback(
        self,
        body: bytes,
    ) -> None:
        """Callback method to receive rabbit mq messages."""
        try:
            message = json.loads(body)
        except json.JSONDecodeError:
            data_logger.warning("could not decode message")
            return
        if "uri" in message:
            self.spawn(message["uri"]["path"], message["uri"]["uuid"])
        elif "chunk" in message:
            self.get_zarr_chunk(
                message["chunk"]["uuid"],
                message["chunk"]["chunk"],
                message["chunk"]["variable"],
            )
        elif "shutdown" in message:
            if message["shutdown"] is True and self.dev_mode is True:
                raise KeyboardInterrupt("Shutdown client")

    def spawn(self, inp_obj: str, uuid5: str) -> str:
        """Subumit a new data loading task to the process pool."""
        data_logger.debug(
            "Assigning %s to %s for future processing", inp_obj, uuid5
        )
        data_cache: Optional[bytes] = cast(Optional[bytes], self.cache.get(uuid5))
        status_dict: LoadDict = {
            "status": 2,
            "obj_path": f"/api/freva-data-portal/zarr/{uuid5}.zarr",
            "reason": "",
            "url": "",
            "meta": None,
            "json_meta": None,
        }
        if data_cache is None:
            self.cache.setex(
                uuid5,
                str_to_int(os.environ.get("API_CACHE_EXP"), 3600),
                cloudpickle.dumps(status_dict),
            )
            self.from_object_path(inp_obj, uuid5)
        else:
            status_dict = cast(LoadDict, cloudpickle.loads(data_cache))
            if status_dict["status"] in (1, 2):
                # Failed job, let's retry
                self.from_object_path(inp_obj, uuid5)

        return status_dict["obj_path"]
