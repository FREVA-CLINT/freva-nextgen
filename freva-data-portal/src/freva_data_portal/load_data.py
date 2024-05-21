"""Load backend for reading different datasets."""

from dataclasses import dataclass
import multiprocessing as mp
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union, cast

import cloudpickle  # fades cloudpickle
from dask.distributed import SSHCluster  # fades distributed
from dask.distributed import Client
from dask.distributed.deploy.cluster import Cluster
import pika  # fades pika
import redis  # fades redis
import xarray as xr  # fades xarray
from xarray.backends.zarr import encode_zarr_variable
from xpublish.utils.zarr import (
    create_zmetadata,
    jsonify_zmetadata,
)  # fades xpublish
from xpublish.utils.zarr import get_data_chunk, encode_chunk
import zarr  # fades zarr
from zarr.meta import encode_array_metadata, encode_group_metadata
from zarr.storage import array_meta_key

from .utils import data_logger, str_to_int, CLIENT
from .backends import load_data

ZARR_CONSOLIDATED_FORMAT = 1
ZARR_FORMAT = 2

CLIENT: Optional[Client] = None
LoadDict = TypedDict(
    "LoadDict",
    {
        "status": Literal[0, 1, 2, 3],
        "url": str,
        "obj": Optional[xr.Dataset],
        "obj_path": str,
        "reason": str,
        "meta": Optional[Dict[str, Any]],
        "json_meta": Optional[Dict[str, Any]],
    },
)
ClusterKw = TypedDict(
    "ClusterKw",
    {
        "hosts": List[str],
        "connect_options": List[Dict[str, str]],
    },
)
BrokerKw = TypedDict(
    "BrokerKw", {"user": str, "passwd": str, "host": str, "port": int}
)

DataLoaderConfig = TypedDict(
    "DataLoaderConfig", {"ssh_config": ClusterKw, "broker_config": BrokerKw}
)


class RedisCacheFactory(redis.Redis):
    """Define a custom redis cache."""

    def __init__(self, db: int = 0) -> None:
        host, _, port = (
            (os.environ.get("REDIS_HOST") or "localhost")
            .replace("redis://", "")
            .partition(":")
        )
        port_i = int(port or "6379")
        super().__init__(host=host, port=port_i, db=db)


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
    obj: Optional[xr.Dataset]
    """pickled memory view of the opened dataset."""
    reason: str
    """if status = 1 reasone why opening the dataset failed."""
    meta: Optional[Dict[str, Any]] = None
    """Meta data of the zarr store"""
    url: str = ""
    """Url of the machine that loads the zarr store."""
    json_meta: Optional[Dict[str, Any]] = None
    """Json representation of the zarr metadata."""

    @staticmethod
    def lookup(status: int) -> str:
        """Translate a status integer to a human readable status."""
        _lookup = {
            0: "finished, ok",
            1: "finished, failed",
            2: "waiting",
            3: "processing",
        }
        return _lookup.get(status, "unkown")

    def dict(self) -> LoadDict:
        """Convert object to dict."""
        return {
            "status": self.status,
            "obj": self.obj,
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
            "obj": None,
            "reason": "",
            "url": "",
            "obj_path": "",
            "meta": None,
            "json_meta": None,
        }
        return cls(**_dict)


def get_dask_client(
    cluster_kw: ClusterKw, client: Optional[Client] = CLIENT
) -> Client:
    """Get or create a cached dask cluster."""
    if client is None:
        print_kw = cluster_kw.copy()
        for num, _ in enumerate(cluster_kw["connect_options"]):
            print_kw["connect_options"][num]["password"] = "***"
        data_logger.debug("setting up ssh cluster with %s", print_kw)
        client = Client(SSHCluster(**cluster_kw))
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

    def __init__(self, cache: Optional[redis.Redis] = None) -> None:
        self.cache = cache or RedisCacheFactory(0)

    def read(self, input_path: str) -> xr.Dataset:
        """Open the dataset."""
        return load_data(inp_data)

    @staticmethod
    def from_posix(input_path: str) -> xr.Dataset:
        """Open a dataset with xarray."""

        return xr.open_dataset(
            input_path,
            decode_cf=False,
            use_cftime=False,
            chunks="auto",
            cache=False,
            decode_coords=False,
        )

    @classmethod
    def from_object_path(
        cls, input_path: str, path_id: str, cache: Optional[redis.Redis] = None
    ) -> None:
        """Create a zarr object from an input path."""
        cache = cache or RedisCacheFactory(0)
        status_dict = LoadStatus.from_dict(
            cast(
                Optional[LoadDict],
                cloudpickle.loads(cache.get(path_id)),
            )
        ).dict()
        expires_in = str_to_int(os.environ.get("API_CACHE_EXP"), 3600)
        status_dict["status"] = 3
        cache.setex(path_id, expires_in, cloudpickle.dumps(status_dict))
        data_logger.debug("Reading %s", input_path)
        try:
            data_logger.info(input_path)
            dset = load_data(input_path)
            metadata = create_zmetadata(dset)
            status_dict["json_meta"] = jsonify_zmetadata(dset, metadata)
            status_dict["obj"] = dset
            status_dict["meta"] = metadata
            status_dict["status"] = 0
        except Exception as error:
            data_logger.exception("Could not process %s: %s", path_id, error)
            status_dict["status"] = 1
            status_dict["reason"] = str(error)
        cache.setex(
            path_id,
            expires_in,
            cloudpickle.dumps(status_dict),
        )

    @classmethod
    def get_zarr_chunk(
        cls,
        key: str,
        chunk: str,
        variable: str,
        cache: Optional[redis.Redis] = None,
    ) -> None:
        """Read the zarr metadata from the cache."""
        cache = cache or RedisCacheFactory(0)
        pickle_data = cls.load_object(key, cache)
        dset = cast(xr.Dataset, pickle_data["obj"])
        meta = cast(Dict[str, Any], pickle_data["meta"])
        arr_meta = meta["metadata"][f"{variable}/{array_meta_key}"]
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
        cache.setex(f"{key}-{variable}-{chunk}", 360, data)

    @staticmethod
    def load_object(key: str, cache: Optional[redis.Redis] = None) -> LoadDict:
        """Load a cached dataset.

        Parameters
        ----------
        key: str, The cache key.
        cache: The redis cache object, if None (default) a new redis isntance
               is created.

        Returns
        -------
        The data that was stored under that key.

        Raises
        ------
        RuntimeError: If the cache key exists but the data could not be loaded,
                      which means that there is a load status != 0
        KeyError: If the key doesn't exist in the cache (anymore).
        """
        cache = cache or RedisCacheFactory(0)
        data_cache = cache.get(key)
        if data_cache is None:
            raise KeyError(f"{key} uuid does not exist (anymore).")
        pickle_data: LoadDict = cloudpickle.loads(cache)
        task_status = pickle_data.get("status", 1)
        if task_status != 0:
            raise RuntimeError(LoadStatus.lookup(task_status))
        return pickle_data


class ProcessQueue:
    """Class that can load datasets on different object stores."""

    def __init__(
        self,
        rest_url: str,
        cluster_kw: ClusterKw,
        redis_cache: Optional[redis.Redis] = None,
    ) -> None:
        self.redis_cache = redis_cache or RedisCacheFactory(0)
        self.client = get_dask_client(cluster_kw)
        self.rest_url = rest_url
        self.cluster_kw = cluster_kw

    def run_for_ever(
        self,
        queue: str,
        config: BrokerKw,
    ) -> None:
        """Start the listner deamon."""
        data_logger.info("Starting data-loading deamon")
        data_logger.debug(
            "Connecting to broker on host %s via port %i",
            config["host"],
            config["port"],
        )
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=config["host"],
                port=config["port"],
                credentials=pika.PlainCredentials(
                    username=config["user"], password=config["passwd"]
                ),
            )
        )
        self.channel = connection.channel()
        self.channel.queue_declare(queue=queue)
        self.channel.basic_consume(
            queue=queue,
            on_message_callback=self.rabbit_mq_callback,
            auto_ack=True,
        )
        data_logger.info("Broker will listen for messages now")
        self.channel.start_consuming()

    def rabbit_mq_callback(
        self,
        ch: pika.channel.Channel,
        method: pika.spec.Basic.Deliver,
        properties: pika.spec.BasicProperties,
        body: bytes,
    ) -> None:
        """Callback method to recieve rabbit mq messages."""
        try:
            message = json.loads(body)
            if "uri" in message:
                self.spawn(message["uri"]["path"], message["uri"]["uuid"])
            elif "chunk" in message:
                DataLoadFactory.get_zarr_chunk(
                    message["chunk"]["uuid"],
                    message["chunk"]["chunk"],
                    message["chunk"]["variable"],
                    self.redis_cache,
                )
        except json.JSONDecodeError:
            data_logger.warning("could not decode message")
            pass

    def spawn(self, inp_obj: str, uuid5: str) -> str:
        """Subumit a new data loading task to the process pool."""
        data_logger.debug(
            "Assigning %s to %s for future processing", inp_obj, uuid5
        )
        cache: Optional[bytes] = self.redis_cache.get(uuid5)
        status_dict: LoadDict = {
            "status": 2,
            "obj_path": f"{self.rest_url}/api/freva-data-portal/zarr/{uuid5}",
            "obj": None,
            "reason": "",
            "url": "",
            "meta": None,
            "json_meta": None,
        }
        if cache is None:
            self.redis_cache.setex(
                uuid5,
                str_to_int(os.environ.get("API_CACHE_EXP"), 3600),
                cloudpickle.dumps(status_dict),
            )
            DataLoadFactory.from_object_path(inp_obj, uuid5, self.redis_cache)
        else:
            status_dict = cast(LoadDict, cloudpickle.loads(cache))
            if status_dict["status"] in (1, 2):
                # Failed job, let's retry
                # self.client.submit(
                DataLoadFactory.from_object_path(
                    inp_obj, uuid5, self.redis_cache
                )
                # )

        return status_dict["obj_path"]
