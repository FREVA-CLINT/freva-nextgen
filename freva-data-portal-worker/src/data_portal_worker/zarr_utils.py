"""Utilities for working with zarr storages."""

from copy import deepcopy
from typing import Any, Dict, Optional, Union, cast

import dask.array
import numpy as np
import xarray as xr
from numcodecs.abc import Codec
from numcodecs.compat import ensure_ndarray
from packaging.version import Version
from xarray.backends.zarr import (
    DIMENSION_KEY,
    encode_zarr_attr_value,
    encode_zarr_variable,
    extract_zarr_variable_encoding,
)
from zarr.meta import encode_fill_value
from zarr.storage import (
    array_meta_key,
    attrs_key,
    default_compressor,
    group_meta_key,
)
from zarr.util import normalize_shape

from .utils import data_logger

DaskArrayType = dask.array.Array
ZARR_FORMAT = 2
ZARR_CONSOLIDATED_FORMAT = 1
ZARR_METADATA_KEY = ".zmetadata"


def extract_dataarray_zattrs(da: xr.DataArray) -> Dict[str, Any]:
    """helper function to extract zattrs dictionary from DataArray"""
    zattrs = {}
    for k, v in da.attrs.items():
        zattrs[k] = encode_zarr_attr_value(v)
    zattrs[DIMENSION_KEY] = list(da.dims)

    # We don't want `_FillValue` in `.zattrs`
    # It should go in `fill_value` section of `.zarray`
    _ = zattrs.pop("_FillValue", None)

    return zattrs


def extract_dataset_zattrs(dataset: xr.Dataset) -> Dict[str, Any]:
    """helper function to create zattrs dictionary from Dataset global attrs"""
    zattrs = {}
    for k, v in dataset.attrs.items():
        zattrs[k] = encode_zarr_attr_value(v)
    return zattrs


def extract_dataarray_coords(
    da: xr.DataArray,
    zattrs: Dict[str, Any],
) -> Dict[str, Any]:
    """helper function to extract coords from DataArray into a directionary"""
    if da.coords:
        # Coordinates are only encoded if there are non-dimension coordinates
        nondim_coords = sorted(str(k) for k in (set(da.coords) - set(da.dims)))

        if len(nondim_coords) > 0 and da.name not in nondim_coords:
            coords = " ".join(nondim_coords)
            zattrs["coordinates"] = encode_zarr_attr_value(coords)
    return zattrs


def extract_zarray(
    da: xr.DataArray,
    encoding: Dict[str, Any],
    dtype: np.dtype[Any],
) -> Dict[str, Any]:
    """helper function to extract zarr array metadata."""

    def _extract_fill_value(
        da: xr.DataArray,
        dtype: np.dtype[Any],
    ) -> Any:
        """helper function to extract fill value from DataArray."""
        fill_value = da.attrs.pop("_FillValue", None)
        return encode_fill_value(fill_value, dtype)

    meta = {
        "compressor": encoding.get(
            "compressor", da.encoding.get("compressor", default_compressor)
        ),
        "filters": encoding.get("filters", da.encoding.get("filters", None)),
        "chunks": encoding.get("chunks", None),
        "dtype": dtype.str,
        "fill_value": _extract_fill_value(da, dtype),
        "order": "C",
        "shape": list(normalize_shape(da.shape)),
        "zarr_format": ZARR_FORMAT,
    }

    if meta["chunks"] is None:
        meta["chunks"] = da.shape

    # validate chunks
    if isinstance(da.data, DaskArrayType):
        var_chunks = tuple([c[0] for c in da.data.chunks])
    else:
        var_chunks = da.shape
    if not var_chunks == tuple(meta["chunks"]):
        raise ValueError(
            "Encoding chunks do not match inferred chunks"
        )  # pragma: no cover

    meta["chunks"] = list(meta["chunks"])  # return chunks as a list

    return meta


def create_zmetadata(dataset: xr.Dataset) -> Dict[str, Any]:
    """Helper function to create a consolidated zmetadata dictionary."""

    zmeta: Dict[str, Any] = {
        "zarr_consolidated_format": ZARR_CONSOLIDATED_FORMAT,
        "metadata": {},
    }
    zmeta["metadata"][group_meta_key] = {"zarr_format": ZARR_FORMAT}
    zmeta["metadata"][attrs_key] = extract_dataset_zattrs(dataset)
    extra_kw = {}
    if Version(xr.__version__) >= Version("2025.3.0"):
        extra_kw["zarr_format"] = ZARR_FORMAT  # pragma: no cover
    for key, dvar in dataset.variables.items():
        da = dataset[key]
        encoded_da = encode_zarr_variable(dvar, name=key)
        encoding = extract_zarr_variable_encoding(dvar, **extra_kw)  # type: ignore
        zattrs = extract_dataarray_zattrs(encoded_da)
        zattrs = extract_dataarray_coords(da, zattrs)
        zmeta["metadata"][f"{key}/{attrs_key}"] = zattrs
        zmeta["metadata"][f"{key}/{array_meta_key}"] = extract_zarray(
            encoded_da,
            encoding,
            encoded_da.dtype,
        )

    return zmeta


def jsonify_zmetadata(
    dataset: xr.Dataset,
    zmetadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Helper function to convert zmetadata dictionary to a json
    compatible dictionary.

    """
    zjson = deepcopy(zmetadata)

    for key in list(dataset.variables):
        # convert compressor to dict
        compressor = zjson["metadata"][f"{key}/{array_meta_key}"]["compressor"]
        if compressor is not None:
            compressor_config = zjson["metadata"][f"{key}/{array_meta_key}"][
                "compressor"
            ].get_config()
            zjson["metadata"][f"{key}/{array_meta_key}"][
                "compressor"
            ] = compressor_config

    return zjson


def encode_chunk(
    chunk: np.typing.ArrayLike,
    filters: Optional[list[Codec]] = None,
    compressor: Optional[Codec] = None,
) -> bytes:
    """helper function largely copied from zarr.Array"""
    # apply filters
    if filters:
        for f in filters:
            chunk = f.encode(chunk)

    # check object encoding
    if ensure_ndarray(chunk).dtype == object:
        raise RuntimeError("cannot write object array without object codec")

    # compress
    if compressor:
        cdata = compressor.encode(chunk)
    else:
        cdata = ensure_ndarray(chunk).tobytes()

    return cast(bytes, cdata)


def get_data_chunk(
    da: Union[xr.DataArray, DaskArrayType],
    chunk_id: str,
    out_shape: tuple[int, ...],
) -> np.typing.NDArray[Any]:
    """Get one chunk of data from this DataArray (da).

    If this is an incomplete edge chunk, pad the returned array to match out_shape.
    """
    ikeys = tuple(map(int, chunk_id.split(".")))
    if isinstance(da, DaskArrayType):
        chunk_data = da.blocks[ikeys]
    else:
        if da.ndim > 0 and ikeys != ((0,) * da.ndim):
            raise ValueError(
                "Invalid chunk_id for numpy array: %s. Should have been: %s"
                % (chunk_id, ((0,) * da.ndim))
            )
        chunk_data = np.asarray(da)

    data_logger.debug(
        "checking chunk output size, %s == %s" % (chunk_data.shape, out_shape)
    )

    if isinstance(chunk_data, DaskArrayType):
        chunk_data = chunk_data.compute()

    # zarr expects full edge chunks, contents out of bounds for the array are undefined
    if chunk_data.shape != tuple(out_shape):
        new_chunk = np.empty_like(chunk_data, shape=out_shape)
        write_slice = tuple([slice(0, s) for s in chunk_data.shape])
        new_chunk[write_slice] = chunk_data
        return cast(np.typing.NDArray[Any], new_chunk)
    else:
        return cast(np.typing.NDArray[Any], chunk_data)
