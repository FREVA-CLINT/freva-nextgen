"""Test various utilities."""

import os.path
from tempfile import TemporaryDirectory

import netCDF4
import numpy as np
import rasterio
import zarr
from data_portal_worker.backends.posix import get_xr_engine
from data_portal_worker.utils import str_to_int as str_to_int2
from freva_rest.utils import str_to_int


def create_netcdf4_file(temp_dir: str) -> str:
    """Create a netcdf file."""
    temp_file = os.path.join(temp_dir, "out.nc")
    with netCDF4.Dataset(temp_file, "w") as dataset:
        dataset.createDimension("dim", 4)
        var = dataset.createVariable("var", "f4", ("dim",))
        var[:] = [1, 2, 3, 4]
    return temp_file


def create_rasterio_file(temp_dir: str) -> str:
    """Create a geotif file."""
    temp_file = os.path.join(temp_dir, "out.tiff")
    with rasterio.open(
        temp_file,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=1,
        dtype="uint8",
    ) as dataset:
        array = np.zeros((10, 10), dtype=np.uint8)
        dataset.write(array, 1)
    return temp_file


def create_zarr_file(temp_dir: str) -> str:
    """Create a zarr dataset."""
    temp = os.path.join(temp_dir, "out.zarr")
    zarr.convenience.save(temp, [1, 2, 3, 4])
    return temp


def test_str_to_int() -> None:
    """Test str to int utility."""
    for func in (str_to_int, str_to_int2):
        assert func(None, 3) == 3
        assert func("a", 3) == 3
        assert func("4", 3) == 4


def test_get_xr_posix_engine() -> None:
    """Test the right xarray engine."""
    with TemporaryDirectory() as temp_dir:
        assert get_xr_engine(create_netcdf4_file(temp_dir)) == "netcdf4"
        assert get_xr_engine(create_rasterio_file(temp_dir)) == "rasterio"
        assert get_xr_engine(create_zarr_file(temp_dir)) == "zarr"
    with TemporaryDirectory() as temp_dir:
        assert get_xr_engine(temp_dir) is None
