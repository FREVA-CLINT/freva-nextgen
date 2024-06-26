"""Load data from a posix file system."""

from pathlib import Path
from typing import Optional, Union

import netCDF4

# import cfgrib
import rasterio
import xarray as xr
import zarr


def get_xr_engine(file_path: str) -> Optional[str]:
    """Get the engine, to open the xarray dataset."""
    try:
        with netCDF4.Dataset(file_path, mode="r"):
            return "netcdf4"
    except Exception:
        pass

    # try:
    #    with cfgrib.open_file(file_path):
    #        return "cfgrib"
    # except:
    #    pass
    try:
        _ = zarr.open(file_path, mode="r")
        return "zarr"
    except Exception:
        pass

    try:
        with rasterio.open(file_path, mode="r"):
            return "rasterio"
    except Exception:
        pass

    return None


def load_posix(inp_file: Union[str, Path]) -> xr.Dataset:
    """Open a dataset with xarray."""
    inp_file = Path(inp_file)
    return xr.open_dataset(
        inp_file,
        decode_cf=False,
        use_cftime=False,
        chunks="auto",
        cache=False,
        decode_coords=False,
        engine=get_xr_engine(str(inp_file)),
    )
