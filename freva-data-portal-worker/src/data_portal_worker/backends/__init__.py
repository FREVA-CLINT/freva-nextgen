"""Load data files."""

from urllib.parse import urlparse

import xarray as xr

from .posix import load_posix


def load_data(inp_path: str) -> xr.Dataset:
    """Open a datasets."""

    parsed_url = urlparse(inp_path)

    implemented_methods = {"file": load_posix, "": load_posix}
    return implemented_methods[parsed_url.scheme](inp_path)
