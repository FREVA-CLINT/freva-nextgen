"""Various utilities for getting the databrowser working."""

import concurrent.futures
import os
import sys
import sysconfig
from configparser import ConfigParser, ExtendedInterpolation
from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import appdirs
import numpy as np
import requests
import tomli
import xarray as xr

from . import logger


class Config:
    """Client config class.

    This class is used for basic configuration of the databrowser
    client.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        uniq_key: Literal["file", "uri"] = "file",
        flavour: str = "freva",
    ) -> None:
        self.databrowser_url = f"{self.get_api_url(host)}/databrowser"
        self.auth_url = f"{self.get_api_url(host)}/auth/v2"
        self.uniq_key = uniq_key
        self._flavour = flavour

    def _read_ini(self, path: Path) -> str:
        """Read an ini file."""
        ini_parser = ConfigParser(interpolation=ExtendedInterpolation())
        ini_parser.read_string(path.read_text())
        config = ini_parser["evaluation_system"]
        scheme, host = self._split_url(
            config.get("databrowser.host", config.get("solr.host", ""))
        )
        host, _, port = (host or "").partition(":")
        port = port or config.get("databrowser.port", "")
        if port:
            host = f"{host}:{port}"
        return f"{scheme}://{host}"

    def _read_toml(self, path: Path) -> str:
        """Read a new style toml config file."""
        try:
            config = tomli.loads(path.read_text()).get("freva", {})
            scheme, host = self._split_url(cast(str, config["host"]))
        except (tomli.TOMLDecodeError, KeyError):
            return ""
        host, _, port = host.partition(":")
        if port:
            host = f"{host}:{port}"
        return f"{scheme}://{host}"

    def _read_config(self, path: Path, file_type: Literal["toml", "ini"]) -> str:
        """Read the configuration."""
        data_types = {"toml": self._read_toml, "ini": self._read_ini}
        try:
            return data_types[file_type](path)
        except KeyError:
            pass
        return ""

    @cached_property
    def overview(self) -> Dict[str, Any]:
        """Get an overview of the all databrowser flavours and search keys."""
        try:
            res = requests.get(f"{self.databrowser_url}/overview", timeout=15)
        except requests.exceptions.ConnectionError:
            raise ValueError(
                f"Could not connect to {self.databrowser_url}"
            ) from None
        return cast(Dict[str, Any], res.json())

    def _get_databrowser_host_from_config(self) -> str:
        """Get the config file order."""

        eval_conf = self.get_dirs(user=False) / "evaluation_system.conf"
        freva_config = Path(
            os.environ.get("FREVA_CONFIG")
            or Path(self.get_dirs(user=False)) / "freva.toml"
        )
        paths: Dict[Path, Literal["toml", "ini"]] = {
            Path(appdirs.user_config_dir("freva")) / "freva.toml": "toml",
            Path(self.get_dirs(user=True)) / "freva.toml": "toml",
            freva_config: "toml",
            Path(
                os.environ.get("EVALUATION_SYSTEM_CONFIG_FILE") or eval_conf
            ): "ini",
        }
        for config_path, config_type in paths.items():
            if config_path.is_file():
                host = self._read_config(config_path, config_type)
                if host:
                    return host
        raise ValueError(
            "No databrowser host configured, please use a"
            " configuration defining a databrowser host or"
            " set a host name using the `host` key"
        )

    @cached_property
    def flavour(self) -> str:
        """Get the flavour."""
        flavours = self.overview.get("flavours", [])
        if self._flavour not in flavours:
            raise ValueError(
                f"Search {self._flavour} not available, select from"
                f" {','.join(flavours)}"
            )
        return self._flavour

    @property
    def search_url(self) -> str:
        """Define the data search endpoint."""
        return (
            f"{self.databrowser_url}/data-search/{self.flavour}/{self.uniq_key}"
        )

    @property
    def zarr_loader_url(self) -> str:
        """Define the url for getting zarr files."""
        return f"{self.databrowser_url}/load/{self.flavour}/"

    @property
    def intake_url(self) -> str:
        """Define the url for creating intake catalogues."""
        return f"{self.databrowser_url}/intake-catalogue/{self.flavour}/{self.uniq_key}"

    @property
    def stac_url(self) -> str:
        """Define the url for creating stac catalogue."""
        return f"{self.databrowser_url}/stac-catalogue/{self.flavour}/{self.uniq_key}"

    @property
    def metadata_url(self) -> str:
        """Define the endpoint for the metadata search."""
        return (
            f"{self.databrowser_url}/metadata-search/"
            f"{self.flavour}/{self.uniq_key}"
        )

    @staticmethod
    def _split_url(url: str) -> Tuple[str, str]:
        scheme, _, hostname = url.partition("://")
        if not hostname:
            hostname = scheme
            scheme = ""
        scheme = scheme or "http"
        return scheme, hostname

    def get_api_url(self, url: Optional[str]) -> str:
        """Construct the databrowser url from a given hostname."""
        url = url or self._get_databrowser_host_from_config()
        scheme, hostname = self._split_url(url)
        hostname, _, port = hostname.partition(":")
        if port:
            hostname = f"{hostname}:{port}"
        hostname = hostname.partition("/")[0]
        return f"{scheme}://{hostname}/api/freva-nextgen"

    @staticmethod
    def get_dirs(user: bool = True) -> Path:
        """Get the 'scripts' and 'purelib' directories we'll install into.

        This is now a thin wrapper around sysconfig.get_paths(). It's not inlined,
        because some tests mock it out to install to a different location.
        """
        if user:
            if (sys.platform == "darwin") and sysconfig.get_config_var(
                "PYTHONFRAMEWORK"
            ):
                scheme = "osx_framework_user"
            else:
                scheme = f"{os.name}_user"
            return Path(sysconfig.get_path("data", scheme)) / "share" / "freva"
        # The default scheme is 'posix_prefix' or 'nt', and should work for e.g.
        # installing into a virtualenv
        return Path(sysconfig.get_path("data")) / "share" / "freva"

    @property
    def userdata_url(self) -> str:
        """Define the url for adding and deleting user-data."""
        return f"{self.databrowser_url}/userdata"


class UserDataHandler:
    """Class for processing user data.

    This class is used for processing user data and extracting metadata
    from the data files.
    """

    def __init__(self, userdata_items: List[Union[str, xr.Dataset]]) -> None:
        self._suffixes = [".nc", ".nc4", ".grb", ".grib", ".zarr", "zar"]
        self.user_metadata: List[
            Dict[str, Union[str, List[str], Dict[str, str]]]
        ] = []
        self._metadata_collection: List[Dict[str, Union[str, List[str]]]] = []
        try:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=min(int(os.cpu_count() or 4), 15)
            )
            self._process_user_data(userdata_items)
        finally:
            self._executor.shutdown(wait=True)

    def _gather_files(self, path: Path, pattern: str = "*") -> Iterator[Path]:
        """Gather all valid files from directory and wildcard pattern."""
        for item in path.rglob(pattern):
            if item.is_file() and item.suffix in self._suffixes:
                yield item

    def _validate_user_data(
        self,
        user_data: Sequence[Union[str, xr.Dataset]],
    ) -> Dict[str, Union[List[Path], List[xr.Dataset]]]:
        validated_paths: List[Path] = []
        validated_xarray_datasets: List[xr.Dataset] = []
        for data in user_data:
            if isinstance(data, (str, Path)):
                path = Path(data).expanduser().absolute()
                if path.is_dir():
                    validated_paths.extend(self._gather_files(path))
                elif path.is_file() and path.suffix in self._suffixes:
                    validated_paths.append(path)
                else:
                    validated_paths.extend(
                        self._gather_files(path.parent, pattern=path.name)
                    )
            elif isinstance(data, xr.Dataset):
                validated_xarray_datasets.append(data)

        if not validated_paths and not validated_xarray_datasets:
            raise FileNotFoundError(
                "No valid file paths or xarray datasets found."
            )
        return {
            "validated_user_paths": validated_paths,
            "validated_user_xrdatasets": validated_xarray_datasets,
        }

    def _process_user_data(
        self,
        userdata_items: List[Union[str, xr.Dataset]],
    ) -> None:
        """Process xarray datasets and file paths using thread pool."""
        futures = []
        validated_userdata: Dict[str, Union[List[Path], List[xr.Dataset]]] = (
            self._validate_user_data(userdata_items)
        )
        if validated_userdata["validated_user_xrdatasets"]:
            futures.append(
                self._executor.submit(
                    self._process_userdata_in_executor,
                    validated_userdata["validated_user_xrdatasets"],
                )
            )

        if validated_userdata["validated_user_paths"]:
            futures.append(
                self._executor.submit(
                    self._process_userdata_in_executor,
                    validated_userdata["validated_user_paths"],
                )
            )
        for future in futures:
            try:
                future.result()
            except Exception as e:  # pragma: no cover
                logger.error(f"Error processing batch: {e}")

    def _process_userdata_in_executor(
        self, validated_userdata: Union[List[Path], List[xr.Dataset]]
    ) -> None:
        for data in validated_userdata:
            metadata = self._get_metadata(data)
            if isinstance(metadata, Exception) or metadata == {}:
                logger.warning("Error getting metadata: %s", metadata)
            else:
                self.user_metadata.append(metadata)

    def _timedelta_to_cmor_frequency(self, dt: float) -> str:
        for total_seconds, frequency in self._time_table.items():
            if dt >= total_seconds:
                return frequency
        return "fx"  # pragma: no cover

    @property
    def _time_table(self) -> dict[int, str]:
        return {
            315360000: "dec",  # Decade
            31104000: "yr",  # Year
            2538000: "mon",  # Month
            1296000: "sem",  # Seasonal (half-year)
            84600: "day",  # Day
            21600: "6h",  # Six-hourly
            10800: "3h",  # Three-hourly
            3600: "hr",  # Hourly
            1: "subhr",  # Sub-hourly
        }

    def _get_time_frequency(self, time_delta: int, freq_attr: str = "") -> str:
        if freq_attr in self._time_table.values():
            return freq_attr
        return self._timedelta_to_cmor_frequency(time_delta)

    def _get_metadata(
        self, path: Union[os.PathLike[str], xr.Dataset]
    ) -> Dict[str, Union[str, List[str], Dict[str, str]]]:
        """Get metadata from a path or xarray dataset."""

        try:
            dset = (
                path
                if isinstance(path, xr.Dataset)
                else xr.open_mfdataset(
                    str(path), parallel=False, use_cftime=True, lock=False
                )
            )
            time_freq = dset.attrs.get("frequency", "")
            data_vars = list(map(str, dset.data_vars))
            coords = list(map(str, dset.coords))
            try:
                times = dset["time"].values[:]
            except (KeyError, IndexError, TypeError):
                times = np.array([])

        except Exception as error:
            logger.error("Failed to open data file %s: %s", str(path), error)
            return {}
        if len(times) > 0:
            try:
                time_str = (
                    f"[{times[0].isoformat()}Z TO {times[-1].isoformat()}Z]"
                )
                dt = (
                    abs((times[1] - times[0]).total_seconds())
                    if len(times) > 1
                    else 0
                )
            except Exception as non_cftime:
                logger.info(
                    "The time var is not based on the cftime: %s", non_cftime
                )
                time_str = (
                    f"[{np.datetime_as_string(times[0], unit='s')}Z TO "
                    f"{np.datetime_as_string(times[-1], unit='s')}Z]"
                )
                dt = (
                    abs(
                        (times[1] - times[0]).astype("timedelta64[s]").astype(int)
                    )
                    if len(times) > 1
                    else 0
                )
        else:
            time_str = "fx"
            dt = 0

        variables = [
            var
            for var in data_vars
            if var not in coords
            and not any(
                term in var.lower() for term in ["lon", "lat", "bnds", "x", "y"]
            )
            and var.lower() not in ["rotated_pole", "rot_pole"]
        ]

        _data: Dict[str, Union[str, List[str], Dict[str, str]]] = {}
        _data.setdefault("variable", variables[0])
        _data.setdefault(
            "time_frequency", self._get_time_frequency(dt, time_freq)
        )
        _data["time"] = time_str
        _data.setdefault("cmor_table", _data["time_frequency"])
        _data.setdefault("version", "")
        if isinstance(path, Path):
            _data["file"] = str(path)
        if isinstance(path, xr.Dataset):
            _data["file"] = str(dset.encoding["source"])
        return _data
