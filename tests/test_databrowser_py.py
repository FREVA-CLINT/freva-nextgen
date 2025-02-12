"""Tests for the databrowser class."""

from copy import deepcopy
from tempfile import TemporaryDirectory

import pytest

from freva_client import databrowser
from freva_client.auth import Auth, authenticate
from freva_client.utils.logger import DatabrowserWarning

def test_search_files(test_server: str) -> None:
    """Test searching for files."""
    db = databrowser(host=test_server)
    assert len(list(db)) > 0
    assert len(list(db)) == len(db)
    db = databrowser(host=test_server, foo="bar", fail_on_error=True)
    with pytest.raises(ValueError):
        len(db)
    db = databrowser(host=test_server, foo="bar", time="2000 to 2050")
    assert len(db) == 0
    db = databrowser(host=test_server, model="bar")
    assert len(db) == len(list(db)) == 0
    db = databrowser(host="foo")
    with pytest.raises(ValueError):
        len(db)
    assert (
        len(
            databrowser(
                "land",
                realm="ocean",
                product="reanalysis",
                host=test_server,
            )
        )
        == 0
        == 0
    )


def test_count_values(test_server: str) -> None:
    """Test counting the facets."""
    db = databrowser(host=test_server)
    assert isinstance(len(db), int)
    counts1 = databrowser.count_values("*", host=test_server)
    assert isinstance(counts1, dict)
    assert "dataset" not in counts1
    counts2 = databrowser.count_values(
        "ocean",
        realm="ocean",
        product="reanalysis",
        host=test_server,
        extended_search=True,
    )
    assert isinstance(counts2, dict)
    assert "dataset" in counts2
    assert isinstance(counts2["dataset"], dict)
    entry = list(counts2["dataset"].keys())[0]
    assert isinstance(counts2["dataset"][entry], int)


def test_metadata_search(test_server: str) -> None:
    """Test the metadata search."""
    db = databrowser(host=test_server)
    assert isinstance(db.metadata, dict)
    metadata = databrowser.metadata_search(host=test_server)
    assert isinstance(metadata, dict)
    assert len(db.metadata) > len(metadata)
    metadata = databrowser.metadata_search(
        host=test_server, extended_search=True
    )
    assert len(db.metadata) == len(metadata)


def test_bad_hostnames() -> None:
    """Test the behaviour of non existing host queries."""
    db = databrowser(host="foo")
    with pytest.raises(ValueError):
        len(db)
    with pytest.raises(ValueError):
        databrowser.metadata_search(host="foo")
    with pytest.raises(ValueError):
        databrowser.count_values(host="foo")


def test_bad_queries(test_server: str) -> None:
    """Test the behaviour of bad queries."""
    db = databrowser(host=test_server, foo="bar")
    with pytest.warns(DatabrowserWarning):
        len(db)
    with pytest.warns(DatabrowserWarning):
        databrowser.count_values(host=test_server, foo="bar")
    with pytest.warns(DatabrowserWarning):
        databrowser.metadata_search(host=test_server, foo="bar")
    db = databrowser(host=test_server, foo="bar", fail_on_error=True)
    with pytest.raises(ValueError):
        len(db)
    with pytest.raises(ValueError):
        databrowser.count_values(
            host=test_server, foo="bar", fail_on_error=True
        )
    with pytest.raises(ValueError):
        databrowser.metadata_search(
            host=test_server, foo="bar", fail_on_error=True
        )
    db = databrowser(host=test_server, foo="bar", flavour="foo")  # type: ignore
    with pytest.raises(ValueError):
        len(db)


def test_repr(test_server: str) -> None:
    """Test the str rep."""
    db = databrowser(host=test_server)
    assert test_server in repr(db)
    assert str(len(db)) in db._repr_html_()
    overview = db.overview(host=test_server)
    assert isinstance(overview, str)
    assert "flavour" in overview
    assert "cmip6" in overview
    assert "freva" in overview


def test_intake_without_zarr(test_server: str) -> None:
    """Test the intake catalogue creation."""
    db = databrowser(host=test_server, dataset="cmip6-fs")
    cat = db.intake_catalogue()
    assert hasattr(cat, "df")
    with TemporaryDirectory() as temp_dir:
        with pytest.raises(ValueError):
            db._create_intake_catalogue_file(temp_dir)
    db = databrowser(host=test_server, dataset="foooo")
    with pytest.raises(ValueError):
        db.intake_catalogue()

def test_stac_catalogue(test_server: str) -> None:
    """Test the STAC Catalogue functionality."""
    # dynamic STAC catalogue
    db = databrowser(host=test_server, dataset="cmip6-fs")
    res = db.stac_catalogue()
    assert "STAC catalog available at: " in res

    # static STAC catalogue
    db = databrowser(host=test_server, dataset="cmip6-fs")
    res = db.stac_catalogue(filename="/tmp/something.tar.gz")
    assert "STAC catalog saved to: /tmp/something.tar.gz" in res

def test_intake_with_zarr(test_server: str, auth_instance: Auth) -> None:
    """Test the intake zarr catalogue creation."""
    token = deepcopy(auth_instance._auth_token)
    try:
        auth_instance.auth_instance = None
        db = databrowser(
            host=test_server, dataset="cmip6-fs", stream_zarr=True
        )
        with pytest.raises(ValueError):
            cat = db.intake_catalogue()
        _ = authenticate(username="janedoe", host=test_server)
        cat = db.intake_catalogue()
        assert hasattr(cat, "df")
    finally:
        auth_instance._auth_token = token


def test_zarr_stream(test_server: str, auth_instance: Auth) -> None:
    """Test creating zarr endpoints for loading."""
    token = deepcopy(auth_instance._auth_token)
    try:
        auth_instance.auth_instance = None
        db = databrowser(
            host=test_server, dataset="cmip6-fs", stream_zarr=True
        )
        with pytest.raises(ValueError):
            files = list(db)
        _ = authenticate(username="janedoe", host=test_server)
        files = list(db)
        print(files)
        assert len(files) == 2
    finally:
        auth_instance._auth_token = token


def test_userdata_add_path_xarray_py(
    test_server: str, auth_instance: Auth
) -> None:
    """Test adding path and xarray user data."""
    import xarray as xr

    token = deepcopy(auth_instance._auth_token)
    try:
        auth_instance.auth_instance = None
        _ = authenticate(username="janedoe", host=test_server)

        databrowser.userdata("delete", metadata={}, host=test_server)
        filename = (
            "./freva-rest/src/freva_rest/databrowser_api/mock/data/model/regional"
            "/cordex/output/EUR-11/CLMcom/MPI-M-MPI-ESM-LR/historical"
            "/r0i0p0/CLMcom-CCLM4-8-17/v1/fx/orog/v20140515/"
            "orog_EUR-11_MPI-M-MPI-ESM-LR_historical_r1i1p1_"
            "CLMcom-CCLM4-8-17_v1_fx_another.nc"
        )
        xarray_data = xr.open_dataset(filename)
        databrowser.userdata(
            "add",
            userdata_items=[xarray_data, xarray_data, filename],
            metadata={},
            host=test_server,
        )
        assert len(databrowser(flavour="user", host=test_server)) == 1

    finally:
        auth_instance._auth_token = token


def test_userdata_failed(test_server: str, auth_instance: Auth) -> None:
    """Test user data wrong paths."""
    token = deepcopy(auth_instance._auth_token)
    try:
        auth_instance.auth_instance = None
        db = databrowser(host=test_server)
        _ = authenticate(username="janedoe", host=test_server)
        length = len(db)
        with pytest.raises(FileNotFoundError) as exc_info:
            databrowser.userdata(
                "add",
                userdata_items=["/somewhere/wrong"],
                metadata={"username": "johndoe"},
                host=test_server,
            )
        assert "No valid file paths or xarray datasets found." in str(
            exc_info.value
        )
        assert len(db) == length
    finally:
        auth_instance._auth_token = token


def test_userdata_post_delete_failure(
    test_server: str, auth_instance: Auth
) -> None:
    """Test failure of adding user data."""

    token = deepcopy(auth_instance._auth_token)
    try:
        auth_instance.auth_instance = None
        _ = authenticate(username="janedoe", host=test_server)
        with pytest.raises(ValueError):
            databrowser.userdata(
                "add",
                userdata_items=[
                    "./freva-rest/src/freva_rest/databrowser_api/mock_broken/bears.nc"
                ],
                metadata={"username": "janedoe"},
                host="foo.bar.de:7777",
                fail_on_error=True,
            )
        with pytest.raises(ValueError):
            databrowser.userdata(
                "delete",
                metadata={"username": "janedoe"},
                host="foo.bar.de:7777",
                fail_on_error=True,
            )
    finally:
        auth_instance._auth_token = token


def test_userdata_post_delete_without_failure(
    test_server: str, auth_instance: Auth
) -> None:
    """Test successful deleting user data."""

    token = deepcopy(auth_instance._auth_token)
    try:
        auth_instance.auth_instance = None
        _ = authenticate(username="janedoe", host=test_server)
        with pytest.raises(ValueError):
            databrowser.userdata(
                "add",
                userdata_items=[
                    "./freva-rest/src/freva_rest/databrowser_api/mock_broken/bears.nc"
                ],
                metadata={"username": "janedoe"},
                host="foo.bar.de:7777",
            )
        with pytest.raises(ValueError):
            databrowser.userdata(
                "delete",
                metadata={"username": "janedoe"},
                host="foo.bar.de:7777",
            )
    finally:
        auth_instance._auth_token = token


def test_userdata_correct_args_wrong_place(
    test_server: str, auth_instance: Auth
) -> None:
    """Test adding user data with wrong arguments."""
    token = deepcopy(auth_instance._auth_token)
    try:
        auth_instance.auth_instance = None
        _ = authenticate(username="janedoe", host=test_server)
        with pytest.raises(FileNotFoundError):
            databrowser.userdata(
                "add", metadata={"username": "johndoe"}, host=test_server
            )
        databrowser.userdata(
            "delete",
            userdata_items=[
                "./freva-rest/src/freva_rest/databrowser_api/mock_broken/bears.nc"
            ],
            metadata={"username": "johndoe"},
            host=test_server,
        )
    finally:
        auth_instance._auth_token = token


def test_userdata_empty_metadata_value_error(
    test_server: str, auth_instance: Auth
) -> None:
    """Test adding user data with wrong arguments."""
    token = deepcopy(auth_instance._auth_token)
    try:
        auth_instance.auth_instance = None
        _ = authenticate(username="janedoe", host=test_server)
        with pytest.raises(ValueError):
            databrowser.userdata(
                "add",
                userdata_items=[
                    (
                        "./freva-rest/src/freva_rest/databrowser_api/mock/data/model/obs/"
                        "reanalysis/reanalysis/NOAA/NODC/OC5/mon/ocean/Omon/"
                        "r1i1p1/v20200101/hc700/"
                        "hc700_mon_NODC_OC5_r1i1p1_201201-201212.nc"
                    )
                ],
                metadata={"username": "johndoe"},
                host=test_server,
            )
    finally:
        auth_instance._auth_token = token


def test_userdata_non_path_xarray(
    test_server: str, auth_instance: Auth
) -> None:
    """Test adding user data with wrong arguments."""
    token = deepcopy(auth_instance._auth_token)
    try:
        auth_instance.auth_instance = None
        _ = authenticate(username="janedoe", host=test_server)
        with pytest.raises(FileNotFoundError):
            databrowser.userdata(
                "add",
                userdata_items=[[1]],
                metadata={"username": "johndoe"},
                host=test_server,
            )
    finally:
        auth_instance._auth_token = token


def test_add_userdata_wild_card(test_server: str, auth_instance: Auth) -> None:
    """Test adding user data with wild card."""
    token = deepcopy(auth_instance._auth_token)
    try:
        auth_instance.auth_instance = None
        _ = authenticate(username="janedoe", host=test_server)
        databrowser.userdata("delete", host=test_server)
        databrowser.userdata(
            "add",
            userdata_items=[
                (
                    "./freva-rest/src/freva_rest/databrowser_api/mock/data/model/global/"
                    "cmip6/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/amip/r2i1p1f1/"
                    "Amon/ua/gn/v20190815/*.nc"
                ),
                (
                    "./freva-rest/src/freva_rest/databrowser_api/mock/data/model/global/"
                    "cmip6/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/amip/r2i1p1f1/Amon/"
                    "ua/gn/v20190815/somewhere_wrong/*.nc"
                ),
            ],
            metadata={"username": "johndoe"},
            host=test_server,
        )
    finally:
        auth_instance._auth_token = token
