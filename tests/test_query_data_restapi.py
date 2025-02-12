"""Unit tests for data queries via the rest-api."""

import json
import os
import time
from typing import Dict

import mock
import requests
from pymongo import MongoClient

from freva_rest.config import ServerConfig


def test_attributes(test_server: str) -> None:
    """Test getting the attributes."""
    res1 = requests.get(f"{test_server}/databrowser/overview")
    assert isinstance(res1.json()["flavours"], list)
    assert isinstance(res1.json()["attributes"], dict)


def test_databrowser(test_server: str) -> None:
    """Test the default databrowser functionality."""
    res1 = requests.get(
        f"{test_server}/databrowser/data-search/cmip6/uri",
        params={"activity_id": "cmipx"},
    )
    assert res1.status_code == 200
    res2 = requests.get(
        f"{test_server}/databrowser/data-search/cmip6/uri",
        params={"translate": "false", "product": "cmip"},
    )
    res3 = requests.get(
        f"{test_server}/databrowser/data-search/freva/uri",
        params={"product": "cmip"},
    )
    assert len(res1.text.split()) == 0
    assert res2.text == res3.text
    res4 = requests.get(
        f"{test_server}/databrowser/data-search/cmip6/uri",
        params={"foo": "bar"},
    )
    assert res4.status_code == 422
    res5 = requests.get(
        f"{test_server}/databrowser/data-search/cmip6/uri",
        params={
            "translate": "false",
            "product": "cmip",
            "multi-version": "true",
        },
    )
    assert len(res2.text.split()) < len(res5.text.split())


def test_no_solr(test_server: str) -> None:
    """Test what happens if there is no connection to the solr."""
    with mock.patch("freva_rest.rest.server_config.solr_host", "foo.bar"):
        res = requests.get(
            f"{test_server}/databrowser/data-search/cmip6/uri",
            params={"activity_id": "cmipx"},
        )
        assert res.status_code == 503


def test_file_select(test_server: str) -> None:
    """Test what happens if we search for files/uris."""
    res = requests.get(
        f"{test_server}/databrowser/data-search/cmip6/file",
        params={"file": "/arch/bb1203/*/CPC/*"},
    )
    assert res.status_code == 200
    assert len(res.text.split()) > 0
    res = requests.get(
        f"{test_server}/databrowser/metadata-search/cmip6/file",
        params={"uri": "slk:///arch/bb1203/*/CPC/*"},
    )
    assert res.status_code == 200
    data = res.json()
    assert len(data) > 0
    assert "cmorph" in data["facets"]["experiment_id"]


def test_time_selection(test_server: str) -> None:
    """Test the time select functionality of the API."""
    res1 = requests.get(
        f"{test_server}/databrowser/data-search/freva/file",
        params={"time": "1898 to 1901"},
    )
    assert len(res1.text.split()) == 1
    res2 = requests.get(
        f"{test_server}/databrowser/data-search/freva/file",
        params={"time": "1898 to 1901", "time_select": "foo"},
    )
    assert res2.status_code == 500
    res3 = requests.get(
        f"{test_server}/databrowser/data-search/freva/file",
        params={"time": "fx"},
    )
    assert res3.status_code == 500

def test_bbox_selection(test_server: str) -> None:
    """Test the bbox select functionality of the API."""
    res1 = requests.get(
        f"{test_server}/databrowser/data-search/freva/file",
        params={"bbox": "-10,10 by -10,10"},
    )
    assert len(res1.text.split()) == 61
    res2 = requests.get(
        f"{test_server}/databrowser/data-search/freva/file",
        params={"bbox": "-10,10 by -10,10", "bbox_select": "foo"},
    )
    assert res2.status_code == 500
    res3 = requests.get(
        f"{test_server}/databrowser/data-search/freva/file",
        params={"bbox": "fx"},
    )
    assert res3.status_code == 500


def test_primary_facets(test_server: str) -> None:
    """Test the functionality of primary facet definitions."""
    res1 = requests.get(
        f"{test_server}/databrowser/metadata-search/freva/file"
    ).json()
    res2 = requests.get(
        f"{test_server}/databrowser/metadata-search/cmip6/file"
    ).json()
    res3 = requests.get(
        f"{test_server}/databrowser/metadata-search/cmip6/file",
        params={"translate": "f"},
    ).json()
    assert "primary_facets" in res1
    assert "primary_facets" in res2
    assert "primary_facets" in res3
    assert (
        len(res1["primary_facets"])
        == len(res2["primary_facets"])
        == len(res3["primary_facets"])
    )
    assert res1["primary_facets"] == res3["primary_facets"]
    assert res1["primary_facets"] != res2["primary_facets"]


def test_extended_search(test_server: str) -> None:
    """Test the facet search functionality."""
    res1 = requests.get(
        f"{test_server}/databrowser/extended-search/cmip6/uri",
        params={"start": 0, "activity_id": "cmip", "max-results": 2},
    ).json()
    assert len(res1["search_results"]) > 0
    assert "activity_id" in res1["facets"]
    res2 = requests.get(
        f"{test_server}/databrowser/extended-search/cmip6/uri",
        params={"start": 1000, "activity_id": "cmip", "max-results": 2},
    ).json()
    assert "rcm_name" not in res2["primary_facets"]
    assert res2["search_results"] == []
    res3 = requests.get(
        f"{test_server}/databrowser/extended-search/cmip5/uri",
        params={"activity_id": "cmip", "translate": "false", "max-results": 2},
    )
    assert res3.status_code == 422
    res4 = requests.get(
        f"{test_server}/databrowser/extended-search/cmip6/uri",
        params={"activity_id": "cmipx", "translate": "true", "max-results": 2},
    )
    assert res4.status_code == 200
    res5 = requests.get(
        f"{test_server}/databrowser/extended-search/cordex/uri",
        params={"domain": "eur-11", "translate": "true"},
    ).json()
    assert "rcm_name" in res5["facets"]
    assert "rcm_name" in res5["primary_facets"]

    res6 = requests.get(
        f"{test_server}/databrowser/extended-search/cmip6/uri",
        params={"facets": "activity_id"},
    ).json()
    assert len(res6["facets"].keys()) == 1
    res7 = requests.get(
        f"{test_server}/databrowser/extended-search/cmip6/uri",
        params={"facets": "activity_id", "max-results": 0},
    ).json()
    assert len(res7["search_results"]) == 0


def test_metadata_search(test_server: str) -> None:
    """Test the facet search functionality."""
    res1 = requests.get(
        f"{test_server}/databrowser/metadata-search/cmip6/uri",
        params={"activity_id": "cmip"},
    ).json()
    assert "search_results" not in res1.keys()
    assert "activity_id" in res1["facets"]
    res3 = requests.get(
        f"{test_server}/databrowser/extended-search/cmip5/uri",
        params={"activity_id": "cmip", "translate": "false"},
    )
    assert res3.status_code == 422
    res4 = requests.get(
        f"{test_server}/databrowser/metadata-search/cmip6/uri",
        params={"activity_id": "cmipx", "translate": "true"},
    )
    assert res4.status_code == 200
    res5 = requests.get(
        f"{test_server}/databrowser/metadata-search/cordex/uri",
        params={"domain": "eur-11", "translate": "true"},
    ).json()
    assert "rcm_name" in res5["facets"]
    assert "rcm_name" in res5["primary_facets"]

    res6 = requests.get(
        f"{test_server}/databrowser/metadata-search/cmip6/uri",
        params={"facets": "activity_id"},
    ).json()
    assert len(res6["facets"].keys()) == 1


def test_contains_not(test_server: str) -> None:
    """Test for searches that should *not* contain values."""

    res1 = requests.get(
        f"{test_server}/databrowser/metadata-search/freva/file",
        params={
            "dataset": ["-cmip6-swift", "not cmip6-fs"],
            "project": "cmip6",
        },
    ).json()
    assert "cmip6-swift" not in res1["facets"]["dataset"]
    assert "cmip6-fs" not in res1["facets"]["dataset"]
    assert "cmip6-hsm" in res1["facets"]["dataset"]
    res2 = requests.get(
        f"{test_server}/databrowser/metadata-search/freva/file",
        params={
            "project_not_": "cmip6",
        },
    ).json()
    assert "cmip6" not in res2["facets"]["project"]


def test_intake_search(test_server: str) -> None:
    """Test the creation of intake catalogues."""
    res1 = requests.get(
        f"{test_server}/databrowser/intake-catalogue/cmip6/uri",
        params={"activity_id": "cmip", "multi-version": True},
    )
    assert res1.json() == json.loads(res1.text)
    res2 = requests.get(
        f"{test_server}/databrowser/intake-catalogue/cmip6/uri",
    )
    assert len(res2.json()["catalog_dict"]) > len(res1.json()["catalog_dict"])
    res3 = requests.get(
        f"{test_server}/databrowser/intake-catalogue/cmip6/uri",
        params={
            "activity_id": "cmip",
            "multi-version": False,
        },
    )
    assert len(res1.json()["catalog_dict"]) > len(res3.json()["catalog_dict"])
    res4 = requests.get(
        f"{test_server}/databrowser/intake-catalogue/cmip6/uri",
        params={
            "multi-version": False,
            "max-results": 1,
        },
    )
    assert res4.status_code == 413


def test_stac_catalogue(test_server: str) -> None:
    """Test the creation of STAC Catalogue."""
    # 303 redirect means 200 OK from freva-rest api endpoint
    res = requests.get(
        f"{test_server}/databrowser/stac-catalogue/cmip6/uri",
        params={
            "activity_id": "cmip", 
            "multi-version": True, 
            "stac_dynamic": True
        },
        allow_redirects=False
    )
    
    assert res.status_code == 303

    assert 'Location' in res.headers
    redirect_url = res.headers['Location']
    assert redirect_url.startswith(('http://', 'https://'))
    assert '/collections/' in redirect_url

    # 413 Request Entity Too Large
    res3 = requests.get(
        f"{test_server}/databrowser/stac-catalogue/cmip6/uri",
        params={"activity_id": "cmip", "multi-version": False, "max-results": 1},
    )
    assert res3.status_code == 413
    # 422 Unprocessable Entity
    res4 = requests.get(
        f"{test_server}/databrowser/stac-catalogue/cmip6/uri",
        params={"collection": "cmip2", "multi-version": False},
    )
    assert res4.status_code == 422
    # 404 Not Found
    res5 = requests.get(
        f"{test_server}/databrowser/stac-catalogue/cmip6/uri",
        params={"activity_id": "cmip3", "multi-version": False},
    )
    assert res5.status_code == 404
    # 500 Internal Server Error, no crendentials or server is no running
    with mock.patch("freva_rest.rest.server_config.stacapi_host", "http://wrong:wrong@foo.bar:8083"), \
         mock.patch("freva_rest.databrowser_api.core.Solr._session_get") as mock_get:
        
        # Mock connection failure
        mock_get.return_value.__aenter__.side_effect = Exception("Connection failed")
        
        res2 = requests.get(
            f"{test_server}/databrowser/stac-catalogue/cmip6/uri",
            params={
                "activity_id": "cmip",
                "multi-version": True,
                "stac_dynamic": True
            }
        )
        assert res2.status_code == 500
def test_bad_intake_request(test_server: str) -> None:
    """Test for a wrong intake request."""
    res1 = requests.get(
        f"{test_server}/databrowser/intake-catalogue/cmip6/uri",
        params={"activity_id": "cmip2"},
    )
    assert res1.status_code == 404


def test_parameter_validation(test_server: str) -> None:
    """Test if only valid parameter requests make it."""

    res1 = requests.get(
        f"{test_server}/databrowser/data-search/cmip6/uri",
        params={"activity_id": "cmip", "translate": "false"},
    ).status_code
    res2 = requests.get(
        f"{test_server}/databrowser/data-search/cmip6/uri",
        params={"product": "cmip", "translate": "true"},
    ).status_code
    res3 = requests.get(
        f"{test_server}/databrowser/data-search/cmip6/uri",
        params={"activity_": "cmip"},
    ).status_code
    assert res1 == res2 == res3 == 422


def test_no_mongo_parameter_insert(test_server: str) -> None:
    """Test the insertion of data into the mongodb."""
    from freva_rest.rest import server_config

    with MongoClient(server_config.mongo_url) as mongo_client:
        collection = mongo_client[server_config.mongo_db]["search_queries"]
        collection.drop()
        assert collection.find_one() is None
    with mock.patch("freva_rest.rest.server_config.mongo_password", "foo"):
        server_config.power_cycle_mongodb()
        res1 = requests.get(
            f"{test_server}/databrowser/data-search/cmip6/uri",
            params={"activity_id": "cmip"},
        ).status_code
        assert res1 == 200
    server_config.power_cycle_mongodb()
    with MongoClient(server_config.mongo_url) as mongo_client:
        collection = mongo_client[server_config.mongo_db]["search_queries"]
        assert collection.find_one() is None


def test_zarr_stream_not_implemented(
    test_server: str, auth: Dict[str, str]
) -> None:
    """Test if zarr request is not served when told not to do so."""
    with mock.patch("freva_rest.rest.server_config.api_services", ""):
        res = requests.get(
            f"{test_server}/databrowser/load/freva",
            headers={"Authorization": f"Bearer {auth['access_token']}"},
        )
        assert res.status_code == 503


def test_mongo_parameter_insert(test_server: str, cfg: ServerConfig) -> None:
    """Test the successfull insertion of the search stats."""
    res1 = requests.get(
        f"{test_server}/databrowser/data-search/cordex/uri",
        params={"variable": ["wind", "cape"]},
    )
    assert res1.status_code == 200
    time.sleep(2)
    mongo_client = MongoClient(cfg.mongo_url)
    collection = mongo_client[cfg.mongo_db]["search_queries"]
    assert len(res1.text.split()) > 0
    stats = list(collection.find())
    assert len(stats) > 0
    assert isinstance(stats[0], dict)
    assert "metadata" in stats[0]
    assert "query" in stats[0]
    assert isinstance(stats[0]["query"], dict)
    assert isinstance(stats[0]["metadata"], dict)
    # The search keys should have been converted to strings.
    assert (
        len([k for k in stats[0]["query"].values() if not isinstance(k, str)])
        == 0
    )
