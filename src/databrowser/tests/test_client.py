"""Unit tests for the freva databrowser module."""
import json

from fastapi.testclient import TestClient
from pymongo import MongoClient

from databrowser.config import ServerConfig


def test_attributes(client: TestClient) -> None:
    """Test getting the attributes."""
    res1 = client.get("/api/databrowser/overview")
    assert isinstance(res1.json()["flavours"], list)
    assert isinstance(res1.json()["attributes"], dict)


def test_databrowser(client: TestClient) -> None:
    """Test the default databrowser functionality."""
    res1 = client.get(
        "/api/databrowser/data_search/cmip6/uri",
        params={"activity_id": "cmipx"},
    )
    assert res1.status_code == 200
    res2 = client.get(
        "/api/databrowser/data_search/cmip6/uri",
        params={"translate": "false", "product": "cmip"},
    )
    res3 = client.get(
        "/api/databrowser/data_search/freva/uri", params={"product": "cmip"}
    )
    assert len(res1.text.split()) == 0
    assert res2.text == res3.text
    res4 = client.get("/api/databrowser/data_search/cmip6/uri", params={"foo": "bar"})
    assert res4.status_code == 422
    res5 = client.get(
        "/api/databrowser/data_search/cmip6/uri",
        params={
            "translate": "false",
            "product": "cmip",
            "multi-version": True,
        },
    )
    assert len(res2.text.split()) < len(res5.text.split())


def test_time_selection(client: TestClient) -> None:
    """Test the time select functionality of the API."""
    res1 = client.get(
        "/api/databrowser/data_search/freva/file",
        params={"time": "1898 to 1901"},
    )
    assert len(res1.text.split()) == 1
    res2 = client.get(
        "/api/databrowser/data_search/freva/file",
        params={"time": "1898 to 1901", "time_select": "foo"},
    )
    assert res2.status_code == 500
    res3 = client.get("/api/databrowser/data_search/freva/file", params={"time": "fx"})
    assert res3.status_code == 500


def test_primary_facets(client: TestClient) -> None:
    """Test the functionality of primary facet definitions."""
    res1 = client.get("api/databrowser/metadata_search/freva/file").json()
    res2 = client.get("api/databrowser/metadata_search/cmip6/file").json()
    res3 = client.get(
        "api/databrowser/metadata_search/cmip6/file", params={"translate": "f"}
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


def test_extended_search(client: TestClient) -> None:
    """Test the facet search functionality."""
    res1 = client.get(
        "api/databrowser/extended_search/cmip6/uri",
        params={"start": 0, "activity_id": "cmip", "max-results": 2},
    ).json()
    assert len(res1["search_results"]) > 0
    assert "activity_id" in res1["facets"]
    res2 = client.get(
        "api/databrowser/extended_search/cmip6/uri",
        params={"start": 1000, "activity_id": "cmip", "max-results": 2},
    ).json()
    assert "rcm_name" not in res2["primary_facets"]
    assert res2["search_results"] == []
    res3 = client.get(
        "api/databrowser/extended_search/cmip5/uri",
        params={"activity_id": "cmip", "translate": "false", "max-results": 2},
    )
    assert res3.status_code == 422
    res4 = client.get(
        "api/databrowser/extended_search/cmip6/uri",
        params={"activity_id": "cmipx", "translate": "true", "max-results": 2},
    )
    assert res4.status_code == 200
    res5 = client.get(
        "api/databrowser/extended_search/cordex/uri",
        params={"domain": "eur-11", "translate": "true"},
    ).json()
    assert "rcm_name" in res5["facets"]
    assert "rcm_name" in res5["primary_facets"]

    res6 = client.get(
        "api/databrowser/extended_search/cmip6/uri",
        params={"facets": "activity_id"},
    ).json()
    assert len(res6["facets"].keys()) == 1
    res7 = client.get(
        "api/databrowser/extended_search/cmip6/uri",
        params={"facets": "activity_id", "max-results": 0},
    ).json()
    assert len(res7["search_results"]) == 0


def test_metadata_search(client: TestClient) -> None:
    """Test the facet search functionality."""
    res1 = client.get(
        "api/databrowser/metadata_search/cmip6/uri",
        params={"activity_id": "cmip"},
    ).json()
    assert "search_results" not in res1.keys()
    assert "activity_id" in res1["facets"]
    res3 = client.get(
        "api/databrowser/extended_search/cmip5/uri",
        params={"activity_id": "cmip", "translate": "false"},
    )
    assert res3.status_code == 422
    res4 = client.get(
        "api/databrowser/metadata_search/cmip6/uri",
        params={"activity_id": "cmipx", "translate": "true"},
    )
    assert res4.status_code == 200
    res5 = client.get(
        "api/databrowser/metadata_search/cordex/uri",
        params={"domain": "eur-11", "translate": "true"},
    ).json()
    assert "rcm_name" in res5["facets"]
    assert "rcm_name" in res5["primary_facets"]

    res6 = client.get(
        "api/databrowser/metadata_search/cmip6/uri",
        params={"facets": "activity_id"},
    ).json()
    assert len(res6["facets"].keys()) == 1


def test_intake_search(client: TestClient) -> None:
    """Test the creation of intake catalogues."""
    res1 = client.get(
        "api/databrowser/intake_catalogue/cmip6/uri",
        params={"activity_id": "cmip", "multi-version": True},
    )
    assert res1.json() == json.loads(res1.text)
    res2 = client.get(
        "api/databrowser/intake_catalogue/cmip6/uri",
    )
    assert len(res2.json()["catalog_dict"]) > len(res1.json()["catalog_dict"])
    res3 = client.get(
        "api/databrowser/intake_catalogue/cmip6/uri",
        params={
            "activity_id": "cmip",
            "multi-version": False,
        },
    )
    assert len(res1.json()["catalog_dict"]) > len(res3.json()["catalog_dict"])
    res4 = client.get(
        "api/databrowser/intake_catalogue/cmip6/uri",
        params={
            "multi-version": False,
            "max-results": 1,
        },
    )
    assert res4.status_code == 400


def test_bad_intake_request(client: TestClient) -> None:
    """Test for a wrong intake request."""
    res1 = client.get(
        "api/databrowser/intake_catalogue/cmip6/uri",
        params={"activity_id": "cmip2"},
    )
    assert res1.status_code == 400


def test_parameter_validation(client: TestClient) -> None:
    """Test if only valid parameter requests make it."""

    res1 = client.get(
        "api/databrowser/data_search/cmip6/uri",
        params={"activity_id": "cmip", "translate": "false"},
    ).status_code
    res2 = client.get(
        "api/databrowser/data_search/cmip6/uri",
        params={"product": "cmip", "translate": "true"},
    ).status_code
    res3 = client.get(
        "api/databrowser/data_search/cmip6/uri", params={"activity_": "cmip"}
    ).status_code
    assert res1 == res2 == res3 == 422


def test_no_mongo_parameter_insert(client_no_mongo: TestClient) -> None:
    """Test the insertion of data into the mongodb."""
    res1 = client_no_mongo.get(
        "api/databrowser/data_search/cmip6/uri",
        params={"activity_id": "cmip"},
    ).status_code
    assert res1 == 200


def tests_mongo_parameter_insert(client: TestClient, cfg: ServerConfig) -> None:
    """Test the successfull insertion of the search stats."""
    res1 = client.get(
        "api/databrowser/data_search/cordex/uri",
        params={"variable": ["wind", "cape"]},
    ).status_code
    assert res1 == 200
    mongo_client = MongoClient(cfg.mongo_url)  # type: ignore
    collection = mongo_client[cfg.mongo_db]["search_queries"]
    stats = list(collection.find({}))
    assert len(stats) > 0
    assert isinstance(stats[0], dict)
    assert "metadata" in stats[0]
    assert "query" in stats[0]
    assert isinstance(stats[0]["query"], dict)
    assert isinstance(stats[0]["metadata"], dict)
    # The search keys should have been converted to strings.
    assert len([k for k in stats[0]["query"].values() if not isinstance(k, str)]) == 0
