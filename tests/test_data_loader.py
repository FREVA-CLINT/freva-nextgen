"""Tests for the data loader."""

import time

import pytest
from data_portal_worker.load_data import RedisCacheFactory, get_dask_client
from distributed import Client
from pytest import LogCaptureFixture


def test_get_client() -> None:
    """Test the dask client creation."""

    assert get_dask_client(None, True) is None
    client = get_dask_client()
    assert isinstance(client, Client)
    client.shutdown()


def test_broker(caplog: LogCaptureFixture) -> None:
    """Test seding messags to the broker."""
    caplog.clear()
    cache = RedisCacheFactory()
    cache.publish("data-portal", "foo")
    time.sleep(0.5)
    assert caplog.records
    assert caplog.records[-1].levelname == "WARNING"
