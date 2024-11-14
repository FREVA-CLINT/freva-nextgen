"""Miscellaneous tests."""

from urllib.parse import urlparse

import requests


def test_help_page(test_server: str) -> None:
    """Test if the help page is available."""
    res = requests.get(f"{test_server}/help")
    assert "redoc" in res.text.lower()


def test_favicon(test_server: str) -> None:
    """Test if the favicon works."""
    parsed_url = urlparse(test_server)

    res = requests.get(
        f"{parsed_url.scheme}://{parsed_url.netloc}/favicon.ico"
    )
    assert res.ok is True
