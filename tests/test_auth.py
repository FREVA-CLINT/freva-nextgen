"""Test for the authorisation utilities."""

from copy import deepcopy
from datetime import datetime, timezone
from typing import Dict

import mock
import pytest
import requests
from fastapi.exceptions import HTTPException
from pytest_mock import MockFixture

from freva_client.auth import Auth, authenticate
from freva_rest.auth import SafeAuth


def raise_for_status() -> None:
    """Mock function used for requests result rais_for_status method."""
    raise requests.HTTPError("Invalid")


def test_missing_ocid_server(test_server: str) -> None:
    """Test the behviour of a missing ocid server."""
    for url in ("", "http://example.org/foo", "http://muhah.zupap"):
        with mock.patch(
            "freva_rest.auth.auth.discovery_url",
            url,
        ):
            res = requests.get(
                f"{test_server}/auth/v2/status",
                headers={"Authorization": "Bearer foo"},
            )
            assert res.status_code == 503


def test_authenticate_with_password(
    mocker: MockFixture, auth_instance: Auth
) -> None:
    """Test authentication using username and password."""
    old_token_data = deepcopy(auth_instance._auth_token)
    try:
        token_data = {
            "access_token": "test_access_token",
            "token_type": "Bearer",
            "expires": int(datetime.now(timezone.utc).timestamp() + 3600),
            "refresh_token": "test_refresh_token",
            "refresh_expires": int(datetime.now(timezone.utc).timestamp() + 7200),
            "scope": "profile email address",
        }
        with mocker.patch(
            "freva_client.auth.OAuth2Session.fetch_token",
            return_value=token_data,
        ):
            auth_instance.authenticate(host="https://example.com")
        assert isinstance(auth_instance._auth_token, dict)
        assert auth_instance._auth_token["access_token"] == "test_access_token"
        assert auth_instance._auth_token["refresh_token"] == "test_refresh_token"
    finally:
        auth_instance._auth_token = old_token_data


def test_authenticate_with_refresh_token(
    mocker: MockFixture, auth_instance: Auth
) -> None:
    """Test authentication using a refresh token."""
    old_token_data = deepcopy(auth_instance._auth_token)
    token_data = {
        "access_token": "test_access_token",
        "token_type": "Bearer",
        "expires": int(datetime.now(timezone.utc).timestamp() + 3600),
        "refresh_token": "test_refresh_token",
        "refresh_expires": int(datetime.now(timezone.utc).timestamp() + 7200),
        "scope": "profile email address",
    }
    try:
        with mocker.patch(
            "freva_client.auth.OAuth2Session.fetch_token",
            return_value=token_data,
        ):
            auth_instance.authenticate(
                host="https://example.com", refresh_token="test_refresh_token"
            )

        assert isinstance(auth_instance._auth_token, dict)
        assert auth_instance._auth_token["access_token"] == "test_access_token"
        assert auth_instance._auth_token["refresh_token"] == "test_refresh_token"
    finally:
        auth_instance._auth_token = old_token_data


def test_refresh_token(mocker: MockFixture, auth_instance: Auth) -> None:
    """Test the token refresh functionality."""
    old_token_data = deepcopy(auth_instance._auth_token)
    token_data = {
        "access_token": "new_access_token",
        "token_type": "Bearer",
        "expires": int(datetime.now(timezone.utc).timestamp() + 3600),
        "refresh_token": "new_refresh_token",
        "refresh_expires": int(datetime.now(timezone.utc).timestamp() + 7200),
        "scope": "profile email address",
    }
    try:
        with mocker.patch(
            "freva_client.auth.OAuth2Session.refresh_token",
            return_value=token_data,
        ):
            auth_instance._auth_token = {
                "access_token": "test_access_token",
                "token_type": "Bearer",
                "expires": int(datetime.now().timestamp() - 3600),
                "refresh_token": "test_refresh_token",
                "refresh_expires": int(datetime.now().timestamp() + 7200),
                "scope": "profile email address",
            }

            auth_instance.check_authentication(auth_url="https://example.com")

        assert isinstance(auth_instance._auth_token, dict)
        assert auth_instance._auth_token["access_token"] == "new_access_token"
        assert auth_instance._auth_token["refresh_token"] == "new_refresh_token"
    finally:
        auth_instance._auth_token = old_token_data


def test_authenticate_function(mocker: MockFixture, auth_instance: Auth) -> None:
    """Test the authenticate function with username and password."""
    old_token_data = deepcopy(auth_instance._auth_token)
    token_data = {
        "access_token": "test_access_token",
        "token_type": "Bearer",
        "expires": int(datetime.now(timezone.utc).timestamp() + 3600),
        "refresh_token": "test_refresh_token",
        "refresh_expires": int(datetime.now(timezone.utc).timestamp() + 7200),
        "scope": "profile email address",
    }
    try:
        with mocker.patch(
            "freva_client.auth.OAuth2Session.fetch_token",
            return_value=token_data,
        ):
            token = authenticate(host="https://example.com")

        assert token["access_token"] == "test_access_token"
        assert token["refresh_token"] == "test_refresh_token"
    finally:
        auth_instance._auth_token = old_token_data


def test_authenticate_function_with_refresh_token(
    mocker: MockFixture, auth_instance: Auth
) -> None:
    """Test the authenticate function using a refresh token."""
    old_token_data = deepcopy(auth_instance._auth_token)
    token_data = {
        "access_token": "test_access_token",
        "token_type": "Bearer",
        "expires": int(datetime.now(timezone.utc).timestamp() + 3600),
        "refresh_token": "test_refresh_token",
        "refresh_expires": int(datetime.now(timezone.utc).timestamp() + 7200),
        "scope": "profile email address",
    }
    try:
        with mocker.patch(
            "freva_client.auth.OAuth2Session.refresh_token",
            return_value=token_data,
        ):
            token = authenticate(
                host="https://example.com", refresh_token="test_refresh_token"
            )

        assert token["access_token"] == "test_access_token"
        assert token["refresh_token"] == "test_refresh_token"
    finally:
        auth_instance._auth_token = old_token_data


def test_authentication_fail(mocker: MockFixture, auth_instance: Auth) -> None:
    """Test the behviour if the authentications fails."""
    old_token_data = deepcopy(auth_instance._auth_token)
    mock_token_data = {
        "access_token": "test_access_token",
        "token_type": "Bearer",
        "expires": int(datetime.now(timezone.utc).timestamp() - 3600),
        "refresh_token": "test_refresh_token",
        "refresh_expires": int(datetime.now(timezone.utc).timestamp() - 7200),
        "scope": "profile email address",
    }
    with mocker.patch(
        "freva_client.auth.OAuth2Session.refresh_token",
        return_value={"detail": "Invalid username or password"},
    ):
        with mocker.patch(
            "freva_client.auth.OAuth2Session.fetch_token",
            return_value={"detail": "Invalid username or password"},
        ):
            try:
                auth_instance._auth_token = None
                with pytest.raises(ValueError):
                    authenticate(host="https://example.com")
                with pytest.raises(ValueError):
                    authenticate(
                        host="https://example.com",
                        refresh_token="test_refresh_token",
                    )
                with pytest.raises(ValueError):
                    auth_instance.check_authentication(
                        auth_url="https://example.com"
                    )
                auth_instance._auth_token = mock_token_data
                with pytest.raises(ValueError):
                    auth_instance.check_authentication(
                        auth_url="https://example.com"
                    )
            finally:
                auth_instance._auth_token = old_token_data


def test_real_auth(test_server: str, auth_instance: Auth) -> None:
    """Test authentication at the keycloak instance."""
    old_token_data = deepcopy(auth_instance._auth_token)
    mock_token_data = {
        "access_token": "test_access_token",
        "token_type": "Bearer",
        "expires": int(datetime.now(timezone.utc).timestamp() - 3600),
        "refresh_token": "test_refresh_token",
        "refresh_expires": int(datetime.now(timezone.utc).timestamp() - 7200),
        "scope": "profile email address",
    }

    try:
        auth_instance._auth_token = mock_token_data
        token_data = authenticate(host=test_server)
        assert token_data["access_token"] != mock_token_data["access_token"]
        token_data = authenticate(host=test_server, force=True)
        assert isinstance(token_data, dict)
        assert "access_token" in token_data
        token = token_data["access_token"]
        token_data2 = authenticate(host=test_server)
        assert token_data2["access_token"] == token
    finally:
        auth_instance._auth_token = old_token_data


def test_userinfo(
    mocker: MockFixture, test_server: str, auth: Dict[str, str]
) -> None:
    """Test getting the user info."""

    res = requests.get(
        f"{test_server}/auth/v2/userinfo",
        headers={"Authorization": f"Bearer {auth['access_token']}"},
        timeout=3,
    )
    assert res.status_code == 200
    assert "last_name" in res.json()
    with mocker.patch("freva_rest.auth.get_userinfo", return_value={}):
        res = requests.get(
            f"{test_server}/auth/v2/userinfo",
            headers={"Authorization": f"Bearer {auth['access_token']}"},
            timeout=3,
        )
        assert res.status_code == 404


def test_token_status(test_server: str, auth: Dict[str, str]) -> None:
    """Check the token status methods."""
    res1 = requests.get(
        f"{test_server}/auth/v2/status",
        headers={"Authorization": f"Bearer {auth['access_token']}"},
    )
    assert res1.status_code == 200
    assert "exp" in res1.json()
    res2 = requests.get(
        f"{test_server}/auth/v2/status",
        headers={"Authorization": "Bearer foo"},
    )
    assert res2.status_code != 200


def test_get_overview(test_server: str) -> None:
    """Test the open id connect discovery endpoint."""
    res = requests.get(f"{test_server}/auth/v2/.well-known/openid-configuration")
    assert res.status_code == 200
