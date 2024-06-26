"""Test for the authorisation utilities."""

from copy import deepcopy
from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest
import requests
from freva_client.auth import Auth, Token, authenticate
from pytest_mock import MockFixture


def raise_for_status() -> None:
    """Mock function used for requests result rais_for_status method."""
    raise requests.HTTPError("Invalid")


def test_authenticate_with_password(
    mocker: MockFixture, auth_instance: Auth
) -> None:
    """Test authentication using username and password."""
    old_token_data = deepcopy(auth_instance.auth_token)
    try:
        token_data = {
            "access_token": "test_access_token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "expires": datetime.now().timestamp() + 3600,
            "refresh_token": "test_refresh_token",
            "refresh_expires_in": 7200,
            "refresh_expires": datetime.now().timestamp() + 7200,
        }
        with mocker.patch(
            "freva_client.auth.requests.post",
            return_value=Mock(status_code=200, json=lambda: token_data),
        ):
            auth_instance.authenticate(host="https://example.com")
        assert isinstance(auth_instance.auth_token, dict)
        assert auth_instance.auth_token["access_token"] == "test_access_token"
        assert (
            auth_instance.auth_token["refresh_token"] == "test_refresh_token"
        )
    finally:
        auth_instance.auth_token = old_token_data


def test_authenticate_with_refresh_token(
    mocker: MockFixture, auth_instance: Auth
) -> None:
    """Test authentication using a refresh token."""
    old_token_data = deepcopy(auth_instance.auth_token)
    token_data = {
        "access_token": "test_access_token",
        "token_type": "Bearer",
        "expires_in": 3600,
        "expires": datetime.now().timestamp() + 3600,
        "refresh_token": "test_refresh_token",
        "refresh_expires_in": 7200,
        "refresh_expires": datetime.now().timestamp() + 7200,
    }
    try:
        with mocker.patch(
            "freva_client.auth.requests.post",
            return_value=Mock(status_code=200, json=lambda: token_data),
        ):
            auth_instance.authenticate(
                host="https://example.com", refresh_token="test_refresh_token"
            )

        assert isinstance(auth_instance.auth_token, dict)
        assert auth_instance.auth_token["access_token"] == "test_access_token"
        assert (
            auth_instance.auth_token["refresh_token"] == "test_refresh_token"
        )
    finally:
        auth_instance.auth_token = old_token_data


def test_token_is_expired(mocker: MockFixture, auth_instance: Auth) -> None:
    """Test the token expiration check."""
    old_token_data = deepcopy(auth_instance.auth_token)
    exp_time = (datetime.now() + timedelta(seconds=3600)).timestamp()
    try:
        with mocker.patch(
            "freva_client.auth.requests.get",
            return_value=Mock(status_code=200, json=lambda: {"exp": exp_time}),
        ):
            is_expired = auth_instance._token_is_expired(
                "https://example.com", "test_access_token"
            )

        assert not is_expired
    finally:
        auth_instance.auth_token = old_token_data


def test_refresh_token(mocker: MockFixture, auth_instance: Auth) -> None:
    """Test the token refresh functionality."""
    old_token_data = deepcopy(auth_instance.auth_token)
    token_data = {
        "access_token": "new_access_token",
        "token_type": "Bearer",
        "expires_in": 3600,
        "expires": datetime.now().timestamp() + 3600,
        "refresh_token": "new_refresh_token",
        "refresh_expires_in": 7200,
        "refresh_expires": datetime.now().timestamp() + 7200,
    }
    try:
        with mocker.patch(
            "freva_client.auth.requests.post",
            return_value=Mock(status_code=200, json=lambda: token_data),
        ):
            auth_instance.auth_token = {
                "access_token": "test_access_token",
                "token_type": "Bearer",
                "expires_in": 3600,
                "expires": datetime.now().timestamp() - 3600,
                "refresh_token": "test_refresh_token",
                "refresh_expires_in": 7200,
                "refresh_expires": datetime.now().timestamp() + 7200,
            }

            auth_instance.check_authentication(auth_url="https://example.com")

        assert isinstance(auth_instance.auth_token, dict)
        assert auth_instance.auth_token["access_token"] == "new_access_token"
        assert auth_instance.auth_token["refresh_token"] == "new_refresh_token"
    finally:
        auth_instance.auth_token = old_token_data


def test_authenticate_function(
    mocker: MockFixture, auth_instance: Auth
) -> None:
    """Test the authenticate function with username and password."""
    old_token_data = deepcopy(auth_instance.auth_token)
    token_data = {
        "access_token": "test_access_token",
        "token_type": "Bearer",
        "expires_in": 3600,
        "expires": datetime.now().timestamp() + 3600,
        "refresh_token": "test_refresh_token",
        "refresh_expires_in": 7200,
        "refresh_expires": datetime.now().timestamp() + 7200,
    }
    try:
        with mocker.patch(
            "freva_client.auth.requests.post",
            return_value=Mock(status_code=200, json=lambda: token_data),
        ):
            token = authenticate(host="https://example.com")

        assert token["access_token"] == "test_access_token"
        assert token["refresh_token"] == "test_refresh_token"
    finally:
        auth_instance.auth_token = old_token_data


def test_authenticate_function_with_refresh_token(
    mocker: MockFixture, auth_instance: Auth
) -> None:
    """Test the authenticate function using a refresh token."""
    old_token_data = deepcopy(auth_instance.auth_token)
    token_data = {
        "access_token": "test_access_token",
        "token_type": "Bearer",
        "expires_in": 3600,
        "expires": datetime.now().timestamp() + 3600,
        "refresh_token": "test_refresh_token",
        "refresh_expires_in": 7200,
        "refresh_expires": datetime.now().timestamp() + 7200,
    }
    try:
        with mocker.patch(
            "freva_client.auth.requests.post",
            return_value=Mock(status_code=200, json=lambda: token_data),
        ):
            token = authenticate(
                host="https://example.com", refresh_token="test_refresh_token"
            )

        assert token["access_token"] == "test_access_token"
        assert token["refresh_token"] == "test_refresh_token"
    finally:
        auth_instance.auth_token = old_token_data


def test_authentication_fail(mocker: MockFixture, auth_instance: Auth) -> None:
    """Test the behviour if the authentications fails."""
    old_token_data = deepcopy(auth_instance.auth_token)
    mock_token_data: Token = {
        "access_token": "test_access_token",
        "token_type": "Bearer",
        "expires_in": 3600,
        "expires": datetime.now().timestamp() - 3600,
        "refresh_token": "test_refresh_token",
        "refresh_expires_in": 7200,
        "refresh_expires": datetime.now().timestamp() - 7200,
    }
    with mocker.patch(
        "freva_client.auth.requests.post",
        return_value=Mock(
            status_code=404,
            raise_for_status=raise_for_status,
            json=lambda: {"detail": "Invalid username or password"},
        ),
    ):
        try:
            auth_instance.auth_token = None
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
            auth_instance.auth_token = mock_token_data
            with pytest.raises(ValueError):
                auth_instance.check_authentication(
                    auth_url="https://example.com"
                )
        finally:
            auth_instance.auth_token = old_token_data


def test_real_auth(test_server: str, auth_instance: Auth) -> None:
    """Test authentication at the keycloak instance."""
    old_token_data = deepcopy(auth_instance.auth_token)
    mock_token_data: Token = {
        "access_token": "test_access_token",
        "token_type": "Bearer",
        "expires_in": 3600,
        "expires": datetime.now().timestamp() - 3600,
        "refresh_token": "test_refresh_token",
        "refresh_expires_in": 7200,
        "refresh_expires": datetime.now().timestamp() - 7200,
    }

    try:
        auth_instance.auth_token = mock_token_data
        token_data = authenticate(host=test_server)
        assert token_data["access_token"] != mock_token_data["access_token"]
        token_data = authenticate(host=test_server, force=True)
        assert isinstance(token_data, dict)
        assert "access_token" in token_data
        token = token_data["access_token"]
        token_data2 = authenticate(host=test_server)
        assert token_data2["access_token"] == token
    finally:
        auth_instance.auth_token = old_token_data
