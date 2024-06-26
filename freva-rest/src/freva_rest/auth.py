"""Definition of routes for authentication."""

import os
from typing import Annotated, Optional

import aiohttp
import jwt
from fastapi import Depends, Form, HTTPException, status
from fastapi.security import (
    OAuth2AuthorizationCodeBearer,
    OAuth2PasswordRequestForm,
)
from pydantic import BaseModel

from .logger import logger
from .rest import app, server_config

TIMEOUT: aiohttp.ClientTimeout = aiohttp.ClientTimeout(total=5)
"""5 seconds for timeout for key cloak interaction."""


class KeycloakOAuth2RequestForm(OAuth2PasswordRequestForm):
    """A password request form for keycloak with a client_id."""

    client_id: str
    refresh_token: str = ""
    username: str = ""
    password: str = ""


class Token(BaseModel):
    """Model representing an OAuth2 token response from Keycloak."""

    access_token: str
    token_type: str
    expires_in: int
    refresh_token: str
    refresh_expires_in: int


async def get_token(
    client_id: str = "",
    username: str = "",
    password: str = "",
    refresh_token: str = "",
) -> Token:
    """
    Retrieve an OAuth2 token from Keycloak using the Resource Owner Password
    Credentials grant type.

    Parameters
    -----------

    username (str):
        The username of the user.
    password (str):
        The password of the user.
    client_id (str):
        Name of the specific application.


    Returns:
        Token: An object containing the access token and related information.

    Raises:
        HTTPException: If the response from Keycloak is not successful.
    """
    if username and password:
        data = {
            "client_id": client_id or os.getenv("KEYCLOAK_CLIENT_ID", "freva"),
            "grant_type": "password",
            "password": password,
            "username": username,
        }
    else:
        data = {
            "client_id": client_id or os.getenv("KEYCLOAK_CLIENT_ID", "freva"),
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
    if os.getenv("KEYCLOAK_CLIENT_SECRET"):
        data["client_secret"] = os.getenv(
            "KEYCLOAK_CLIENT_SECRET", ""
        )  # pragma: no cover
    async with aiohttp.ClientSession(timeout=TIMEOUT) as client:
        response = await client.post(
            server_config.keycloak_overview["token_endpoint"], data=data
        )
        if response.status != 200:
            raise HTTPException(
                status_code=response.status,
                detail="Invalid username or password",
            )
        return Token(**(await response.json()))


oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl=server_config.keycloak_overview["authorization_endpoint"],
    tokenUrl=server_config.keycloak_overview["token_endpoint"],
)


class TokenPayload(BaseModel):
    """Model representing the payload of a JWT token."""

    sub: str
    exp: int
    email: Optional[str] = None


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    verify_exp: bool = True,
) -> TokenPayload:
    """Get the current user by verifying the JWT token.

    Parameters
    ----------
    token (str):
        The JWT token from the Authorization header.

    Returns
    -------
    TokenPayload: The payload data from the token.

    Raises
    ------
    HTTPException: If the token is invalid or verification fails.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        jwks_client = jwt.PyJWKClient(
            server_config.keycloak_overview["jwks_uri"],
            headers={"User-agent": "custom-user-agent"},
        )
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            options={"verify_exp": verify_exp, "verify_aud": False},
        )
        token_data = TokenPayload(**payload)
    except jwt.exceptions.PyJWTError as error:
        logger.critical(error)
        raise credentials_exception from None

    return token_data


async def get_user_status(token: str = Depends(oauth2_scheme)) -> TokenPayload:
    """Get the status of an existing user token."""
    return await get_current_user(token, verify_exp=False)


@app.post(
    "/api/auth/v2/refresh", response_model=Token, tags=["Authentication"]
)
async def refresh_access_token(
    refresh_token: Annotated[
        str,
        Form(
            title="Refresh token",
            alias="refresh-token",
            help="The refresh token used to renew the OAuth2 token",
        ),
    ],
    client_id: Annotated[
        Optional[str],
        Form(
            title="Client id",
            alias="client_id",
            help="The client id that is used for the refresh token",
        ),
    ] = None,
) -> Token:
    """Renew the login token with help of a refresh token."""
    return await get_token(
        client_id=client_id or "", refresh_token=refresh_token
    )


@app.get("/api/auth/v2/status", tags=["Authentication"])
async def get_token_status(
    current_user: TokenPayload = Depends(get_user_status),
) -> TokenPayload:
    """Check the status of an access token."""
    return current_user


@app.post("/api/auth/v2/token/", response_model=Token, tags=["Authentication"])
async def login_for_access_token(
    form_data: KeycloakOAuth2RequestForm = Depends(),
) -> Token:
    """Create an new login token from a username and password.

    You should either set a username and password or an existing refresh token.
    You can also set the client_id. Client id's are configured to gain access,
    specific access for certain users. If you don't set the client_id, the
    default id will be chosen.

    """
    return await get_token(
        client_id=form_data.client_id or "",
        username=form_data.username,
        password=form_data.password,
    )
