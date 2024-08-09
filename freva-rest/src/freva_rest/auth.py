"""Definition of routes for authentication."""

import datetime
import os
from typing import Annotated, Any, Dict, Literal, Optional, cast

import aiohttp
from fastapi import Form, HTTPException, Request, Security
from fastapi.responses import RedirectResponse
from fastapi_third_party_auth import Auth, IDToken
from pydantic import BaseModel, Field, ValidationError

from .logger import logger
from .rest import app, server_config
from .utils import get_userinfo

auth = Auth(openid_connect_url=server_config.oidc_discovery_url)

Required: Any = Ellipsis

TIMEOUT: aiohttp.ClientTimeout = aiohttp.ClientTimeout(total=5)
"""5 seconds for timeout for key cloak interaction."""


class UserInfo(BaseModel):
    """Basic user info."""

    username: Annotated[str, Field(min_length=1)]
    last_name: Annotated[str, Field(min_length=1)]
    first_name: Annotated[str, Field(min_length=1)]
    email: str


class TokenPayload(BaseModel):
    """Model representing the payload of a JWT token."""

    sub: str
    exp: int
    email: Optional[str] = None


class Token(BaseModel):
    """Model representing an OAuth2 token response."""

    access_token: str
    token_type: str
    expires: int
    refresh_token: str
    refresh_expires: int
    scope: str


@app.get("/api/auth/v2/status", tags=["Authentication"])
async def get_token_status(id_token: IDToken = Security(auth.required)) -> TokenPayload:
    """Check the status of an access token."""
    return cast(TokenPayload, id_token)


@app.get("/api/auth/v2/userinfo", tags=["Authentication"])
async def userinfo(
    id_token: IDToken = Security(auth.required), request: Request = Required
) -> UserInfo:
    """Get userinfo for the current token."""
    token_data = {k.lower(): str(v) for (k, v) in dict(id_token).items()}
    try:
        return UserInfo(**get_userinfo(token_data))
    except ValidationError:
        authorization = dict(request.headers)["authorization"]
        try:
            async with aiohttp.ClientSession(timeout=TIMEOUT) as client:
                response = await client.get(
                    server_config.oidc_overview["userinfo_endpoint"],
                    headers={"Authorization": authorization},
                )
                response.raise_for_status()
                token_data = await response.json()
                return UserInfo(
                    **get_userinfo({k.lower(): str(v) for (k, v) in token_data.items()})
                )
        except Exception as error:
            logger.error(error)
            raise HTTPException(status_code=404) from error


@app.get(
    "/api/auth/v2/.well-known/openid-configuration",
    tags=["Authentication"],
    response_class=RedirectResponse,
)
async def open_id_config() -> RedirectResponse:
    """Get the open id connect configuration."""
    return RedirectResponse(server_config.oidc_discovery_url)


@app.post("/api/auth/v2/token", tags=["Authentication"])
async def fetch_or_refresh_token(
    username: Annotated[
        Optional[str],
        Form(
            title="Username",
            help="Username to create a OAuth2 token.",
        ),
    ] = None,
    password: Annotated[
        Optional[str],
        Form(
            title="Password",
            help="Password to create a OAuth2 token.",
        ),
    ] = None,
    grant_type: Annotated[
        Literal["password", "refresh_token"],
        Form(
            title="Grant type",
            alias="grant_type",
            help="The authorization code grant type.",
        ),
    ] = "password",
    refresh_token: Annotated[
        Optional[str],
        Form(
            title="Refresh token",
            alias="refresh-token",
            help="The refresh token used to renew the OAuth2 token",
        ),
    ] = None,
    client_id: Annotated[
        Optional[str],
        Form(
            title="Client id",
            alias="client_id",
            help="The client id that is used for the refresh token",
        ),
    ] = None,
    client_secret: Annotated[
        Optional[str],
        Form(
            title="Client secret",
            alias="client_secret",
            help="The client secret that is used for the refresh token",
        ),
    ] = None,
) -> Token:
    """Interact with the openID connect endpoint for client authentication."""
    data: Dict[str, Optional[str]] = {
        "client_id": (client_id or "").replace("None", "") or server_config.oidc_client,
        "client_secret": client_secret or os.getenv("OIDC_CLIENT_SECRET", ""),
        "grant_type": grant_type,
    }
    if grant_type == "password":
        data["password"] = password
        data["username"] = username
    else:
        data["refresh_token"] = refresh_token
    async with aiohttp.ClientSession(timeout=TIMEOUT) as client:
        try:
            response = await client.post(
                server_config.oidc_overview["token_endpoint"],
                data={k: v for (k, v) in data.items() if v},
            )
            response.raise_for_status()
        except Exception as error:
            logger.error(error)
            raise HTTPException(status_code=404)
        token_data = await response.json()
        expires_at = (
            token_data.get("exp")
            or token_data.get("expires")
            or token_data.get("expires_at")
        )
        now = datetime.datetime.now(datetime.timezone.utc).timestamp()
        refresh_expires_at = (
            token_data.get("refresh_exp")
            or token_data.get("refresh_expires")
            or token_data.get("refresh_expires_at")
        )
        expires_at = expires_at or now + token_data.get("expires_in", 180)
        refresh_expires_at = refresh_expires_at or now + token_data.get(
            "refresh_expires_in", 180
        )
    return Token(
        access_token=token_data["access_token"],
        token_type=token_data["token_type"],
        expires=int(expires_at),
        refresh_token=token_data["refresh_token"],
        refresh_expires=int(refresh_expires_at),
        scope=token_data["scope"],
    )
