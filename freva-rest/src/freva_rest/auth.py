"""Definition of routes for authentication."""

import asyncio
import datetime
from typing import (
    Annotated,
    Any,
    Awaitable,
    Callable,
    Dict,
    Literal,
    Optional,
    cast,
)

import aiohttp
from fastapi import Depends, Form, HTTPException, Request, Security
from fastapi.responses import RedirectResponse
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
    SecurityScopes,
)
from fastapi_third_party_auth import Auth, IDToken
from pydantic import BaseModel, Field, ValidationError

from .logger import logger
from .rest import app, server_config
from .utils import get_userinfo

Required: Any = Ellipsis

TIMEOUT: aiohttp.ClientTimeout = aiohttp.ClientTimeout(total=5)
"""5 seconds for timeout for key cloak interaction."""


class SafeAuth:
    """
    A wrapper around fastapi_third_party_auth.Auth that safely delays
    initialization until the OIDC discovery URL is reachable.

    This allows FastAPI routes to use the Auth.required() dependency without
    failing at application startup if the OIDC server is temporarily
    unavailable.
    """

    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self, discovery_url: Optional[str] = None) -> None:
        """
        Initialize the SafeAuth wrapper.

        Parameters:
            discovery_url (str): The full URL to the OIDC discovery document,
                                 e.g., "https://issuer/.well-known/openid-configuration"
        """
        self.discovery_url: str = (discovery_url or "").strip()
        self._auth: Optional[Auth] = None

    async def _check_server_available(self) -> bool:
        """
        Check whether the OIDC server is reachable by requesting the
            discovery document.

        Returns
        -------
            bool: True if the server is up and the document is reachable,
                  False otherwise.
        """
        if not self.discovery_url:
            return False
        try:
            async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
                async with session.get(self.discovery_url) as response:
                    return response.status == 200
        except aiohttp.ClientError:
            return False

    async def _ensure_auth_initialized(self) -> None:
        """
        Initialize the internal Auth instance if the server is available
        and not yet initialized.
        """
        async with self._lock:
            if self._auth is None and await self._check_server_available():
                self._auth = Auth(self.discovery_url)

    def required_dependency(
        self,
    ) -> Callable[
        [SecurityScopes, Optional[HTTPAuthorizationCredentials]],
        Awaitable[IDToken],
    ]:
        """
        Return a FastAPI dependency function to validate a token.

        Returns
        -------
            Callable: A dependency function to use with `Security(...)` in
                      FastAPI routes.

        Raises
        ------
        HTTPException: 503 if the auth server is not available
        """

        async def dependency(
            security_scopes: SecurityScopes,
            authorization_credentials: Optional[
                HTTPAuthorizationCredentials
            ] = Depends(HTTPBearer()),
        ) -> IDToken:
            await self._ensure_auth_initialized()

            if self._auth is None:
                raise HTTPException(
                    status_code=503,
                    detail="OIDC server unavailable, cannot validate token.",
                )

            return self._auth.required(security_scopes, authorization_credentials)

        return dependency


auth = SafeAuth(server_config.oidc_discovery_url)


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


@app.get("/api/freva-nextgen/auth/v2/status", tags=["Authentication"])
async def get_token_status(
    id_token: IDToken = Security(auth.required_dependency()),
) -> TokenPayload:
    """Check the status of an access token."""
    return cast(TokenPayload, id_token)


@app.get("/api/freva-nextgen/auth/v2/userinfo", tags=["Authentication"])
async def userinfo(
    id_token: IDToken = Security(auth.required_dependency()),
    request: Request = Required,
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
                    **get_userinfo(
                        {k.lower(): str(v) for (k, v) in token_data.items()}
                    )
                )
        except Exception as error:
            logger.error(error)
            raise HTTPException(status_code=404) from error


@app.get(
    "/api/freva-nextgen/auth/v2/.well-known/openid-configuration",
    tags=["Authentication"],
    response_class=RedirectResponse,
)
async def open_id_config() -> RedirectResponse:
    """Get the open id connect configuration."""
    return RedirectResponse(server_config.oidc_discovery_url)


@app.post("/api/freva-nextgen/auth/v2/token", tags=["Authentication"])
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
        "client_id": (client_id or "").replace("None", "")
        or server_config.oidc_client_id,
        "client_secret": client_secret or server_config.oidc_client_secret,
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
            raise HTTPException(status_code=404) from None
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
