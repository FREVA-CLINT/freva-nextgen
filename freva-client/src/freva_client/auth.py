"""Module that handles the authentication at the rest service."""

import datetime
import json
import socket
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Dict, List, Optional, Union

import requests

from .utils import logger
from .utils.auth_utils import (
    Token,
    choose_token_strategy,
    get_default_token_file,
    load_token,
    wait_for_port,
)
from .utils.databrowser_utils import Config

REDIRECT_URI = "http://localhost:{port}/callback"
AUTH_FAILED_MSG = """Login failed or no valid token found in this environment.
If you appear to be in a non-interactive or remote session create a new token
via:

   {command}

Then pass the token file using:"

"    {token_path} or set the $TOKEN_ENV_VAR env. variable."""


class AuthError(Exception):
    """Athentication error."""

    pass


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        logger.debug(format, *args)

    def do_GET(self) -> None:
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)
        if "code" in params:
            setattr(self.server, "auth_code", params["code"][0])
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Login successful! You can close this tab.")
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Authorization code not found.")


def start_local_server(port: int, event: Event) -> HTTPServer:
    """Start local HTTP server to wait for a single callback."""
    server = HTTPServer(("localhost", port), OAuthCallbackHandler)

    def handle() -> None:
        logger.info("Waiting for browser callback on port %s ...", port)
        while not event.is_set():
            server.handle_request()
            if getattr(server, "auth_code", None):
                event.set()

    thread = Thread(target=handle, daemon=True)
    thread.start()
    return server


class Auth:
    """Helper class for authentication."""

    _instance: Optional["Auth"] = None
    _auth_token: Optional[Token] = None
    _thread_lock: Lock = Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "Auth":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, token_file: Optional[Union[str, Path]] = None) -> None:
        self.token_file = str(token_file or "").strip() or None

    def get_token(
        self,
        token_url: str,
        data: Dict[str, str],
    ) -> Token:
        try:
            response = requests.post(token_url, data=data)
            response.raise_for_status()
        except requests.exceptions.RequestException as error:
            raise AuthError(f"Fetching token failed: {error}")
        auth = response.json()
        return self.set_token(
            access_token=auth["access_token"],
            token_type=auth["token_type"],
            expires=auth["expires"],
            refresh_token=auth["refresh_token"],
            refresh_expires=auth["refresh_expires"],
            scope=auth["scope"],
        )

    def _login(
        self,
        auth_url: str,
        force: bool = False,
        _timeout: int = 30,
    ) -> Token:
        login_endpoint = f"{auth_url}/login"
        token_endpoint = f"{auth_url}/token"
        port = self.find_free_port(auth_url)
        redirect_uri = REDIRECT_URI.format(port=port)
        params = {
            "redirect_uri": redirect_uri,
        }
        if force:
            params["prompt"] = "login"
        query = urllib.parse.urlencode(params)
        login_url = f"{login_endpoint}?{query}"
        logger.info("Opening browser for login:\n%s", login_url)
        logger.info(
            "If you are using this on a remote host, you might need to "
            "forward port %d:\n"
            "    ssh -L %d:localhost:%d user@remotehost",
            port,
            port,
            port,
        )
        event = Event()
        server = start_local_server(port, event)
        code: Optional[str] = None
        reason = "Login failed."
        try:
            wait_for_port("localhost", port)
            webbrowser.open(login_url)
            success = event.wait(timeout=_timeout)
            if not success:
                raise TimeoutError(
                    f"Login did not complete within {_timeout} seconds. "
                    "Possibly headless environment."
                )
            code = getattr(server, "auth_code", None)
        except Exception as error:
            logger.warning(
                "Could not open browser automatically. %s"
                "Please open the URL manually.",
                error,
            )
            reason = str(error)

        finally:
            logger.debug("Cleaning up login state")
            if hasattr(server, "server_close"):
                try:
                    server.server_close()
                except Exception as error:
                    logger.debug("Failed to close server cleanly: %s", error)
        if not code:
            raise AuthError(reason)
        return self.get_token(
            token_endpoint,
            data={
                "code": code,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        )

    @staticmethod
    def find_free_port(auth_url: str) -> int:
        """Get a free port where we can start the test server."""
        ports: List[int] = (
            requests.get(f"{auth_url}/auth-ports").json().get("valid_ports", [])
        )
        for port in ports:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("localhost", port))
                    return port
                except (OSError, PermissionError):
                    pass
        raise OSError("No free ports available for login flow")

    def set_token(
        self,
        access_token: str,
        refresh_token: Optional[str] = None,
        expires_in: int = 10,
        refresh_expires_in: int = 10,
        expires: Optional[Union[float, int]] = None,
        refresh_expires: Optional[Union[float, int]] = None,
        token_type: str = "Bearer",
        scope: str = "profile email address",
    ) -> Token:
        """Override the existing auth token."""
        now = datetime.datetime.now(datetime.timezone.utc).timestamp()

        self._auth_token = Token(
            access_token=access_token or "",
            refresh_token=refresh_token or "",
            token_type=token_type,
            expires=int(expires or now + expires_in),
            refresh_expires=int(refresh_expires or now + refresh_expires_in),
            scope=scope,
        )
        default_token_file = get_default_token_file()
        for _file in map(
            Path, set((self.token_file or default_token_file, default_token_file))
        ):
            _file.parent.mkdir(exist_ok=True, parents=True)
            _file.write_text(json.dumps(self._auth_token))
            _file.chmod(0o600)

        return self._auth_token

    def _refresh(
        self,
        url: str,
        refresh_token: str,
    ) -> Token:
        """Refresh the access_token with a refresh token."""
        try:
            return self.get_token(
                f"{url}/token",
                data={"refresh-token": refresh_token or ""},
            )
        except (AuthError, KeyError) as error:
            logger.warning("Failed to refresh token: %s", error)
            return self._login(url)

    def authenticate(
        self,
        host: Optional[str] = None,
        config: Optional[Config] = None,
        *,
        force: bool = False,
        _cli: bool = False,
    ) -> Token:
        """Authenticate the user to the host."""
        cfg = config or Config(host)
        token = self._auth_token or load_token(self.token_file)
        reason: Optional[str] = None
        if force:
            strategy = "browser_auth"
        else:
            strategy = choose_token_strategy(token)
        if strategy == "use_token" and token:
            self._auth_token = token
            return self._auth_token
        try:
            if strategy == "refresh_token" and token:
                return self._refresh(cfg.auth_url, token["refresh_token"])
            if strategy == "browser_auth":
                return self._login(cfg.auth_url, force=force)
        except AuthError as error:
            reason = str(error)

        command, token_path = {
            True: ("freva-client auth", "--token-file /path/to/token.json"),
            False: (
                "freva_client.auth",
                "`token_file='/path/to/token.json'`",
            ),
        }[_cli]

        reason = reason or AUTH_FAILED_MSG.format(
            command=command, token_path=token_path
        )
        if _cli:
            logger.critical(reason)
            raise SystemExit(1)
        else:
            raise AuthError(reason)


def authenticate(
    *,
    token_file: Optional[Union[Path, str]] = None,
    host: Optional[str] = None,
    force: bool = False,
) -> Token:
    """Authenticate to the host.

    This method generates a new access token that should be used for restricted methods.

    Parameters
    ----------
    refresh_token: str, optional
        Instead of setting a password, you can set a refresh token to refresh
        the access token. This is recommended for non-interactive environments.
    host: str, optional
        The hostname of the REST server.
    force: bool, default: False
        Force token recreation, even if current token is still valid.

    Returns
    -------
    Token: The authentication token.

    Examples
    --------
    Interactive authentication:

    .. code-block:: python

        from freva_client import authenticate
        token = authenticate()
        print(token)

    Batch mode authentication with a refresh token:

    .. code-block:: python

        from freva_client import authenticate
        token = authenticate(token_file="~/.freva-login-token.json")
    """
    auth = Auth(token_file=token_file or None)
    return auth.authenticate(
        host=host,
        force=force,
    )
