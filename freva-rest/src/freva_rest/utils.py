"""Various utilities for the restAPI."""

import re
import ssl
from typing import Any, Dict, List, Optional, cast

import jwt
import redis.asyncio as redis
from fastapi import HTTPException, status
from typing_extensions import NotRequired, TypedDict

from freva_rest.config import ServerConfig
from freva_rest.logger import logger

REDIS_CACHE: Optional[redis.Redis] = None
CACHING_SERVICES = set(("zarr-stream",))
"""All the services that need the redis cache."""
CONFIG = ServerConfig()


class SystemUserInfo(TypedDict):
    """Encoded token inforamation."""

    email: NotRequired[str]
    last_name: NotRequired[str]
    first_name: NotRequired[str]
    username: NotRequired[str]


def token_field_matches(token: str) -> bool:
    """
    Flattens the token[key] value to a string and checks if the regex pattern matches.

    Parameters:
        token (dict): The encoded JWT token.

    Returns:
        bool: True if a match is found, False otherwise.
    """

    def _walk_dict(inp: Any, keys: List[str]) -> Any:
        if not keys or not isinstance(inp, dict) or not inp:
            return inp or ""
        return _walk_dict(inp.get(keys[0]), keys[1:])

    matches: List[bool] = []
    token_data: Dict[str, Any] = {}
    for claim, pattern in (CONFIG.oidc_token_claims or {}).items():
        if not token_data:
            token_data = jwt.decode(token, options={"verify_signature": False})
        value_str = str(_walk_dict(token_data, claim.split(".")))
        for p in pattern:
            matches.append(
                bool(re.search(rf"\b{re.escape(str(p))}\b", value_str))
            )
    return all(matches)


def get_userinfo(
    user_info: Dict[str, str],
) -> SystemUserInfo:
    """Convert a user_info dictionary to the UserInfo Model."""
    output: Dict[str, str] = {}
    keys = {
        "email": ("mail", "email"),
        "username": ("preferred-username", "user-name", "uid"),
        "last_name": ("last-name", "family-name", "name", "surname"),
        "first_name": ("first-name", "given-name"),
    }
    for key, entries in keys.items():
        for entry in entries:
            if user_info.get(entry):
                output[key] = user_info[entry]
                break
            if user_info.get(entry.replace("-", "_")):
                output[key] = user_info[entry.replace("-", "_")]
                break
            if user_info.get(entry.replace("-", "")):
                output[key] = user_info[entry.replace("-", "")]
                break
    # Strip all the middle names
    name = output.get("first_name", "") + " " + output.get("last_name", "")
    output["first_name"] = name.partition(" ")[0]
    output["last_name"] = name.rpartition(" ")[-1]
    return cast(SystemUserInfo, output)


async def create_redis_connection(
    cache: Optional[redis.Redis] = REDIS_CACHE,
) -> redis.Redis:
    """Reuse a potentially created redis connection."""
    kwargs = dict(
        host=CONFIG.redis_url,
        port=CONFIG.redis_port,
        username=CONFIG.redis_user or None,
        password=CONFIG.redis_password or None,
        ssl=CONFIG.redis_ssl_certfile is not None,
        ssl_certfile=CONFIG.redis_ssl_certfile or None,
        ssl_keyfile=CONFIG.redis_ssl_keyfile or None,
        ssl_ca_certs=CONFIG.redis_ssl_certfile or None,
        db=0,
    )
    if CACHING_SERVICES - set(CONFIG.services or []) == CACHING_SERVICES:
        # All services that would need caching are disabled.
        # If this is the case and we ended up here, we shouldn't be here.
        # tell the users.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not enabled.",
        )

    if cache is None:
        logger.info("Creating redis connection using: %s", kwargs)
    cache = cache or redis.Redis(
        host=CONFIG.redis_url,
        port=CONFIG.redis_port,
        username=CONFIG.redis_user or None,
        password=CONFIG.redis_password or None,
        ssl=CONFIG.redis_ssl_certfile is not None,
        ssl_certfile=CONFIG.redis_ssl_certfile or None,
        ssl_keyfile=CONFIG.redis_ssl_keyfile or None,
        ssl_ca_certs=CONFIG.redis_ssl_certfile or None,
        ssl_cert_reqs=ssl.CERT_NONE,
        db=0,
    )
    try:
        await cache.ping()
    except Exception as error:
        logger.error("Cloud not connect to redis cache: %s", error)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache gone.",
        ) from None
    return cache


def str_to_int(inp_str: Optional[str], default: int) -> int:
    """Convert an integer from a string. If it's not working return default."""
    inp_str = inp_str or ""
    try:
        return int(inp_str)
    except ValueError:
        return default
