"""Various utilities for the restAPI."""

from typing import Dict, Optional

import redis.asyncio as redis
from fastapi import HTTPException, status

from freva_rest.config import ServerConfig
from freva_rest.logger import logger

REDIS_CACHE: Optional[redis.Redis] = None
CACHING_SERVICES = set(("zarr-stream",))
"""All the services that need the redis cache."""
CONFIG = ServerConfig()


def get_userinfo(user_info: Dict[str, str]) -> Dict[str, str]:
    """Convert a user_info dictionary to the UserInfo Model."""
    output = {}
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
    return output


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
    if CACHING_SERVICES - CONFIG.services == CACHING_SERVICES:
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
