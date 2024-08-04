"""Various utilities for the restAPI."""

import os
from typing import Optional

import redis.asyncio as redis
from fastapi import HTTPException, status
from freva_rest.logger import logger

REDIS_CACHE: Optional[redis.Redis] = None
REDIS_HOST, _, REDIS_PORT = (
    (os.environ.get("REDIS_HOST") or "localhost")
    .removeprefix("redis://")
    .partition(":")
)
REDIS_SSL_CERTFILE = os.getenv("REDIS_SSL_CERTFILE") or None
REDIS_SSL_KEYFILE = os.getenv("REDIS_SSL_KEYFILE") or None
CACHING_SERVICES = set(("zarr-stream",))
"""All the services that need the redis cache."""


async def create_redis_connection(
    cache: Optional[redis.Redis] = REDIS_CACHE,
) -> redis.Redis:
    """Reuse a potentially created redis connection."""
    kwargs = dict(
        host=REDIS_HOST,
        username=os.getenv("REDIS_USER"),
        password=os.getenv("REDIS_PASS"),
        port=int(REDIS_PORT or "6379"),
        ssl=REDIS_SSL_CERTFILE is not None,
        ssl_certfile=REDIS_SSL_CERTFILE,
        ssl_keyfile=REDIS_SSL_KEYFILE,
        ssl_ca_certs=REDIS_SSL_CERTFILE,
        db=0,
    )
    services = set(
        [
            s.strip()
            for s in os.getenv("API_SERVICES", "").split(",")
            if s.strip()
        ]
    )
    if CACHING_SERVICES - services == CACHING_SERVICES:
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
        host=REDIS_HOST,
        username=os.getenv("REDIS_USER"),
        password=os.getenv("REDIS_PASS"),
        port=int(REDIS_PORT or "6379"),
        ssl=REDIS_SSL_CERTFILE is not None,
        ssl_certfile=REDIS_SSL_CERTFILE,
        ssl_keyfile=REDIS_SSL_KEYFILE,
        ssl_ca_certs=REDIS_SSL_CERTFILE,
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
