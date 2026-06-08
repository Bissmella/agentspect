from collections.abc import AsyncGenerator

import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.db.session import async_session

_redis_pool: aioredis.Redis | None = None


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session


async def get_redis() -> AsyncGenerator[aioredis.Redis, None]:
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = aioredis.Redis.from_url(
            settings.redis_url, decode_responses=True
        )
    yield _redis_pool


async def close_redis() -> None:
    global _redis_pool
    if _redis_pool is not None:
        await _redis_pool.aclose()
        _redis_pool = None
