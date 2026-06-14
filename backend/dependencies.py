from collections.abc import AsyncGenerator

import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.session import async_session
from backend.pubsub import close_async_redis, get_async_redis


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session


async def get_redis() -> AsyncGenerator[aioredis.Redis, None]:
    yield await get_async_redis()


async def close_redis() -> None:
    await close_async_redis()
