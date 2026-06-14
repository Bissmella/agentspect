import json
import time
from collections.abc import AsyncGenerator
from typing import Any

import redis
import redis.asyncio as aioredis

from backend.config import settings

CHANNEL_PREFIX = "ata:suite:"
HISTORY_SUFFIX = ":history"
HISTORY_TTL_SECONDS = 3600

_seq_counters: dict[str, int] = {}

_sync_redis: redis.Redis | None = None
_async_redis: aioredis.Redis | None = None


def _get_sync_redis() -> redis.Redis:
    global _sync_redis
    if _sync_redis is None:
        _sync_redis = redis.Redis.from_url(
            settings.redis_url, decode_responses=True
        )
    return _sync_redis


def close_sync_redis() -> None:
    global _sync_redis
    if _sync_redis is not None:
        _sync_redis.close()
        _sync_redis = None


async def get_async_redis() -> aioredis.Redis:
    global _async_redis
    if _async_redis is None:
        _async_redis = aioredis.Redis.from_url(
            settings.redis_url, decode_responses=True
        )
    return _async_redis


async def close_async_redis() -> None:
    global _async_redis
    if _async_redis is not None:
        await _async_redis.aclose()
        _async_redis = None


def _channel(suite_id: str) -> str:
    return f"{CHANNEL_PREFIX}{suite_id}"


def _history_key(suite_id: str) -> str:
    return f"{CHANNEL_PREFIX}{suite_id}{HISTORY_SUFFIX}"


def publish_event(
    suite_id: str, event: str, data: dict[str, Any] | None = None
) -> None:
    seq = _seq_counters.get(suite_id, 0) + 1
    _seq_counters[suite_id] = seq

    payload = {
        "event": event,
        "data": data or {},
        "seq": seq,
        "timestamp": time.time(),
    }
    encoded = json.dumps(payload, default=str)

    client = _get_sync_redis()
    client.publish(_channel(suite_id), encoded)
    client.rpush(_history_key(suite_id), encoded)
    client.expire(_history_key(suite_id), HISTORY_TTL_SECONDS)


async def get_event_history(suite_id: str) -> list[dict[str, Any]]:
    client = await get_async_redis()
    raw_events = await client.lrange(_history_key(suite_id), 0, -1)
    return [json.loads(e) for e in raw_events]


async def subscribe_events(
    suite_id: str,
) -> AsyncGenerator[dict[str, Any], None]:
    client = await get_async_redis()
    pubsub = client.pubsub()
    try:
        await pubsub.subscribe(_channel(suite_id))
        async for message in pubsub.listen():
            if message["type"] == "message":
                yield json.loads(message["data"])
    finally:
        await pubsub.unsubscribe(_channel(suite_id))
        await pubsub.aclose()
