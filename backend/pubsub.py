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

    client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        client.publish(_channel(suite_id), encoded)
        client.rpush(_history_key(suite_id), encoded)
        client.expire(_history_key(suite_id), HISTORY_TTL_SECONDS)
    finally:
        client.close()


async def get_event_history(suite_id: str) -> list[dict[str, Any]]:
    client = aioredis.Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        raw_events = await client.lrange(_history_key(suite_id), 0, -1)
        return [json.loads(e) for e in raw_events]
    finally:
        await client.aclose()


async def subscribe_events(
    suite_id: str,
) -> AsyncGenerator[dict[str, Any], None]:
    client = aioredis.Redis.from_url(settings.redis_url, decode_responses=True)
    pubsub = client.pubsub()
    try:
        await pubsub.subscribe(_channel(suite_id))
        async for message in pubsub.listen():
            if message["type"] == "message":
                yield json.loads(message["data"])
    finally:
        await pubsub.unsubscribe(_channel(suite_id))
        await pubsub.aclose()
        await client.aclose()
