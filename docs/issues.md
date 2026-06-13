
- in pubsub.py in subscribe_events implementation, every single user connections is opening a new TCP connection to Redis:
 `client = aioredis.Redis.from_url(settings.redis_url, decode_responses=True)`
 Redis has a default limit on concurrent connection (usually 10,000). if there are a lot of users it might crash.
 possible fix: initialize one global redis client when FastAPI starts and share it across all websockets, or use a shared connection pool.
 if the current implementation is done as is intentionally then provide explanation.
