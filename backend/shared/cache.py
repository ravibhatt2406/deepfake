"""
shared/cache.py — Async Redis cache with fallback to in-memory LRU.

Usage:
    cache = Cache(redis_url)
    await cache.get("key")
    await cache.set("key", value, ttl=3600)
    await cache.delete("key")
    await cache.get_or_compute("key", compute_fn, ttl=3600)
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Any, Callable, Optional

logger = logging.getLogger("truthscan.cache")

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis package not installed — falling back to in-memory LRU cache")


class InMemoryLRU:
    """Thread-safe LRU cache with TTL for dev / fallback mode."""

    def __init__(self, maxsize: int = 1024):
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._maxsize = maxsize
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key not in self._store:
                return None
            value, expires_at = self._store[key]
            if expires_at and time.time() > expires_at:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return value

    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        async with self._lock:
            expires_at = time.time() + ttl if ttl else None
            self._store[key] = (value, expires_at)
            self._store.move_to_end(key)
            if len(self._store) > self._maxsize:
                self._store.popitem(last=False)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def close(self) -> None:
        pass


class Cache:
    """
    Unified cache interface. Tries Redis first, falls back to in-memory LRU.
    All values are JSON-serialised, so any JSON-able type is supported.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self._redis: Optional[Any] = None
        self._lru = InMemoryLRU(maxsize=2048)
        self._redis_url = redis_url
        self._use_redis = False
        self._connect_task: Optional[asyncio.Task] = None

    async def _connect(self) -> None:
        if not REDIS_AVAILABLE:
            return
        try:
            client = aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            await client.ping()
            self._redis = client
            self._use_redis = True
            logger.info("✅ Redis cache connected: %s", self._redis_url)
        except Exception as e:
            logger.warning("⚠️  Redis unavailable (%s) — using in-memory LRU", e)

    async def _ensure_connected(self) -> None:
        if self._redis is None and not self._use_redis:
            await self._connect()

    @staticmethod
    def make_key(*parts: str) -> str:
        """Build a deterministic cache key from arbitrary parts."""
        raw = ":".join(str(p) for p in parts)
        return "ts:" + hashlib.sha256(raw.encode()).hexdigest()[:32]

    @staticmethod
    def hash_content(content: bytes | str) -> str:
        if isinstance(content, str):
            content = content.encode()
        return hashlib.sha256(content).hexdigest()

    async def get(self, key: str) -> Optional[Any]:
        await self._ensure_connected()
        try:
            if self._use_redis:
                raw = await self._redis.get(key)
                return json.loads(raw) if raw else None
        except Exception:
            pass
        return await self._lru.get(key)

    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        await self._ensure_connected()
        serialised = json.dumps(value, default=str)
        try:
            if self._use_redis:
                await self._redis.setex(key, ttl, serialised)
                return
        except Exception:
            pass
        await self._lru.set(key, value, ttl)

    async def delete(self, key: str) -> None:
        await self._ensure_connected()
        try:
            if self._use_redis:
                await self._redis.delete(key)
        except Exception:
            pass
        await self._lru.delete(key)

    async def get_or_compute(
        self,
        key: str,
        compute_fn: Callable,
        ttl: int = 3600,
    ) -> Any:
        """
        Cache-aside pattern:
          1. Check cache
          2. On miss → call compute_fn() (sync or async)
          3. Store result
          4. Return
        """
        cached = await self.get(key)
        if cached is not None:
            logger.debug("Cache HIT: %s", key[:20])
            return cached

        logger.debug("Cache MISS: %s", key[:20])
        if asyncio.iscoroutinefunction(compute_fn):
            result = await compute_fn()
        else:
            result = compute_fn()

        await self.set(key, result, ttl)
        return result

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
