"""
Lightweight in-process async message bus.

Ported verbatim from forex-trading/python/core/bus.py — this module is
entirely asset-agnostic (generic pub/sub over string topics), so no
US-equity-specific changes were needed.

Usage:
    bus = MessageBus()
    bus.subscribe("snapshot", my_coroutine)
    await bus.publish("snapshot", snapshot_obj)

Each subscriber is called as an asyncio Task so that a slow subscriber
cannot block the publisher or other subscribers.
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)

Subscriber = Callable[[Any], Awaitable[None]]


class MessageBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Subscriber]] = defaultdict(list)
        self._published: int = 0
        self._errors: int = 0

    # ── Registration ─────────────────────────────────────────────────────────

    def subscribe(self, topic: str, callback: Subscriber) -> None:
        self._subs[topic].append(callback)
        log.debug("subscribed %s → %s", callback.__qualname__, topic)

    def unsubscribe(self, topic: str, callback: Subscriber) -> None:
        self._subs[topic] = [s for s in self._subs[topic] if s is not callback]

    # ── Dispatch ─────────────────────────────────────────────────────────────

    async def publish(self, topic: str, message: Any) -> None:
        """Fire-and-forget: each subscriber runs as an independent Task."""
        self._published += 1
        subscribers = self._subs.get(topic, [])
        for sub in subscribers:
            asyncio.create_task(self._safe_call(sub, message))

    async def publish_wait(self, topic: str, message: Any) -> None:
        """Publish and await all subscribers (useful for backtest replay)."""
        self._published += 1
        for sub in self._subs.get(topic, []):
            await self._safe_call(sub, message)

    async def _safe_call(self, sub: Subscriber, message: Any) -> None:
        try:
            await sub(message)
        except Exception:
            self._errors += 1
            log.exception("subscriber %s raised on message %r", sub.__qualname__, message)

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "published": self._published,
            "errors": self._errors,
            "topics": list(self._subs.keys()),
        }
