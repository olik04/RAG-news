from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Hashable
from dataclasses import dataclass, field
from time import monotonic


@dataclass(slots=True)
class SlidingWindowRateLimiter:
    max_requests: int
    window_seconds: int = 60
    _events: dict[Hashable, deque[float]] = field(
        init=False, default_factory=lambda: defaultdict(deque)
    )

    def allow(self, key: Hashable) -> bool:
        now = monotonic()
        events = self._events[key]
        cutoff = now - self.window_seconds

        while events and events[0] <= cutoff:
            events.popleft()

        if len(events) >= self.max_requests:
            return False

        events.append(now)
        return True
