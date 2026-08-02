"""
Layer 5 - Rate Limiter

A minimal sliding-window rate limiter so the (expensive) VLM is never
called more often than the budget allows, regardless of how many shoppers
Layer 4 escalates at once. Table 2 describes Layer 5 as "rate-limited" -
this is the piece of code that enforces that word.
"""

import time
from collections import deque
from typing import Deque, Optional


class RateLimiter:
    def __init__(self, max_calls: int, per_seconds: float):
        self.max_calls = max_calls
        self.per_seconds = per_seconds
        self._timestamps: Deque[float] = deque()

    def allow(self, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.time()
        while self._timestamps and now - self._timestamps[0] > self.per_seconds:
            self._timestamps.popleft()
        if len(self._timestamps) < self.max_calls:
            self._timestamps.append(now)
            return True
        return False