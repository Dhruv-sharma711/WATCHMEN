"""
Layer 4 - Escalation Policy

Watches risk scores and decides when to fire the (expensive) Layer 5 VLM call.
Critically: once a shopper has been escalated, we must NOT re-escalate every
subsequent frame just because risk is still above threshold - that would spam
Layer 5 with duplicate calls for the same incident. So we track "already
escalated" state per shopper and only re-arm it after a cooldown or a reset.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import time


@dataclass
class EscalationDecision:
    should_escalate: bool
    reason: Optional[str] = None


class EscalationPolicy:
    def __init__(self, threshold: float = 1.0, cooldown_seconds: float = 60.0):
        """
        threshold: risk score (tau) that triggers escalation
        cooldown_seconds: minimum time before the SAME shopper can escalate again
                          (e.g. if their risk dips below threshold then climbs again -
                          a genuinely new incident, not a duplicate alert)
        """
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self._last_escalated_at: Dict[str, float] = {}
        self._currently_escalated: Dict[str, bool] = {}

    def check(self, shopper_id: str, current_risk: float, now: Optional[float] = None) -> EscalationDecision:
        now = now if now is not None else time.time()

        if current_risk < self.threshold:
            # Risk dropped back down - clear the "currently escalated" flag so
            # a future crossing of the threshold counts as a fresh incident.
            self._currently_escalated[shopper_id] = False
            return EscalationDecision(should_escalate=False, reason="below_threshold")

        already_escalated = self._currently_escalated.get(shopper_id, False)
        if already_escalated:
            return EscalationDecision(should_escalate=False, reason="already_escalated_this_incident")

        last_time = self._last_escalated_at.get(shopper_id)
        if last_time is not None and (now - last_time) < self.cooldown_seconds:
            return EscalationDecision(should_escalate=False, reason="cooldown_active")

        # Fires exactly once per incident.
        self._currently_escalated[shopper_id] = True
        self._last_escalated_at[shopper_id] = now
        return EscalationDecision(should_escalate=True, reason="threshold_crossed")