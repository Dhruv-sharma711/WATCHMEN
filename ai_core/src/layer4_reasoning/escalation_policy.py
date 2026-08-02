"""
Layer 4 - Escalation Policy

Watches risk scores and decides when to fire the (expensive) Layer 5 VLM call.
Critically: once a shopper has been escalated, we must NOT re-escalate every
subsequent frame just because risk is still above threshold - that would spam
Layer 5 with duplicate calls for the same incident. So we track "already
escalated" state per shopper and only re-arm it after a cooldown or a reset.

Now category-aware: each threat category (concealment, weapon, violence,
collusion) uses its own threshold, matching Table 2's "category-specific
confidence bars" - a lower bar for visually subtle behaviours like
concealment, since CRBA's per-class error analysis (Section 3.7) shows
these are the hardest to classify correctly, and a low bar for weapons
since a missed weapon is far costlier than one extra VLM call.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import time

from .confidence_state import DEFAULT_CATEGORY_BARS, category_for


@dataclass
class EscalationDecision:
    should_escalate: bool
    reason: Optional[str] = None
    category: Optional[str] = None


class EscalationPolicy:
    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        cooldown_seconds: float = 60.0,
    ):
        """
        thresholds: risk score (tau) per category that triggers escalation.
                    Defaults to the escalate_at values in
                    confidence_state.DEFAULT_CATEGORY_BARS, so the two
                    modules can't silently drift out of sync. Pass a plain
                    {"default": 1.0} dict here if you want the old
                    single-threshold-for-everyone behaviour back.
        cooldown_seconds: minimum time before the SAME shopper can escalate
                          again (e.g. if their risk dips below threshold then
                          climbs again - a genuinely new incident, not a
                          duplicate alert).
        """
        self.thresholds = thresholds or {
            cat: bar.escalate_at for cat, bar in DEFAULT_CATEGORY_BARS.items()
        }
        self.cooldown_seconds = cooldown_seconds
        self._last_escalated_at: Dict[str, float] = {}
        self._currently_escalated: Dict[str, bool] = {}

    def _threshold_for(self, event_type: str) -> float:
        category = category_for(event_type)
        return self.thresholds.get(category, self.thresholds.get("default", 1.0))

    def check(
        self,
        shopper_id: str,
        current_risk: float,
        event_type: str,
        now: Optional[float] = None,
    ) -> EscalationDecision:
        now = now if now is not None else time.time()
        category = category_for(event_type)
        threshold = self._threshold_for(event_type)

        if current_risk < threshold:
            # Risk dropped back down - clear the "currently escalated" flag so
            # a future crossing of the threshold counts as a fresh incident.
            self._currently_escalated[shopper_id] = False
            return EscalationDecision(should_escalate=False, reason="below_threshold", category=category)

        already_escalated = self._currently_escalated.get(shopper_id, False)
        if already_escalated:
            return EscalationDecision(
                should_escalate=False, reason="already_escalated_this_incident", category=category
            )

        last_time = self._last_escalated_at.get(shopper_id)
        if last_time is not None and (now - last_time) < self.cooldown_seconds:
            return EscalationDecision(should_escalate=False, reason="cooldown_active", category=category)

        # Fires exactly once per incident.
        self._currently_escalated[shopper_id] = True
        self._last_escalated_at[shopper_id] = now
        return EscalationDecision(should_escalate=True, reason="threshold_crossed", category=category)