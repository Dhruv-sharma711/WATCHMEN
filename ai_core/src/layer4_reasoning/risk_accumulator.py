"""
Layer 4 - Risk Accumulator

Implements: R(id, t) = lambda * R(id, t-1) + sum(w_i * evidence_i(t))

Each tracked shopper has a running risk score. Old evidence decays over time
(the lambda term); new evidence is added in, weighted by how serious it is.
This module does NOT decide suppression/escalation multipliers - that's the
Symbolic Context Arbiter's job. This module just tracks and updates the number.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time


@dataclass
class RiskEvent:
    """A single logged update, kept for audit trail / explainability (Layer 6)."""
    shopper_id: str
    timestamp: float
    event_type: str
    raw_confidence: float
    weight_applied: float
    multiplier_applied: float
    evidence_contribution: float
    risk_after: float


class RiskAccumulator:
    """
    Tracks a decaying risk score per shopper_id.

    decay: float in (0, 1). Higher = evidence "remembered" longer.
           0.9 means roughly ~90% of previous risk carries into the next update.
    weights: base importance of each evidence type before any arbiter multiplier.
    """

    DEFAULT_WEIGHTS = {
        "hand_object_interaction": 0.3,
        "item_disappeared": 0.4,
        "restricted_zone_entry": 0.5,
        "weapon_detected": 1.0,
        "aggressive_pose": 0.7,
        "till_collusion": 0.6,
    }

    def __init__(self, decay: float = 0.9, weights: Optional[Dict[str, float]] = None):
        if not (0.0 < decay < 1.0):
            raise ValueError("decay must be strictly between 0 and 1")
        self.decay = decay
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)
        self._scores: Dict[str, float] = {}
        self._history: Dict[str, List[RiskEvent]] = {}

    def get_score(self, shopper_id: str) -> float:
        return self._scores.get(shopper_id, 0.0)

    def get_history(self, shopper_id: str) -> List[RiskEvent]:
        return self._history.get(shopper_id, [])

    def update(
        self,
        shopper_id: str,
        event_type: str,
        confidence: float = 1.0,
        multiplier: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> float:
        """
        Apply one piece of evidence for a shopper and return the new risk score.

        confidence: 0-1, how sure Layer 3 is this event actually happened
        multiplier: applied by the Symbolic Context Arbiter BEFORE calling this
                    (e.g. 0.0 to suppress entirely, 3.0 to escalate a weapon event)
        """
        if not (0.0 <= confidence <= 1.0):
            raise ValueError("confidence must be between 0 and 1")

        prev = self._scores.get(shopper_id, 0.0)
        base_weight = self.weights.get(event_type, 0.1)
        contribution = base_weight * confidence * multiplier
        new_score = self.decay * prev + contribution
        self._scores[shopper_id] = new_score

        event = RiskEvent(
            shopper_id=shopper_id,
            timestamp=timestamp if timestamp is not None else time.time(),
            event_type=event_type,
            raw_confidence=confidence,
            weight_applied=base_weight,
            multiplier_applied=multiplier,
            evidence_contribution=contribution,
            risk_after=new_score,
        )
        self._history.setdefault(shopper_id, []).append(event)
        return new_score

    def reset(self, shopper_id: str) -> None:
        """Call this once a shopper's visit ends (left the store) to free memory."""
        self._scores.pop(shopper_id, None)
        self._history.pop(shopper_id, None)