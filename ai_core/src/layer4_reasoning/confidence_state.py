"""
Layer 4 - Confidence State Machine

Implements the six-state discrete confidence model described in the
WATCHMEN synopsis (Section 8.4, Table 3):

    Certain - Benign     | Certain - Risk        | Probable
    Uncertain            | Contradictory         | Insufficient Evidence

This replaces a single binary "escalate / don't escalate" threshold with an
explicit vocabulary that the Explainability layer (Layer 6) can render, and
that the VLM Verdict layer (Layer 5) writes back into once it has run.

Design notes / assumptions - flag these in your viva, they are judgment
calls made in the absence of a finalised Layer 3 schema:

  - CONTRADICTORY detection is currently a heuristic over a short rolling
    event history per shopper (looking for a flip in `reappeared_in_basket`
    across two `item_disappeared` events). Once Manisha's real Layer 3
    output exposes an explicit interaction-chain id and state
    (open/closed/concealed/unresolved), replace this with a proper
    chain-conflict check instead of inferring it from raw event history.
  - INSUFFICIENT_EVIDENCE fires when a shopper has too little history to
    say anything meaningful yet. This stands in for "chain never
    completed" until Layer 3 exposes occlusion/identity-loss flags
    directly.
  - CERTAIN_RISK is only ever reached after a VLM verdict agrees with the
    symbolic layer (Table 3: "Symbolic rules and VLM re-check agree on
    high risk"). Before Layer 5 exists, crossing the escalation threshold
    can only produce PROBABLE (pending VLM confirmation) - that is
    intentional, not a bug: Layer 4 alone should never be able to reach
    Certain - Risk by itself.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ConfidenceState(Enum):
    CERTAIN_BENIGN = "certain_benign"
    CERTAIN_RISK = "certain_risk"
    PROBABLE = "probable"
    UNCERTAIN = "uncertain"
    CONTRADICTORY = "contradictory"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass
class CategoryBar:
    """Per-threat-category confidence bar (Table 2: 'category-specific
    confidence bars').

    escalate_at:     risk score above which Layer 4 should ask Layer 5 to look.
    benign_ceiling:  risk score below which the shopper can be marked
                     Certain - Benign without ever bothering Layer 5.
    """
    escalate_at: float
    benign_ceiling: float


# Lower escalate_at = ask the VLM sooner. Concealment is visually subtle
# (CRBA's per-class error analysis, Section 3.7) so Layer 4 shouldn't trust
# itself to auto-resolve it alone; weapon events get a low bar too, since a
# missed weapon is far costlier than one extra VLM call, even though the
# arbiter's multiplier usually pushes weapon risk over threshold in a
# single event anyway.
DEFAULT_CATEGORY_BARS: Dict[str, CategoryBar] = {
    "concealment": CategoryBar(escalate_at=0.75, benign_ceiling=0.2),
    "weapon": CategoryBar(escalate_at=0.5, benign_ceiling=0.1),
    "violence": CategoryBar(escalate_at=0.6, benign_ceiling=0.15),
    "collusion": CategoryBar(escalate_at=0.7, benign_ceiling=0.2),
    "default": CategoryBar(escalate_at=1.0, benign_ceiling=0.2),
}

# Maps RiskAccumulator.DEFAULT_WEIGHTS keys to a threat category. Keep this
# in sync with risk_accumulator.py's event types.
EVENT_CATEGORY: Dict[str, str] = {
    "hand_object_interaction": "concealment",
    "item_disappeared": "concealment",
    "restricted_zone_entry": "concealment",
    "weapon_detected": "weapon",
    "aggressive_pose": "violence",
    "till_collusion": "collusion",
}


def category_for(event_type: str) -> str:
    return EVENT_CATEGORY.get(event_type, "default")


@dataclass
class _ShopperMemory:
    history: List[Dict[str, Any]] = field(default_factory=list)
    state: ConfidenceState = ConfidenceState.INSUFFICIENT_EVIDENCE
    awaiting_vlm: bool = False
    first_vlm_verdict: Optional[str] = None


class ConfidenceStateMachine:
    HISTORY_WINDOW = 8  # events kept per shopper for contradiction checks
    MIN_EVENTS_FOR_OPINION = 2  # fewer than this -> Insufficient Evidence

    def __init__(self, category_bars: Optional[Dict[str, CategoryBar]] = None):
        self.category_bars = category_bars or dict(DEFAULT_CATEGORY_BARS)
        self._memory: Dict[str, _ShopperMemory] = {}

    def _mem(self, shopper_id: str) -> _ShopperMemory:
        return self._memory.setdefault(shopper_id, _ShopperMemory())

    def _bar(self, category: str) -> CategoryBar:
        return self.category_bars.get(category, self.category_bars["default"])

    def _check_contradiction(self, mem: _ShopperMemory) -> bool:
        """Two 'item_disappeared' events for the same shopper whose
        'reappeared_in_basket' flag flips is treated as conflicting
        evidence within the same visit (Table 3: Contradictory)."""
        flags = [
            e["metadata"].get("reappeared_in_basket")
            for e in mem.history
            if e.get("event_type") == "item_disappeared"
            and "reappeared_in_basket" in e.get("metadata", {})
        ]
        return len(flags) >= 2 and len(set(flags[-2:])) > 1

    def update(
        self,
        shopper_id: str,
        event: Dict[str, Any],
        risk_score: float,
        fired_rule: Optional[str],
        multiplier: float,
        will_escalate_now: bool,
    ) -> ConfidenceState:
        """Call once per event, after the arbiter multiplier and the new
        risk score are known, and after EscalationPolicy has decided
        whether this event is the one that fires the VLM call.

        will_escalate_now: True exactly on the event where EscalationPolicy
                            returns should_escalate=True.
        """
        mem = self._mem(shopper_id)
        mem.history.append(event)
        mem.history = mem.history[-self.HISTORY_WINDOW:]

        bar = self._bar(category_for(event.get("event_type", "")))

        if self._check_contradiction(mem):
            mem.state = ConfidenceState.CONTRADICTORY
            return mem.state

        if will_escalate_now:
            mem.awaiting_vlm = True
            mem.first_vlm_verdict = None
            mem.state = ConfidenceState.PROBABLE
            return mem.state

        if mem.awaiting_vlm:
            # Already escalated this incident, VLM hasn't answered yet -
            # keep reporting Probable rather than flip-flopping state.
            mem.state = ConfidenceState.PROBABLE
            return mem.state

        if fired_rule is not None and multiplier == 0.0:
            # The arbiter explicitly suppressed THIS event (staff badge,
            # returned item, replica prop, etc). That is a definitive call,
            # not "wait and see" - trust it immediately rather than letting
            # leftover decayed risk from an earlier, harmless event or a
            # thin history hold the shopper at Probable/Insufficient.
            mem.state = ConfidenceState.CERTAIN_BENIGN
            return mem.state

        if len(mem.history) < self.MIN_EVENTS_FOR_OPINION:
            mem.state = ConfidenceState.INSUFFICIENT_EVIDENCE
            return mem.state

        if risk_score >= bar.escalate_at:
            mem.state = ConfidenceState.PROBABLE
        elif risk_score <= bar.benign_ceiling:
            mem.state = ConfidenceState.CERTAIN_BENIGN
        else:
            mem.state = ConfidenceState.PROBABLE

        return mem.state

    def record_vlm_verdict(
        self, shopper_id: str, verdict: str, is_recheck: bool = False
    ) -> ConfidenceState:
        """Called by Layer 5 once it has a verdict for a shopper currently
        awaiting one. verdict is one of 'CONFIRMED', 'UNCERTAIN', 'NORMAL'.

        Per TRACE's finding (Section 3.7): disagreement is a signal in its
        own right, never averaged away. So:
          - First pass NORMAL     -> Certain - Benign immediately.
          - First pass UNCERTAIN  -> Uncertain immediately (the VLM being
            unsure of itself IS the disagreement signal - Layer 5 should
            not spend a second call trying to talk it into a firmer answer).
          - First pass CONFIRMED  -> stays Probable, pending the re-check.
          - Re-check CONFIRMED after first pass CONFIRMED -> Certain - Risk.
          - Re-check disagrees with the first pass          -> Uncertain.
        """
        mem = self._mem(shopper_id)

        if not is_recheck:
            mem.first_vlm_verdict = verdict
            if verdict == "NORMAL":
                mem.state = ConfidenceState.CERTAIN_BENIGN
                mem.awaiting_vlm = False
            elif verdict == "UNCERTAIN":
                mem.state = ConfidenceState.UNCERTAIN
                mem.awaiting_vlm = False
            else:  # CONFIRMED - hold pending the re-check
                mem.state = ConfidenceState.PROBABLE
            return mem.state

        # This is the re-check.
        if mem.first_vlm_verdict == verdict == "CONFIRMED":
            mem.state = ConfidenceState.CERTAIN_RISK
        elif mem.first_vlm_verdict != verdict:
            mem.state = ConfidenceState.UNCERTAIN
        elif verdict == "NORMAL":
            mem.state = ConfidenceState.CERTAIN_BENIGN
        else:
            mem.state = ConfidenceState.UNCERTAIN

        mem.awaiting_vlm = False
        return mem.state

    def get_state(self, shopper_id: str) -> ConfidenceState:
        return self._mem(shopper_id).state

    def reset(self, shopper_id: str) -> None:
        self._memory.pop(shopper_id, None)