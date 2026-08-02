"""
Layer 4 - Reasoning Engine (orchestrator)

Wires the four Layer 4 components together into a single entry point, so
Layer 3 (upstream) and Layer 5 (downstream) don't need to know how the
Symbolic Arbiter, Risk Accumulator, Escalation Policy and Confidence State
Machine talk to each other internally.

    event (from Layer 3) -> process() -> Layer4Decision -> maybe Layer 5
    Layer 5 verdict -> record_vlm_verdict() -> updated confidence state

This is the module the rest of the pipeline (and your tests) should import
from - not the four sub-modules directly.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .symbolic_arbiter import SymbolicContextArbiter
from .risk_accumulator import RiskAccumulator
from .escalation_policy import EscalationPolicy
from .confidence_state import ConfidenceStateMachine, ConfidenceState, category_for


@dataclass
class Layer4Decision:
    shopper_id: str
    timestamp: float
    event_type: str
    category: str
    risk_score: float
    multiplier_applied: float
    fired_rule: Optional[str]
    confidence_state: ConfidenceState
    should_call_vlm: bool
    reason: str


class Layer4ReasoningEngine:
    def __init__(
        self,
        suppression_rules_path: str,
        escalation_rules_path: str,
        decay: float = 0.9,
        cooldown_seconds: float = 60.0,
    ):
        self.arbiter = SymbolicContextArbiter(suppression_rules_path, escalation_rules_path)
        self.accumulator = RiskAccumulator(decay=decay)
        self.escalation = EscalationPolicy(cooldown_seconds=cooldown_seconds)
        self.confidence = ConfidenceStateMachine()

    def process(self, event: Dict[str, Any]) -> Layer4Decision:
        shopper_id = event["shopper_id"]
        event_type = event["event_type"]
        timestamp = event.get("timestamp")
        confidence_in_event = event.get("confidence", 1.0)

        # Step 1: suppress or escalate the raw evidence via hand-editable rules.
        arbiter_decision = self.arbiter.evaluate(event)

        # Step 2: fold it into the shopper's decaying running risk score.
        risk_score = self.accumulator.update(
            shopper_id=shopper_id,
            event_type=event_type,
            confidence=confidence_in_event,
            multiplier=arbiter_decision.final_multiplier,
            timestamp=timestamp,
        )

        # Step 3: decide if THIS event is the one that should fire Layer 5.
        escalation_decision = self.escalation.check(
            shopper_id=shopper_id,
            current_risk=risk_score,
            event_type=event_type,
            now=timestamp,
        )

        # Step 4: map all of the above onto the six-state confidence model.
        state = self.confidence.update(
            shopper_id=shopper_id,
            event=event,
            risk_score=risk_score,
            fired_rule=arbiter_decision.fired_rule,
            multiplier=arbiter_decision.final_multiplier,
            will_escalate_now=escalation_decision.should_escalate,
        )

        return Layer4Decision(
            shopper_id=shopper_id,
            timestamp=timestamp,
            event_type=event_type,
            category=category_for(event_type),
            risk_score=risk_score,
            multiplier_applied=arbiter_decision.final_multiplier,
            fired_rule=arbiter_decision.fired_rule,
            confidence_state=state,
            should_call_vlm=escalation_decision.should_escalate,
            reason=escalation_decision.reason,
        )

    def record_vlm_verdict(
        self, shopper_id: str, verdict: str, is_recheck: bool = False
    ) -> ConfidenceState:
        """Layer 5 calls this once it has a CONFIRMED / UNCERTAIN / NORMAL
        verdict for a shopper that process() flagged with should_call_vlm=True."""
        return self.confidence.record_vlm_verdict(shopper_id, verdict, is_recheck=is_recheck)

    def get_state(self, shopper_id: str) -> ConfidenceState:
        return self.confidence.get_state(shopper_id)

    def reset(self, shopper_id: str) -> None:
        """Call when a shopper's visit ends - frees memory across all sub-modules."""
        self.accumulator.reset(shopper_id)
        self.confidence.reset(shopper_id)