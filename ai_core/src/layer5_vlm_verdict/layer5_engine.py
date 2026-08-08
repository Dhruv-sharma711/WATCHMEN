"""
Layer 5 - VLM Verdict Engine (orchestrator)

Consumes a Layer4Decision flagged should_call_vlm=True, packages a small
evidence window, makes a rate-limited VLM call, and - only if that call
comes back CONFIRMED - runs a single re-check under a varied crop before
writing the result back into Layer 4's confidence state (Table 2's
Fail-Safe pattern: never finalise a high-risk alert off one pass alone).
An UNCERTAIN or NORMAL first pass is treated as final and does NOT spend a
second call - see the note in confidence_state.record_vlm_verdict.

The re-check spends from the SAME rate budget as the first pass - Table 2
describes Layer 5 as "rate-limited" as a whole, not just its first pass.
If the budget is saturated when a CONFIRMED first pass tries to schedule
its re-check, the re-check is skipped for now and `recheck_rate_limited`
is set True on the result: the incident is left exactly where
record_vlm_verdict(is_recheck=False) put it (NOT finalised as
Certain-Risk) rather than being silently treated as confirmed off a
single pass.

    Layer4Decision (should_call_vlm=True)
        -> build_evidence_packet()
        -> VLMClient.query()                       (first pass, rate-limited)
        -> [only if CONFIRMED] VLMClient.query()    (re-check, varied crop, rate-limited)
        -> Layer4ReasoningEngine.record_vlm_verdict()  (once or twice)
"""

from dataclasses import dataclass
from typing import Optional

from layer4_reasoning.reasoning_engine import Layer4ReasoningEngine, Layer4Decision
from layer4_reasoning.confidence_state import ConfidenceState

from .evidence_packager import build_evidence_packet, build_recheck_packet
from .vlm_client import VLMClient
from .rate_limiter import RateLimiter
from .verdict_schema import VLMVerdict


@dataclass
class Layer5Result:
    shopper_id: str
    first_pass: Optional[VLMVerdict]
    recheck: Optional[VLMVerdict]
    final_confidence_state: ConfidenceState
    rate_limited: bool = False
    recheck_rate_limited: bool = False


class Layer5VLMVerdict:
    def __init__(
        self,
        vlm_client: VLMClient,
        reasoning_engine: Layer4ReasoningEngine,
        max_calls: int = 30,
        per_seconds: float = 60.0,
    ):
        self.vlm_client = vlm_client
        self.reasoning_engine = reasoning_engine
        self.rate_limiter = RateLimiter(max_calls=max_calls, per_seconds=per_seconds)

    def handle(self, decision: Layer4Decision) -> Optional[Layer5Result]:
        """Call this whenever Layer 4 returns should_call_vlm=True.

        Returns None if decision.should_call_vlm was False (nothing to do).
        Returns a rate_limited=True result if the budget is currently
        saturated before even the first pass can run - the incident is
        left at Probable and will be picked up again the next time this
        shopper's risk re-crosses threshold, or by an operator manually
        reviewing the queue in the meantime.
        """
        if not decision.should_call_vlm:
            return None

        if not self.rate_limiter.allow(now=decision.timestamp):
            return Layer5Result(
                shopper_id=decision.shopper_id,
                first_pass=None,
                recheck=None,
                final_confidence_state=self.reasoning_engine.get_state(decision.shopper_id),
                rate_limited=True,
            )

        packet = build_evidence_packet(decision)
        first_pass = self.vlm_client.query(packet, is_recheck=False)
        state = self.reasoning_engine.record_vlm_verdict(
            decision.shopper_id, first_pass.verdict, is_recheck=False
        )

        recheck = None
        recheck_rate_limited = False
        if first_pass.verdict == "CONFIRMED":
            if self.rate_limiter.allow(now=decision.timestamp):
                recheck_packet = build_recheck_packet(packet)
                recheck = self.vlm_client.query(recheck_packet, is_recheck=True)
                state = self.reasoning_engine.record_vlm_verdict(
                    decision.shopper_id, recheck.verdict, is_recheck=True
                )
            else:
                # Fail-Safe re-check itself got rate limited: per Table 2, a
                # high-risk alert is never finalised off one pass alone, so
                # the CONFIRMED first pass is deliberately NOT treated as
                # final here - state stays whatever record_vlm_verdict
                # already set above (pending), and the caller learns via
                # recheck_rate_limited that this incident needs revisiting.
                recheck_rate_limited = True
        # UNCERTAIN or NORMAL on the first pass is treated as final - see
        # the module docstring and confidence_state.record_vlm_verdict.

        return Layer5Result(
            shopper_id=decision.shopper_id,
            first_pass=first_pass,
            recheck=recheck,
            final_confidence_state=state,
            recheck_rate_limited=recheck_rate_limited,
        )