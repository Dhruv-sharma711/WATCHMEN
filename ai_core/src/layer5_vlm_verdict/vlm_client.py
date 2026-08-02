"""
Layer 5 - VLM Client

Abstract interface for the vision-language model call, plus a mock
implementation so the rest of Layer 5 - and Layer 4's callback contract -
can be built, tested, and demoed before a real model backend is wired in.

Prompt design follows Borodin et al.'s finding (Section 3.5): across six
compact VLMs, the best score came from the shortest, plainest,
constrained-output prompt - few-shot and chain-of-thought prompts made
accuracy worse and latency 5-7x higher. Do NOT add reasoning steps,
examples, or persona text to PROMPT_TEMPLATE below - that is a deliberate
design choice made in Section 8.3/9.11, not an oversight.
"""

import random
from abc import ABC, abstractmethod
from typing import Dict, Optional

from .verdict_schema import EvidencePacket, VLMVerdict

PROMPT_TEMPLATE = """You are comparing two frames from a retail security camera.
Category: {category}
Matched rule: {fired_rule}
Risk score: {risk_score:.2f}

Frame 1 ({contact_ts}): the moment of contact with the item.
Frame 2 ({resolution_ts}): what happened after.

Answer with exactly one word - CONFIRMED, UNCERTAIN, or NORMAL - followed
by a rationale of 20 words or fewer citing what differs between the two
frames."""


def build_prompt(packet: EvidencePacket) -> str:
    contact = next((f for f in packet.frames if f.role == "contact"), packet.frames[0])
    resolution = next((f for f in packet.frames if f.role == "resolution"), packet.frames[-1])
    return PROMPT_TEMPLATE.format(
        category=packet.category,
        fired_rule=packet.fired_rule,
        risk_score=packet.risk_score,
        contact_ts=contact.timestamp,
        resolution_ts=resolution.timestamp,
    )


class VLMClient(ABC):
    """Implement this against whichever backend you settle on (a hosted
    VLM API, a locally served model, etc). Layer 5's orchestration logic
    (rate limiting, re-check flow, Layer 4 callback) never needs to
    change - only this class does."""

    @abstractmethod
    def query(self, packet: EvidencePacket, is_recheck: bool = False) -> VLMVerdict:
        raise NotImplementedError


class MockVLMClient(VLMClient):
    """Deterministic-ish stand-in for development and pipeline testing.
    Derives a plausible verdict from the symbolic metadata already in the
    packet, so Layer 5's orchestration can be built and tested end to end
    without waiting on a real model integration.
    """

    def __init__(self, agreement_rate: float = 1.0, seed: Optional[int] = None):
        """agreement_rate: probability the re-check agrees with the first
        pass - lower this in a test to exercise the Uncertain-via-disagreement path."""
        self.agreement_rate = agreement_rate
        self._rng = random.Random(seed)
        self._first_pass_cache: Dict[str, str] = {}

    def _base_verdict(self, packet: EvidencePacket) -> str:
        if packet.category == "weapon":
            return "CONFIRMED"
        if packet.risk_score >= 0.9:
            return "CONFIRMED"
        if packet.risk_score >= 0.5:
            return "UNCERTAIN"
        return "NORMAL"

    def query(self, packet: EvidencePacket, is_recheck: bool = False) -> VLMVerdict:
        if is_recheck:
            first = self._first_pass_cache.get(packet.shopper_id, self._base_verdict(packet))
            verdict = first if self._rng.random() < self.agreement_rate else "UNCERTAIN"
        else:
            verdict = self._base_verdict(packet)
            self._first_pass_cache[packet.shopper_id] = verdict

        return VLMVerdict(
            verdict=verdict,
            rationale=f"mock verdict from risk={packet.risk_score:.2f}, rule={packet.fired_rule}",
            cited_frame_ids=[f.frame_id for f in packet.frames],
            is_recheck=is_recheck,
        )