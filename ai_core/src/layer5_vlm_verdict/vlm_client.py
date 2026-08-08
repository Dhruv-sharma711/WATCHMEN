"""
Layer 5 - VLM Client

Abstract interface for the vision-language model call, a mock
implementation so the rest of Layer 5 - and Layer 4's callback contract -
can be built, tested, and demoed before a real model backend is wired in,
and a real implementation (AnthropicVLMClient) that sends the evidence
window to a Claude vision model.

Prompt design follows Borodin et al.'s finding (Section 3.5): across six
compact VLMs, the best score came from the shortest, plainest,
constrained-output prompt - few-shot and chain-of-thought prompts made
accuracy worse and latency 5-7x higher. Do NOT add reasoning steps,
examples, or persona text to PROMPT_TEMPLATE below - that is a deliberate
design choice made in Section 8.3/9.11, not an oversight.
"""

import base64
import random
import re
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from .verdict_schema import EvidencePacket, VLMVerdict, VALID_VERDICTS

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


def parse_vlm_response(raw_text: str) -> Tuple[str, str]:
    """Turn a raw model response into (verdict_label, rationale).

    The prompt asks for exactly one label word followed by a short
    rationale, but a real model won't always comply (extra whitespace,
    markdown bolding, lower case, a missing label entirely). Any response
    that doesn't clearly resolve to one of VALID_VERDICTS is coerced to
    UNCERTAIN rather than guessed toward CONFIRMED or NORMAL - per the
    Burden-of-Proof principle (Section 8.4/9.10), a parse failure must
    fail toward human review, never toward silence.
    """
    text = (raw_text or "").strip()
    match = re.match(r"^\**\s*(CONFIRMED|UNCERTAIN|NORMAL)\b[\s,:\-]*", text, re.IGNORECASE)
    if match:
        verdict = match.group(1).upper()
        rationale = text[match.end():].strip() or "(model returned no rationale)"
        return verdict, rationale

    return "UNCERTAIN", f"unparseable model response, defaulting to UNCERTAIN: {text[:120]!r}"


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


class FrameSource(ABC):
    """Resolves a Layer 5 frame_id to raw image bytes.

    Layers 1-3 don't expose real stored frames yet (see the NOTE at the
    top of evidence_packager.py) - contact_frame_id/resolution_frame_id
    are synthetic placeholders for now. This interface exists so
    AnthropicVLMClient below can be written and tested today; once a real
    frame store lands (e.g. Layer 1/7's short-retention evidence store),
    only a new FrameSource implementation needs to be written - nothing
    in AnthropicVLMClient changes.
    """

    @abstractmethod
    def get_image(self, frame_id: str) -> Tuple[bytes, str]:
        """Return (raw_image_bytes, media_type), e.g. (b"...", "image/jpeg")."""
        raise NotImplementedError


class AnthropicVLMClient(VLMClient):
    """Real backend: sends the two-frame evidence window plus the fixed,
    constrained-output prompt (PROMPT_TEMPLATE) to a Claude vision model.

    Requires the `anthropic` package (add it to requirements.txt) and an
    ANTHROPIC_API_KEY in the environment. Swap `model` for whichever
    backend the team settles on - orchestration in layer5_engine.py
    doesn't care which VLMClient implementation it holds.
    """

    def __init__(
        self,
        frame_source: FrameSource,
        model: str = "claude-sonnet-5",
        max_tokens: int = 200,
    ):
        import anthropic  # local import: keeps the SDK optional for tests/mocks

        self.frame_source = frame_source
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()

    def query(self, packet: EvidencePacket, is_recheck: bool = False) -> VLMVerdict:
        prompt = build_prompt(packet)

        content = []
        for frame in packet.frames:
            image_bytes, media_type = self.frame_source.get_image(frame.frame_id)
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64.b64encode(image_bytes).decode("utf-8"),
                    },
                }
            )
        content.append({"type": "text", "text": prompt})

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        raw_text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )

        verdict_label, rationale = parse_vlm_response(raw_text)
        if verdict_label not in VALID_VERDICTS:
            verdict_label, rationale = "UNCERTAIN", f"unexpected label, defaulting to UNCERTAIN: {rationale}"

        return VLMVerdict(
            verdict=verdict_label,
            rationale=rationale,
            cited_frame_ids=[f.frame_id for f in packet.frames],
            is_recheck=is_recheck,
        )