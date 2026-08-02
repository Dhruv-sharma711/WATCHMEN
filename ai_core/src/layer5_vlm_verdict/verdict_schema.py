"""
Layer 5 - Verdict Schema

Structured output contract for the VLM Verdict layer. Every VLM call -
first pass or re-check - must produce exactly this shape, per Borodin et
al.'s finding that a short, constrained-output prompt beats elaborate
chain-of-thought reasoning (Section 3.5), and modelled on Mohamed et al.'s
structured suspect-report format (Section 3.5).
"""

from dataclasses import dataclass, field
from typing import List

# Kept as a plain string rather than a Literal/Enum union with the type
# checker, since the mock/real client boundary should be free to validate
# and coerce whatever a raw model response contains.
VerdictLabel = str  # one of "CONFIRMED", "UNCERTAIN", "NORMAL"


@dataclass
class EvidenceFrame:
    """One of the (usually two) keyframes sent to the VLM."""
    frame_id: str
    timestamp: float
    role: str  # "contact" or "resolution", per SmartEyes' two-keyframe design
    crop_variant: str = "primary"  # "primary" or "varied" (varied = used for the re-check)


@dataclass
class EvidencePacket:
    """Everything Layer 5 sends to the VLM for one escalation."""
    shopper_id: str
    category: str
    fired_rule: str
    risk_score: float
    frames: List[EvidenceFrame] = field(default_factory=list)


@dataclass
class VLMVerdict:
    verdict: VerdictLabel
    rationale: str
    cited_frame_ids: List[str] = field(default_factory=list)
    is_recheck: bool = False