"""
Layer 5 - Evidence Packager

Builds the small, curated evidence window sent to the VLM out of a
Layer 4 escalation, instead of handing the model a raw video clip.
Follows SmartEyes' two-keyframe design (Section 3.5): a "contact" frame
(first interaction with the object) and a "resolution" frame (the point
the interaction chain closed, or the point the escalation threshold was
crossed), plus a short packet of symbolic metadata the model can cite
against rather than re-derive from pixels alone.

NOTE: Layers 1-3 don't produce real stored frames yet, so
`contact_frame_id` / `resolution_frame_id` are synthetic placeholders
below. Once Layer 3 exposes real frame references per interaction chain,
wire those in here - nothing downstream of this module needs to change.
"""

from typing import Optional

from layer4_reasoning.reasoning_engine import Layer4Decision
from .verdict_schema import EvidenceFrame, EvidencePacket


def build_evidence_packet(
    decision: Layer4Decision,
    contact_frame_id: Optional[str] = None,
    resolution_frame_id: Optional[str] = None,
) -> EvidencePacket:
    contact_frame_id = contact_frame_id or f"{decision.shopper_id}_contact"
    resolution_frame_id = resolution_frame_id or f"{decision.shopper_id}_resolution"

    frames = [
        EvidenceFrame(frame_id=contact_frame_id, timestamp=(decision.timestamp or 0.0) - 1.0, role="contact"),
        EvidenceFrame(frame_id=resolution_frame_id, timestamp=decision.timestamp or 0.0, role="resolution"),
    ]

    return EvidencePacket(
        shopper_id=decision.shopper_id,
        category=decision.category,
        fired_rule=decision.fired_rule or "risk_threshold_only",
        risk_score=decision.risk_score,
        frames=frames,
    )


def build_recheck_packet(packet: EvidencePacket) -> EvidencePacket:
    """Same evidence window, but frames marked as a 'varied crop' - the
    Fail-Safe re-check step (Table 2) uses a different crop/context on the
    same incident before a high-risk alert is finalised."""
    varied_frames = [
        EvidenceFrame(frame_id=f.frame_id, timestamp=f.timestamp, role=f.role, crop_variant="varied")
        for f in packet.frames
    ]
    return EvidencePacket(
        shopper_id=packet.shopper_id,
        category=packet.category,
        fired_rule=packet.fired_rule,
        risk_score=packet.risk_score,
        frames=varied_frames,
    )