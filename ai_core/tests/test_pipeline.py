"""
Layer 4 end-to-end smoke test.

Run: python tests/test_pipeline.py

Wires the four Layer 4 sub-modules through Layer4ReasoningEngine - the
single orchestrating entry point reasoning_engine.py's own docstring says
the rest of the pipeline (and tests) should import, instead of manually
re-wiring symbolic_arbiter -> risk_accumulator -> escalation_policy by hand.

That manual re-wiring is what caused this file to break: EscalationPolicy
was refactored to take category-aware thresholds (a `thresholds` dict
keyed by threat category, and `check(event_type=...)` so it can look the
right threshold up), but this test was still calling the old single-value
API (`EscalationPolicy(threshold=1.0, ...)` and `policy.check(id, score,
now=...)` with no event_type). reasoning_engine.py had already been updated
to the new signature - only this file had drifted. Routing through the
orchestrator instead of the sub-modules means this file can no longer
drift out of sync the same way again.

Going through the orchestrator also means the six-state Confidence State
Machine (confidence_state.py) is exercised for the first time in this
test, not just the raw risk number - which matters, because
"contradictory_evidence" only tests anything meaningful once you can see
it produce a CONTRADICTORY state below (the old version only ever printed
its risk score, which looks identical to any other mid-range event).

Prints a risk trajectory + confidence state per event, per scenario, and
saves a PNG plot so you can visually sanity-check that:
  - normal_shopper stays low, resolves Certain-Benign
  - staff_restocking stays at zero (suppressed), resolves Certain-Benign
  - slow_theft climbs and eventually escalates to a Layer 5 call
  - weapon_at_register escalates on the very first event
  - item_return / replica_toy_at_register resolve Certain-Benign via
    suppression rules rather than lingering as suspicious
  - contradictory_evidence is flagged Contradictory, not silently averaged
    into an ordinary risk number
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from layer4_reasoning.reasoning_engine import Layer4ReasoningEngine
from layer4_reasoning.mock_evidence_generator import ALL_SCENARIOS

RULES_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "layer4_reasoning", "rules")

# Reference line only - EscalationPolicy is category-aware, so the real
# threshold an event is checked against depends on its category (weapon:
# 0.5, violence: 0.6, collusion: 0.7, concealment: 0.75, default: 1.0 -
# see confidence_state.DEFAULT_CATEGORY_BARS). 1.0 is just the "default"
# bucket's threshold, plotted so trajectories have a fixed visual anchor.
DEFAULT_THRESHOLD_REFERENCE = 1.0


def run_scenario(name, events):
    engine = Layer4ReasoningEngine(
        suppression_rules_path=os.path.join(RULES_DIR, "suppression_rules.yaml"),
        escalation_rules_path=os.path.join(RULES_DIR, "escalation_rules.yaml"),
        decay=0.85,
        cooldown_seconds=60.0,
    )

    trajectory = []
    escalated_at = None

    for event in events:
        decision = engine.process(event)
        trajectory.append(decision.risk_score)

        if decision.should_call_vlm and escalated_at is None:
            escalated_at = len(trajectory) - 1

        rule_note = f" [rule: {decision.fired_rule}]" if decision.fired_rule else ""
        state_note = f" ({decision.confidence_state.value})"
        escalate_note = "  <-- ESCALATED (Layer 5 call)" if decision.should_call_vlm else ""
        print(f"  {name}: {event['event_type']:<25} risk={decision.risk_score:.3f}"
              f"{rule_note}{state_note}{escalate_note}")

    return trajectory, escalated_at


def main():
    plt.figure(figsize=(9, 5))
    for name, generator_fn in ALL_SCENARIOS.items():
        print(f"\n=== {name} ===")
        events = generator_fn()
        trajectory, escalated_at = run_scenario(name, events)
        x = list(range(len(trajectory)))
        plt.plot(x, trajectory, marker="o", label=name)
        if escalated_at is not None:
            plt.scatter([escalated_at], [trajectory[escalated_at]], s=120,
                        facecolors="none", edgecolors="red", linewidths=2, zorder=5)

    plt.axhline(y=DEFAULT_THRESHOLD_REFERENCE, color="gray", linestyle="--", linewidth=1,
                label="default-category threshold (categories vary - see confidence_state.py)")
    plt.xlabel("Event index")
    plt.ylabel("Risk score R(id, t)")
    plt.title("WATCHMEN Layer 4 - Risk trajectories by scenario")
    plt.legend(fontsize=8)
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "risk_trajectories.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()