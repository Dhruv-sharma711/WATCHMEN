"""
Layer 4 end-to-end smoke test.

Run: python tests/test_pipeline.py

Wires: mock_evidence_generator -> symbolic_arbiter -> risk_accumulator -> escalation_policy
Prints a risk trajectory per scenario and saves a PNG plot so you can visually
sanity-check that:
  - normal_shopper stays low
  - staff_restocking stays near zero (suppressed)
  - slow_theft climbs and eventually escalates
  - weapon_at_register escalates immediately
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from layer4_reasoning.risk_accumulator import RiskAccumulator
from layer4_reasoning.symbolic_arbiter import SymbolicContextArbiter
from layer4_reasoning.escalation_policy import EscalationPolicy
from layer4_reasoning.mock_evidence_generator import ALL_SCENARIOS

RULES_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "layer4_reasoning", "rules")


def run_scenario(name, events):
    accumulator = RiskAccumulator(decay=0.85)
    arbiter = SymbolicContextArbiter(
        suppression_rules_path=os.path.join(RULES_DIR, "suppression_rules.yaml"),
        escalation_rules_path=os.path.join(RULES_DIR, "escalation_rules.yaml"),
    )
    policy = EscalationPolicy(threshold=1.0, cooldown_seconds=60.0)

    trajectory = []
    escalated_at = None

    for event in events:
        decision = arbiter.evaluate(event)
        score = accumulator.update(
            shopper_id=event["shopper_id"],
            event_type=event["event_type"],
            confidence=event["confidence"],
            multiplier=decision.final_multiplier,
            timestamp=event["timestamp"],
        )
        trajectory.append(score)

        esc = policy.check(event["shopper_id"], score, now=event["timestamp"])
        if esc.should_escalate and escalated_at is None:
            escalated_at = len(trajectory) - 1

        rule_note = f" [rule: {decision.fired_rule}]" if decision.fired_rule else ""
        print(f"  {name}: {event['event_type']:<25} risk={score:.3f}{rule_note}"
              f"{'  <-- ESCALATED' if esc.should_escalate else ''}")

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

    plt.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label="threshold (tau=1.0)")
    plt.xlabel("Event index")
    plt.ylabel("Risk score R(id, t)")
    plt.title("WATCHMEN Layer 4 - Risk trajectories by scenario")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "risk_trajectories.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()