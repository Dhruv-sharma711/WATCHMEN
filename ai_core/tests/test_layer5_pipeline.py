"""
Layer 5 end-to-end smoke test.

Run: python tests/test_layer5_pipeline.py

Feeds every should_call_vlm=True decision produced by the Layer 4 mock
scenarios (see tests/test_pipeline.py) through Layer5VLMVerdict, using
MockVLMClient as a stand-in backend. Exercises:
  - evidence packaging (contact/resolution frames)
  - the CONFIRMED -> re-check -> Layer 4 callback loop
  - the rate limiter, including the Fail-Safe re-check call also
    spending from the budget (previously only the first pass did -
    see the docstring in layer5_engine.py)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from layer4_reasoning.reasoning_engine import Layer4ReasoningEngine
from layer4_reasoning.mock_evidence_generator import ALL_SCENARIOS

from layer5_vlm_verdict.layer5_engine import Layer5VLMVerdict
from layer5_vlm_verdict.vlm_client import MockVLMClient

RULES_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "layer4_reasoning", "rules")


def _new_engine():
    return Layer4ReasoningEngine(
        suppression_rules_path=os.path.join(RULES_DIR, "suppression_rules.yaml"),
        escalation_rules_path=os.path.join(RULES_DIR, "escalation_rules.yaml"),
        decay=0.85,
        cooldown_seconds=60.0,
    )


def run_scenario(name, events, layer5):
    engine = layer5.reasoning_engine
    results = []
    for event in events:
        decision = engine.process(event)
        if not decision.should_call_vlm:
            continue

        result = layer5.handle(decision)
        results.append(result)
        if result is None:
            continue

        if result.rate_limited:
            print(f"  {name}: RATE LIMITED at t={decision.timestamp}")
            continue

        line = f"  {name}: first_pass={result.first_pass.verdict}"
        if result.recheck is not None:
            line += f" recheck={result.recheck.verdict}"
        elif result.recheck_rate_limited:
            line += " recheck=RATE_LIMITED (left pending, not finalised)"
        print(line)
    return results


def test_all_scenarios_produce_valid_verdicts():
    total_confirmed = 0
    for name, generator_fn in ALL_SCENARIOS.items():
        print(f"\n=== {name} ===")
        events = generator_fn()
        engine = _new_engine()
        vlm_client = MockVLMClient(agreement_rate=1.0, seed=0)
        layer5 = Layer5VLMVerdict(vlm_client, engine, max_calls=30, per_seconds=60.0)

        for result in run_scenario(name, events, layer5):
            if result is None or result.first_pass is None:
                continue
            assert result.first_pass.verdict in ("CONFIRMED", "UNCERTAIN", "NORMAL")
            if result.first_pass.verdict == "CONFIRMED":
                total_confirmed += 1
                # A CONFIRMED first pass must either get a re-check or be
                # explicitly flagged as rate-limited - never silently
                # treated as final off one pass (the Fail-Safe pattern).
                assert result.recheck is not None or result.recheck_rate_limited

    print(f"\nTotal CONFIRMED first-passes across all scenarios: {total_confirmed}")
    assert total_confirmed > 0, "expected at least one scenario (weapon_at_register) to confirm"


def test_recheck_also_spends_the_rate_budget():
    """A budget of exactly 1 call should let the first pass through but
    block its own Fail-Safe re-check - proving the re-check is no longer
    a free call outside the rate limiter."""
    engine = _new_engine()
    vlm_client = MockVLMClient(agreement_rate=1.0, seed=1)
    layer5 = Layer5VLMVerdict(vlm_client, engine, max_calls=1, per_seconds=60.0)

    events = ALL_SCENARIOS["weapon_at_register"]()
    decision = None
    for event in events:
        d = engine.process(event)
        if d.should_call_vlm:
            decision = d
            break
    assert decision is not None, "expected weapon_at_register to escalate"

    first = layer5.handle(decision)
    assert first is not None
    assert not first.rate_limited
    assert first.first_pass.verdict == "CONFIRMED"
    assert first.recheck is None
    assert first.recheck_rate_limited, (
        "the re-check should have been blocked by the same 1-call budget "
        "the first pass just used"
    )

    second = layer5.handle(decision)
    assert second is not None
    assert second.rate_limited, "the budget should already be exhausted"

    print("\nRate limiter correctly gates both the first pass and the re-check.")


if __name__ == "__main__":
    test_all_scenarios_produce_valid_verdicts()
    test_recheck_also_spends_the_rate_budget()
    print("\nAll Layer 5 smoke tests passed.")