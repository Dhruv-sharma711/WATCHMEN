"""
Layer 4 - Symbolic Context Arbiter

Before a piece of evidence is fed into the RiskAccumulator, it's checked
against human-readable rules loaded from rules/*.yaml. A rule can:
  - suppress evidence entirely (multiplier = 0.0), e.g. staff badge visible
  - escalate evidence (multiplier > 1.0), e.g. weapon near a register

Rules are intentionally simple condition strings evaluated against the
event's metadata dict - this keeps them auditable and editable by a
non-programmer reviewing the YAML, which is the whole point of doing this
symbolically instead of learning it.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class Rule:
    name: str
    condition: str          # python expression evaluated against event context
    action: str              # "suppress" or "escalate"
    multiplier: float = 0.0  # used only for "escalate"; suppress always -> 0.0


@dataclass
class ArbiterDecision:
    final_multiplier: float
    fired_rule: Optional[str]  # name of the rule that matched, or None


class SymbolicContextArbiter:
    def __init__(self, suppression_rules_path: str, escalation_rules_path: str):
        self.suppression_rules = self._load_rules(suppression_rules_path, default_action="suppress")
        self.escalation_rules = self._load_rules(escalation_rules_path, default_action="escalate")

    @staticmethod
    def _load_rules(path: str, default_action: str) -> List[Rule]:
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or []
        rules = []
        for r in raw:
            rules.append(
                Rule(
                    name=r["name"],
                    condition=r["condition"],
                    action=r.get("action", default_action),
                    multiplier=float(r.get("multiplier", 0.0 if default_action == "suppress" else 1.0)),
                )
            )
        return rules

    @staticmethod
    def _safe_eval(condition: str, event: Dict[str, Any]) -> bool:
        """
        Evaluate a rule condition like "metadata.staff_badge_visible == true"
        against the event dict, without exposing builtins. Supports dotted
        access into nested dicts via a tiny wrapper object.
        """
        class _Dotted:
            def __init__(self, d):
                self._d = d or {}

            def __getattr__(self, key):
                val = self._d.get(key)
                return _Dotted(val) if isinstance(val, dict) else val

        context = {k: (_Dotted(v) if isinstance(v, dict) else v) for k, v in event.items()}
        context["true"] = True
        context["false"] = False
        try:
            return bool(eval(condition, {"__builtins__": {}}, context))
        except Exception:
            # A malformed rule should never crash the pipeline - log and skip it.
            return False

    def evaluate(self, event: Dict[str, Any]) -> ArbiterDecision:
        """
        event example:
        {
          "event_type": "hand_object_interaction",
          "metadata": {"staff_badge_visible": False, "zone": "electronics"}
        }

        Suppression rules are checked first (safety: never escalate something
        that should be zeroed out). Escalation rules only apply if nothing
        was suppressed.
        """
        for rule in self.suppression_rules:
            if self._safe_eval(rule.condition, event):
                return ArbiterDecision(final_multiplier=0.0, fired_rule=rule.name)

        for rule in self.escalation_rules:
            if self._safe_eval(rule.condition, event):
                return ArbiterDecision(final_multiplier=rule.multiplier, fired_rule=rule.name)

        return ArbiterDecision(final_multiplier=1.0, fired_rule=None)