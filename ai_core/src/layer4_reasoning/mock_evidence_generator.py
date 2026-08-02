"""
Mock Evidence Generator

Produces fake Layer 3 -> Layer 4 event streams so you can build and tune
Layer 4 WITHOUT waiting on Manisha's real pipeline. This is the schema you
should confirm with her - if her real output differs, you only need to
change the adapter that feeds events into RiskAccumulator, not your logic.

Each event dict shape:
{
  "shopper_id": str,
  "timestamp": float,
  "event_type": str,          # matches RiskAccumulator.DEFAULT_WEIGHTS keys
  "confidence": float,        # 0-1
  "metadata": {...}           # arbitrary context fields the arbiter rules read
}
"""

import time
from typing import Dict, List


def scenario_normal_shopper(shopper_id: str = "shopper_normal") -> List[Dict]:
    """Browses, picks something up, puts it back. Should stay low-risk."""
    t0 = time.time()
    return [
        {"shopper_id": shopper_id, "timestamp": t0, "event_type": "hand_object_interaction",
         "confidence": 0.9, "metadata": {"zone": "electronics", "staff_badge_visible": False}},
        {"shopper_id": shopper_id, "timestamp": t0 + 5, "event_type": "item_disappeared",
         "confidence": 0.8, "metadata": {"reappeared_in_basket": True, "item_high_value": False}},
    ]


def scenario_staff_restocking(shopper_id: str = "shopper_staff") -> List[Dict]:
    """Staff member handling many items - should be suppressed almost entirely."""
    t0 = time.time()
    events = []
    for i in range(6):
        events.append({
            "shopper_id": shopper_id, "timestamp": t0 + i * 2, "event_type": "hand_object_interaction",
            "confidence": 0.9, "metadata": {"zone": "electronics", "staff_badge_visible": True,
                                             "movement_pattern": "restock"}
        })
    return events


def scenario_slow_theft(shopper_id: str = "shopper_theft") -> List[Dict]:
    """Lingers, checks surroundings, conceals a high-value item. Should escalate."""
    t0 = time.time()
    return [
        {"shopper_id": shopper_id, "timestamp": t0, "event_type": "hand_object_interaction",
         "confidence": 0.85, "metadata": {"zone": "electronics", "staff_badge_visible": False}},
        {"shopper_id": shopper_id, "timestamp": t0 + 3, "event_type": "restricted_zone_entry",
         "confidence": 0.6, "metadata": {"zone": "blind_spot", "staff_badge_visible": False}},
        {"shopper_id": shopper_id, "timestamp": t0 + 8, "event_type": "item_disappeared",
         "confidence": 0.9, "metadata": {"reappeared_in_basket": False, "item_high_value": True,
                                          "staff_badge_visible": False}},
    ]


def scenario_weapon_at_register(shopper_id: str = "shopper_weapon") -> List[Dict]:
    """Immediate high-priority case - should escalate on the very first event."""
    t0 = time.time()
    return [
        {"shopper_id": shopper_id, "timestamp": t0, "event_type": "weapon_detected",
         "confidence": 0.95, "metadata": {"zone": "register", "staff_badge_visible": False}},
    ]


def scenario_item_return(shopper_id: str = "shopper_return") -> List[Dict]:
    """Picks something up, carries it a while, then puts it back on the
    shelf. This is the Gap-5 'return' case - should resolve to
    Certain - Benign once the suppression rule fires, not linger as
    suspicious just because an item briefly disappeared."""
    t0 = time.time()
    return [
        {"shopper_id": shopper_id, "timestamp": t0, "event_type": "hand_object_interaction",
         "confidence": 0.9, "metadata": {"zone": "grocery", "staff_badge_visible": False}},
        {"shopper_id": shopper_id, "timestamp": t0 + 20, "event_type": "item_disappeared",
         "confidence": 0.7, "metadata": {"returned_to_shelf": True, "reappeared_in_basket": False,
                                          "item_high_value": False}},
    ]


def scenario_replica_toy_at_register(shopper_id: str = "shopper_toy") -> List[Dict]:
    """A child's toy gun at the register - looks exactly like
    weapon_detected to Layer 2/3, but should be fully suppressed once the
    replica flag is set (Gap 5)."""
    t0 = time.time()
    return [
        {"shopper_id": shopper_id, "timestamp": t0, "event_type": "weapon_detected",
         "confidence": 0.6, "metadata": {"zone": "register", "item_is_replica": True,
                                          "staff_badge_visible": False}},
    ]


def scenario_contradictory_evidence(shopper_id: str = "shopper_contradiction") -> List[Dict]:
    """Item disappears, apparently reappears, then Layer 3 flags it missing
    again shortly after - the kind of noisy tracker output that should
    surface explicitly as Contradictory rather than being silently
    averaged into one risk number (Table 3)."""
    t0 = time.time()
    return [
        {"shopper_id": shopper_id, "timestamp": t0, "event_type": "item_disappeared",
         "confidence": 0.7, "metadata": {"reappeared_in_basket": False, "item_high_value": True,
                                          "staff_badge_visible": False}},
        {"shopper_id": shopper_id, "timestamp": t0 + 4, "event_type": "item_disappeared",
         "confidence": 0.65, "metadata": {"reappeared_in_basket": True, "item_high_value": True,
                                           "staff_badge_visible": False}},
    ]


def scenario_aggressive_customer_near_till(shopper_id: str = "shopper_aggressive") -> List[Dict]:
    """Aggressive posture right at the register - should escalate through
    the violence category (its own threshold/weight) rather than being
    forced through the concealment path."""
    t0 = time.time()
    return [
        {"shopper_id": shopper_id, "timestamp": t0, "event_type": "aggressive_pose",
         "confidence": 0.8, "metadata": {"zone": "register", "staff_badge_visible": False}},
    ]


ALL_SCENARIOS = {
    "normal_shopper": scenario_normal_shopper,
    "staff_restocking": scenario_staff_restocking,
    "slow_theft": scenario_slow_theft,
    "weapon_at_register": scenario_weapon_at_register,
    "item_return": scenario_item_return,
    "replica_toy_at_register": scenario_replica_toy_at_register,
    "contradictory_evidence": scenario_contradictory_evidence,
    "aggressive_customer_near_till": scenario_aggressive_customer_near_till,
}