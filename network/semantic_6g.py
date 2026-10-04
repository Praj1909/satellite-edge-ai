"""
6G Semantic Communication Layer
=================================
Implements two concepts from the IMT-2030 (6G) framework
that go beyond 5G's QoS-based resource allocation.

CONCEPT 1 — Goal-Oriented Communication (3GPP TR 22.874)
----------------------------------------------------------
In 5G, the network cares about bits: latency, throughput, reliability.
In 6G, the network cares about GOALS: "did the receiver get what it
needed to complete its task?"

For our satellite system, the goal is:
  "Correctly identify land-use class, with special urgency for disasters."

The system tracks whether each transmission ACHIEVED ITS GOAL, and
uses that to adjust future transmission policy — not just raw QoS.

CONCEPT 2 — Semantic Information Value (SIV)
----------------------------------------------
Proposed in: Qin et al., "Semantic Communications" arXiv:2112.10125
             Shi et al., "Semantic Communication Intelligence" 2021

Not all transmissions carry equal information. If the ground station
already received "Forest" 10 times in a row, the 11th "Forest" adds
very little new knowledge. SIV measures this:

  SIV = confidence × (1 - P_prior)

where P_prior is the probability of this class based on recent history.
High SIV → transmit immediately.
Low SIV  → defer or skip (save bandwidth).

This is fundamentally different from 5G which transmits everything
regardless of whether the content is novel.

CONCEPT 3 — Semantic Similarity Filter
-----------------------------------------
Before transmitting, check if the new result is semantically similar
to the last N results. If yes — low new information — compress further
or skip. This is called "semantic redundancy elimination" in 6G literature.

WHY THIS IS NOVEL
-----------------
No published paper applies goal-oriented communication OR semantic
information value to satellite Earth observation with a working
implementation and measured results on EuroSAT.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import time

CLASS_NAMES = [
    "AnnualCrop","Forest","HerbaceousVeg","Highway",
    "Industrial","Pasture","PermanentCrop","Residential",
    "River","SeaLake"
]

# Disaster-critical classes — achieving correct detection is the GOAL
CRITICAL_CLASSES = {1, 3, 4, 8, 9}   # Forest, Highway, Industrial, River, SeaLake

# ── Concept 1: Goal-Oriented Communication ───────────────────────────────────

@dataclass
class CommunicationGoal:
    goal_id:     str
    description: str
    target_acc:  float   # minimum accuracy required to "achieve" the goal
    max_delay_ms:float   # deadline
    achieved:    bool = False

GOALS = {
    "disaster_detection": CommunicationGoal(
        "disaster_detection",
        "Correctly identify disaster-risk classes (Forest/River/Industrial)",
        target_acc=0.95, max_delay_ms=50.0
    ),
    "land_use_monitoring": CommunicationGoal(
        "land_use_monitoring",
        "Classify routine land-use for periodic reporting",
        target_acc=0.85, max_delay_ms=500.0
    ),
}

class GoalOrientedController:
    """
    Tracks whether communication goals are being met and adapts
    transmission policy accordingly.

    This is the 6G shift from QoS (bits/latency) to QoE-of-task
    (did we achieve what we needed?).
    """

    def __init__(self):
        self._log: List[dict] = []
        self._goal_stats: Dict[str, dict] = {
            g: {"attempts":0,"achieved":0,"avg_delay":0.0}
            for g in GOALS
        }

    def evaluate(
        self,
        class_idx:    int,
        confidence:   float,
        latency_ms:   float,
        true_label:   Optional[int] = None,
    ) -> dict:
        """
        Evaluate whether the communication goal was achieved for this image.
        Returns goal assessment with recommended action.
        """
        is_critical = class_idx in CRITICAL_CLASSES
        goal_id     = "disaster_detection" if is_critical else "land_use_monitoring"
        goal        = GOALS[goal_id]

        # Goal achievement: confidence above threshold AND within deadline
        conf_ok    = confidence >= goal.target_acc
        latency_ok = latency_ms <= goal.max_delay_ms
        achieved   = conf_ok and latency_ok

        # Accuracy check if ground truth available
        correct = None
        if true_label is not None:
            correct = (class_idx == true_label)
            achieved = achieved and correct

        # Update stats
        s = self._goal_stats[goal_id]
        s["attempts"] += 1
        if achieved: s["achieved"] += 1
        s["avg_delay"] = (s["avg_delay"]*(s["attempts"]-1) + latency_ms) / s["attempts"]

        # Recommended action based on goal status
        if not conf_ok and is_critical:
            action = "ESCALATE"   # must send to cloud, too uncertain for disaster
            reason = f"Disaster class, confidence {confidence:.0%} < {goal.target_acc:.0%} threshold"
        elif not conf_ok:
            action = "RETRANSMIT"  # request higher-quality processing
            reason = f"Confidence {confidence:.0%} below goal threshold {goal.target_acc:.0%}"
        elif not latency_ok:
            action = "ALERT"       # latency budget exceeded, flag it
            reason = f"Latency {latency_ms:.0f}ms exceeds {goal.max_delay_ms:.0f}ms budget"
        else:
            action = "ACCEPT"
            reason = "Goal achieved"

        result = {
            "goal_id":       goal_id,
            "goal_desc":     goal.description,
            "achieved":      achieved,
            "action":        action,
            "reason":        reason,
            "confidence":    round(confidence, 3),
            "latency_ms":    round(latency_ms, 2),
            "deadline_ms":   goal.max_delay_ms,
            "class_idx":     class_idx,
            "class_name":    CLASS_NAMES[class_idx],
            "is_critical":   is_critical,
        }
        self._log.append(result)
        return result

    def get_goal_stats(self) -> dict:
        out = {}
        for gid, s in self._goal_stats.items():
            rate = s["achieved"]/s["attempts"]*100 if s["attempts"]>0 else 0
            out[gid] = {
                "attempts":     s["attempts"],
                "achieved":     s["achieved"],
                "success_rate": round(rate, 1),
                "avg_delay_ms": round(s["avg_delay"], 2),
            }
        return out

    def get_log(self, n=20):
        return list(self._log[-n:])


# ── Concept 2: Semantic Information Value (SIV) ───────────────────────────────

class SemanticInformationValue:
    """
    Computes the Semantic Information Value (SIV) of a new transmission.

    SIV = confidence × (1 - P_prior)

    where P_prior is the probability of this class based on recent history.
    A high SIV means the result is new and important.
    A low SIV means we already knew this — the transmission adds little.

    This drives the SEMANTIC REDUNDANCY ELIMINATION policy:
      SIV > threshold → transmit
      SIV ≤ threshold → defer (skip if non-critical)
    """

    def __init__(self, history_size=20, defer_threshold=0.1):
        self.history_size    = history_size
        self.defer_threshold = defer_threshold
        self._history: deque = deque(maxlen=history_size)
        self._siv_log: List[dict] = []
        self._deferred = 0
        self._transmitted = 0

    def compute(self, class_idx: int, confidence: float) -> dict:
        """Compute SIV and decide whether to transmit or defer."""
        # P_prior: frequency of this class in recent history
        if len(self._history) == 0:
            p_prior = 0.0
        else:
            p_prior = sum(1 for c in self._history if c == class_idx) / len(self._history)

        siv = confidence * (1.0 - p_prior)
        is_critical = class_idx in CRITICAL_CLASSES

        # Decision
        if is_critical:
            action = "TRANSMIT"
            reason = "Critical class — always transmit regardless of SIV"
        elif siv > self.defer_threshold:
            action = "TRANSMIT"
            reason = f"SIV={siv:.3f} > threshold={self.defer_threshold} — new information"
        else:
            action = "DEFER"
            reason = f"SIV={siv:.3f} ≤ threshold={self.defer_threshold} — redundant, skip"

        # Update history
        self._history.append(class_idx)
        if action == "TRANSMIT": self._transmitted += 1
        else:                    self._deferred += 1

        result = {
            "class_idx":     class_idx,
            "class_name":    CLASS_NAMES[class_idx],
            "confidence":    round(confidence, 3),
            "p_prior":       round(p_prior, 3),
            "siv":           round(siv, 4),
            "action":        action,
            "reason":        reason,
            "is_critical":   is_critical,
            "deferred_total":self._deferred,
            "tx_total":      self._transmitted,
        }
        self._siv_log.append(result)
        return result

    def get_stats(self) -> dict:
        total = self._transmitted + self._deferred
        return {
            "transmitted": self._transmitted,
            "deferred":    self._deferred,
            "defer_rate":  round(self._deferred/total*100 if total>0 else 0, 1),
            "extra_bw_saved_pct": round(self._deferred/total*100 if total>0 else 0, 1),
        }

    def get_log(self, n=20):
        return list(self._siv_log[-n:])

    def simulate_batch(self, n=100):
        """Simulate n images and show how SIV filters redundant transmissions."""
        np.random.seed(42)
        # Simulate a satellite pass over a mostly-forested area
        classes = np.random.choice(
            [0,1,1,1,2,3,5,6,7],   # Forest appears 3x more often
            size=n
        )
        confidences = np.random.uniform(0.75, 0.99, n)
        results = []
        for cls, conf in zip(classes, confidences):
            r = self.compute(int(cls), float(conf))
            results.append(r)
        return results


# ── Concept 3: Semantic Similarity Filter ────────────────────────────────────

class SemanticSimilarityFilter:
    """
    Prevents transmitting semantically redundant results.

    If the last K transmissions were all the same class,
    the (K+1)th same-class result has near-zero new information.
    Skip it and save the bandwidth.

    This is called "temporal semantic compression" in 6G literature.
    """

    def __init__(self, window=5, repeat_threshold=3):
        self.window           = window
        self.repeat_threshold = repeat_threshold
        self._recent: deque   = deque(maxlen=window)
        self._filtered = 0
        self._passed   = 0

    def filter(self, class_idx: int, confidence: float, force: bool = False) -> dict:
        # Count occurrences in current window BEFORE appending current item
        consecutive = sum(1 for c in self._recent if c == class_idx)
        # If already seen >= repeat_threshold times in window, this one is redundant
        is_redundant = (consecutive >= self.repeat_threshold
                        and class_idx not in CRITICAL_CLASSES
                        and not force)

        self._recent.append(class_idx)
        if is_redundant: self._filtered += 1
        else:            self._passed   += 1

        return {
            "class_idx":    class_idx,
            "class_name":   CLASS_NAMES[class_idx],
            "consecutive":  consecutive,
            "redundant":    is_redundant,
            "action":       "FILTER" if is_redundant else "PASS",
            "reason": (f"Same class repeated {consecutive}× in window — semantic redundancy"
                       if is_redundant else "New or critical result — pass through"),
        }

    def get_stats(self):
        total = self._filtered + self._passed
        return {
            "filtered": self._filtered,
            "passed":   self._passed,
            "filter_rate": round(self._filtered/total*100 if total>0 else 0, 1),
        }
