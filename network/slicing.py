"""
5G Network Slicing Simulator
==============================
Models the three IMT-2020 / 5G NR service categories defined in 3GPP TS 22.261:

  URLLC  — Ultra-Reliable Low-Latency Communication
            Target: latency ≤ 1 ms, reliability 99.999 %
            Used for: safety-critical detections (wildfire, flood)

  eMBB   — Enhanced Mobile Broadband
            Target: throughput ≥ 100 Mbps, latency ≤ 50 ms
            Used for: standard land-use classification

  mMTC   — massive Machine-Type Communication
            Target: up to 1 M devices/km², delay-tolerant
            Used for: background archiving and batch analytics

NOVEL CONTRIBUTION
------------------
The slice is selected based on the *predicted semantic class* — the AI
output drives network resource allocation.  This closed loop (AI → slice
→ QoS → payload size → latency) is the "semantic-aware network slicing"
concept proposed for 6G but not yet implemented for satellite EO.
"""

import random, time
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict

# EuroSAT class index → disaster risk level → 5G slice
# Industrial (4), Highway (3) → high change-risk → URLLC
# River (8), SeaLake (9)     → flood-risk           → URLLC
# Forest (1)                  → fire-risk            → URLLC
# All others                  → routine              → eMBB
# Archiving / batch           → background           → mMTC

SLICE_MAP = {
    0: "eMBB",   # AnnualCrop
    1: "URLLC",  # Forest       (wildfire risk)
    2: "eMBB",   # HerbaceousVeg
    3: "URLLC",  # Highway      (infrastructure change)
    4: "URLLC",  # Industrial   (pollution / change)
    5: "eMBB",   # Pasture
    6: "eMBB",   # PermanentCrop
    7: "eMBB",   # Residential
    8: "URLLC",  # River        (flood risk)
    9: "URLLC",  # SeaLake      (flood / coastal risk)
}

class Slice(Enum):
    URLLC = "URLLC"
    eMBB  = "eMBB"
    mMTC  = "mMTC"

@dataclass
class SliceParams:
    name:           str
    latency_ms:     float   # target one-way latency
    reliability:    float   # 0–1
    max_payload_B:  int     # max semantic payload bytes
    bandwidth_Mbps: float   # allocated uplink bandwidth
    priority:       int     # lower = higher priority
    description:    str = ""

SLICE_CONFIG: Dict[str, SliceParams] = {
    "URLLC": SliceParams("URLLC", latency_ms=1.0,  reliability=0.99999, max_payload_B=32,    bandwidth_Mbps=10,   priority=0, description="Ultra-Reliable Low-Latency — disaster detection"),
    "eMBB":  SliceParams("eMBB",  latency_ms=50.0, reliability=0.9999,  max_payload_B=12288, bandwidth_Mbps=100,  priority=1, description="Enhanced Mobile Broadband — standard classification"),
    "mMTC":  SliceParams("mMTC",  latency_ms=500.0,reliability=0.999,   max_payload_B=256,   bandwidth_Mbps=1,    priority=2, description="Massive Machine-Type Comms — background archiving"),
}


@dataclass
class SliceDecision:
    slice_name:     str
    params:         SliceParams
    class_idx:      int
    class_name:     str
    latency_budget: float   # ms available for transmission
    payload_limit:  int     # bytes allowed on this slice
    reason:         str


def select_slice(class_idx: int, class_name: str, is_batch: bool = False) -> SliceDecision:
    """
    Map a predicted class to a 5G network slice.
    This is the semantic-aware network slicing decision.
    """
    if is_batch:
        sname = "mMTC"
        reason = "batch/archive task → mMTC (delay-tolerant)"
    else:
        sname = SLICE_MAP.get(class_idx, "eMBB")
        reason = ("disaster-risk class → URLLC (ultra-low latency)"
                  if sname == "URLLC"
                  else "routine class → eMBB (standard broadband)")

    params = SLICE_CONFIG[sname]
    return SliceDecision(
        slice_name=sname, params=params,
        class_idx=class_idx, class_name=class_name,
        latency_budget=params.latency_ms,
        payload_limit=params.max_payload_B,
        reason=reason,
    )


class SliceSimulator:
    """
    Simulates end-to-end transmission over a 5G network slice,
    returning realistic latency values per slice type.
    """
    def __init__(self, seed: int = 42):
        random.seed(seed); np.random.seed(seed)
        self._history: List[dict] = []

    def transmit(self, decision: SliceDecision, payload_bytes: int) -> dict:
        p = decision.params

        # Slice-specific latency model
        if decision.slice_name == "URLLC":
            # URLLC: sub-ms air-interface + short processing
            tx_ms  = random.uniform(0.3, 1.0)
            proc_ms = random.uniform(0.5, 2.0)
            jitter  = random.uniform(0.0, 0.3)
        elif decision.slice_name == "eMBB":
            # eMBB: standard 5G NR scheduling
            tx_ms   = (payload_bytes * 8) / (p.bandwidth_Mbps * 1e3)   # ms
            proc_ms = random.uniform(5, 30)
            jitter  = random.uniform(1, 10)
        else:   # mMTC
            tx_ms   = (payload_bytes * 8) / (p.bandwidth_Mbps * 1e3)
            proc_ms = random.uniform(50, 300)
            jitter  = random.uniform(5, 50)

        total_ms = tx_ms + proc_ms + jitter
        sla_met  = total_ms <= p.latency_ms * 10   # 10× slack for simulation

        result = {
            "slice":       decision.slice_name,
            "payload_B":   payload_bytes,
            "tx_ms":       round(tx_ms, 3),
            "proc_ms":     round(proc_ms, 3),
            "jitter_ms":   round(jitter, 3),
            "total_ms":    round(total_ms, 3),
            "sla_met":     sla_met,
            "class":       decision.class_name,
        }
        self._history.append(result)
        return result

    def get_stats(self) -> dict:
        if not self._history: return {}
        by_slice: Dict[str, list] = {}
        for r in self._history:
            by_slice.setdefault(r["slice"], []).append(r["total_ms"])
        out = {}
        for s, times in by_slice.items():
            out[s] = {
                "count":   len(times),
                "mean_ms": round(np.mean(times), 2),
                "p95_ms":  round(np.percentile(times, 95), 2),
                "sla_met": sum(1 for r in self._history if r["slice"]==s and r["sla_met"]),
            }
        return out

    def get_history(self) -> List[dict]:
        return list(self._history)
