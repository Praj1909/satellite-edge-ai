"""
Channel Quality Indicator (CQI) Model
=======================================
Based on 3GPP TS 36.213 Table 7.2.3-1.

CQI 1–15 maps to:
  - SINR range (dB)
  - Modulation scheme (QPSK, 16QAM, 64QAM, 256QAM)
  - Spectral efficiency (bits/s/Hz)
  - Maximum semantic payload size (derived)

NOVEL CONTRIBUTION
------------------
The CQI level jointly determines:
  1. How much semantic information can be reliably transmitted
  2. Whether to use local edge processing or MEC/cloud offload
  3. The compression level of the semantic payload

This CQI-adaptive semantic depth is distinct from all prior work,
which uses fixed payload sizes regardless of channel state.
"""

import random
import numpy as np
from dataclasses import dataclass
from typing import Optional

# 3GPP TS 36.213 Table 7.2.3-1 (simplified)
CQI_TABLE = {
    1:  {"sinr_min": -6.7,  "modulation": "QPSK",   "efficiency": 0.1523, "max_payload_B": 16},
    2:  {"sinr_min": -4.7,  "modulation": "QPSK",   "efficiency": 0.2344, "max_payload_B": 24},
    3:  {"sinr_min": -2.3,  "modulation": "QPSK",   "efficiency": 0.3770, "max_payload_B": 32},
    4:  {"sinr_min":  0.2,  "modulation": "QPSK",   "efficiency": 0.6016, "max_payload_B": 48},
    5:  {"sinr_min":  2.4,  "modulation": "QPSK",   "efficiency": 0.8770, "max_payload_B": 64},
    6:  {"sinr_min":  4.3,  "modulation": "QPSK",   "efficiency": 1.1758, "max_payload_B": 96},
    7:  {"sinr_min":  5.9,  "modulation": "16QAM",  "efficiency": 1.4766, "max_payload_B": 128},
    8:  {"sinr_min":  8.1,  "modulation": "16QAM",  "efficiency": 1.9141, "max_payload_B": 200},
    9:  {"sinr_min": 10.3,  "modulation": "16QAM",  "efficiency": 2.4063, "max_payload_B": 512},
    10: {"sinr_min": 11.7,  "modulation": "64QAM",  "efficiency": 2.7305, "max_payload_B": 1024},
    11: {"sinr_min": 14.1,  "modulation": "64QAM",  "efficiency": 3.3223, "max_payload_B": 2048},
    12: {"sinr_min": 16.3,  "modulation": "64QAM",  "efficiency": 3.9023, "max_payload_B": 4096},
    13: {"sinr_min": 18.7,  "modulation": "64QAM",  "efficiency": 4.5234, "max_payload_B": 8192},
    14: {"sinr_min": 21.0,  "modulation": "256QAM", "efficiency": 5.1152, "max_payload_B": 10240},
    15: {"sinr_min": 22.7,  "modulation": "256QAM", "efficiency": 5.5547, "max_payload_B": 12288},
}

# Semantic payload levels (bytes) — what gets transmitted at each CQI tier
# Level 0 (CQI 1-3):  class index only                     = 1 byte
# Level 1 (CQI 4-6):  class + confidence                   = 8 bytes
# Level 2 (CQI 7-10): class + confidence + top-3 probs     = 20 bytes
# Level 3 (CQI 11-14): class + conf + top-3 + geo-hash     = 32 bytes
# Level 4 (CQI 15):   full feature vector (raw image)      = 12288 bytes

SEMANTIC_LEVELS = {
    "level_0": {"cqi_range": (1, 3),  "bytes": 1,     "content": "class index only"},
    "level_1": {"cqi_range": (4, 6),  "bytes": 8,     "content": "class + confidence"},
    "level_2": {"cqi_range": (7, 10), "bytes": 20,    "content": "class + confidence + top-3"},
    "level_3": {"cqi_range": (11,14), "bytes": 32,    "content": "class + confidence + top-3 + geo-hash"},
    "level_4": {"cqi_range": (15,15), "bytes": 12288, "content": "full raw image"},
}


@dataclass
class ChannelState:
    cqi:          int
    sinr_db:      float
    modulation:   str
    efficiency:   float
    max_payload_B:int
    semantic_level: int     # 0–4
    semantic_bytes: int
    semantic_content: str
    condition:    str       # "poor" / "moderate" / "good" / "excellent"


def cqi_to_semantic_level(cqi: int) -> tuple:
    """Return (semantic_level, bytes, content) for a given CQI."""
    for lvl, info in SEMANTIC_LEVELS.items():
        lo, hi = info["cqi_range"]
        if lo <= cqi <= hi:
            return int(lvl[-1]), info["bytes"], info["content"]
    return 3, 32, "class + confidence + top-3 + geo-hash"


def sinr_to_cqi(sinr_db: float) -> int:
    """Convert SINR (dB) to CQI index (1-15)."""
    best = 1
    for cqi, info in CQI_TABLE.items():
        if sinr_db >= info["sinr_min"]:
            best = cqi
    return best


class ChannelModel:
    """
    Simulates time-varying channel quality for a satellite-to-ground link.
    Models LEO satellite pass geometry: channel improves as satellite
    approaches zenith and degrades toward horizon.
    """
    def __init__(self, seed: int = 42):
        random.seed(seed); np.random.seed(seed)
        self._t = 0
        self._history = []

    def sample(self, condition: Optional[str] = None) -> ChannelState:
        """
        Sample current channel state.
        condition: 'poor'|'moderate'|'good'|'excellent'|None (auto)
        """
        if condition is None:
            # Simulate time-varying SINR (LEO pass model)
            base_sinr = 10 * np.sin(self._t / 50 * np.pi)  # peak at zenith
            noise     = np.random.normal(0, 3)
            sinr      = float(np.clip(base_sinr + noise, -7, 25))
            self._t  += 1
        else:
            ranges = {"poor": (-7, 0), "moderate": (0, 8),
                      "good": (8, 18), "excellent": (18, 25)}
            lo, hi = ranges.get(condition, (0, 25))
            sinr   = float(np.random.uniform(lo, hi))

        cqi = sinr_to_cqi(sinr)
        info = CQI_TABLE[cqi]
        sem_lvl, sem_bytes, sem_content = cqi_to_semantic_level(cqi)

        if   cqi <= 3:  cond = "poor"
        elif cqi <= 6:  cond = "moderate"
        elif cqi <= 11: cond = "good"
        else:           cond = "excellent"

        state = ChannelState(
            cqi=cqi, sinr_db=round(sinr, 2),
            modulation=info["modulation"],
            efficiency=info["efficiency"],
            max_payload_B=info["max_payload_B"],
            semantic_level=sem_lvl,
            semantic_bytes=sem_bytes,
            semantic_content=sem_content,
            condition=cond,
        )
        self._history.append(state)
        return state

    def simulate_pass(self, n_samples: int = 100) -> list:
        """Simulate a full LEO satellite pass (improving then degrading CQI)."""
        return [self.sample() for _ in range(n_samples)]

    def get_cqi_distribution(self, n: int = 200) -> dict:
        states = [self.sample() for _ in range(n)]
        counts = {cqi: 0 for cqi in range(1, 16)}
        for s in states: counts[s.cqi] += 1
        return {k: round(v/n*100, 1) for k, v in counts.items()}

    def get_history(self): return list(self._history)
