"""
QoS Manager and Adaptive Semantic Encoder
==========================================
Maps predicted classes to 3GPP QoS Class Identifiers (QCI),
and adaptively encodes semantic payloads based on CQI.

QCI Table (3GPP TS 23.203 Table 6.1.7.2-1 — simplified):
  QCI 1 : GBR,  priority 2,  delay 100ms  — conversational voice
  QCI 2 : GBR,  priority 4,  delay 150ms  — video streaming
  QCI 65: GBR,  priority 0.7,delay 75ms   — mission-critical (used here)
  QCI 70: GBR,  priority 5.5,delay 200ms  — V2X
  QCI 9 : Non-GBR, priority 9, delay 300ms — default internet

We define custom QCIs for satellite EO:
  QCI-S1: CRITICAL EO (wildfire/flood)  — priority 1, delay 50ms
  QCI-S2: STANDARD EO (land-use)        — priority 5, delay 500ms
  QCI-S3: BATCH EO   (archive)          — priority 9, delay 5000ms
"""

import struct
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class QCIProfile:
    qci_id:      str
    priority:    int        # lower = higher priority
    delay_ms:    float      # packet delay budget
    loss_rate:   float      # max packet loss (0–1)
    gbr:         bool       # guaranteed bit rate?
    description: str


QCI_PROFILES = {
    "QCI-S1": QCIProfile("QCI-S1", priority=1, delay_ms=50,   loss_rate=1e-5, gbr=True,  description="Critical EO — wildfire/flood"),
    "QCI-S2": QCIProfile("QCI-S2", priority=5, delay_ms=500,  loss_rate=1e-3, gbr=False, description="Standard EO — land-use classification"),
    "QCI-S3": QCIProfile("QCI-S3", priority=9, delay_ms=5000, loss_rate=1e-2, gbr=False, description="Batch EO — archive / background analytics"),
}

# EuroSAT class → QCI profile
CLASS_QCI_MAP = {
    0: "QCI-S2",   # AnnualCrop
    1: "QCI-S1",   # Forest       (wildfire risk)
    2: "QCI-S2",   # HerbaceousVeg
    3: "QCI-S1",   # Highway
    4: "QCI-S1",   # Industrial
    5: "QCI-S2",   # Pasture
    6: "QCI-S2",   # PermanentCrop
    7: "QCI-S2",   # Residential
    8: "QCI-S1",   # River         (flood)
    9: "QCI-S1",   # SeaLake       (flood/coastal)
}


def get_qci(class_idx: int, is_batch: bool = False) -> QCIProfile:
    if is_batch:
        return QCI_PROFILES["QCI-S3"]
    return QCI_PROFILES[CLASS_QCI_MAP.get(class_idx, "QCI-S2")]


# ── Adaptive Semantic Encoder ─────────────────────────────────────────────────

class SemanticEncoder:
    """
    Encodes inference results into variable-length semantic payloads
    based on channel quality (CQI).

    The payload depth adapts to channel conditions:
      CQI 1–3  : 1 byte   (class index only)
      CQI 4–6  : 8 bytes  (class + confidence float32)
      CQI 7–10 : 20 bytes (+ top-3 class indices + probs)
      CQI 11–14: 32 bytes (+ geo-hash placeholder)
      CQI 15   : raw flag → escalate to full image upload

    This is the key semantic-communication contribution:
    channel-adaptive payload depth.
    """

    BANDWIDTH_SAVINGS = {
        1:  round((1 - 1/12288)*100,    2),
        8:  round((1 - 8/12288)*100,    2),
        20: round((1 - 20/12288)*100,   2),
        32: round((1 - 32/12288)*100,   2),
        12288: 0.0,
    }

    def encode(
        self,
        class_idx:  int,
        confidence: float,
        all_probs:  Optional[np.ndarray],
        cqi:        int,
        geo_hash:   int = 0,
    ) -> dict:
        """
        Produce a semantic payload dict for the given CQI level.
        Returns payload bytes + metadata.
        """
        if cqi <= 3:
            # Level 0: just the class byte
            payload_bytes = struct.pack("B", class_idx)
            level = 0; content = "class index only"
        elif cqi <= 6:
            # Level 1: class (uint8) + confidence (float32) = 5 bytes padded to 8
            payload_bytes = struct.pack("Bf", class_idx, confidence) + b"\x00\x00\x00"
            level = 1; content = "class + confidence"
        elif cqi <= 10:
            # Level 2: level-1 + top-3 class indices + probs (3×uint8 + 3×float32)
            top3_idx  = (np.argsort(all_probs)[-3:][::-1]
                         if all_probs is not None else [class_idx, 0, 0])
            top3_prob = ([float(all_probs[i]) for i in top3_idx]
                         if all_probs is not None else [confidence, 0.0, 0.0])
            payload_bytes = (struct.pack("Bf", class_idx, confidence) +
                             struct.pack("BBB", *[int(i) for i in top3_idx]) +
                             struct.pack("fff", *top3_prob) +
                             b"\x00")   # pad to 20
            level = 2; content = "class + confidence + top-3"
        elif cqi <= 14:
            # Level 3: level-2 + 4-byte geo-hash = 32 bytes
            top3_idx  = (np.argsort(all_probs)[-3:][::-1]
                         if all_probs is not None else [class_idx, 0, 0])
            top3_prob = ([float(all_probs[i]) for i in top3_idx]
                         if all_probs is not None else [confidence, 0.0, 0.0])
            payload_bytes = (struct.pack("Bf", class_idx, confidence) +
                             struct.pack("BBB", *[int(i) for i in top3_idx]) +
                             struct.pack("fff", *top3_prob) +
                             struct.pack("I", geo_hash) +
                             b"\x00" * 7)   # pad to 32
            level = 3; content = "class + confidence + top-3 + geo-hash"
        else:
            # Level 4: full image (signal to upload raw)
            payload_bytes = b"\xFF" * 4   # flag — actual image sent separately
            level = 4; content = "raw image upload (CQI=15)"

        n_bytes = len(payload_bytes) if level < 4 else 12288
        savings = self.BANDWIDTH_SAVINGS.get(n_bytes, round((1-n_bytes/12288)*100, 2))

        return {
            "level":           level,
            "content":         content,
            "bytes":           n_bytes,
            "payload_hex":     payload_bytes[:8].hex(),
            "cqi":             cqi,
            "bw_saving_pct":   savings,
            "upload_raw":      level == 4,
        }
