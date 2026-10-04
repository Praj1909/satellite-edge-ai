"""
Confidence-Gated 5G-Aware Hybrid Offloading Policy
=====================================================
The main decision engine. Combines:
  1. Model confidence          — how certain is SatNet?
  2. Channel quality (CQI)     — how good is the link?
  3. 5G network slice          — what QoS is needed?
  4. MEC routing (Dijkstra)    — where should processing happen?

Decision outputs:
  LOCAL    — process on edge device, transmit semantic result
  MEC      — offload to nearest available MEC node
  CLOUD    — send raw image to remote cloud server

This is the core novel contribution: joint optimization of
confidence, channel, slice, and topology in one unified policy.
"""

import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from network.slicing  import select_slice, SliceSimulator
from network.channel  import ChannelModel
from network.mec_graph import MECTopology
from network.qos      import SemanticEncoder, get_qci

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "hybrid_policy.json")

CONF_LABEL   = 0.90   # above this: trust edge result, send semantic payload
CONF_OFFLOAD = 0.70   # below this: must offload to MEC or cloud

CLASS_NAMES = [
    "AnnualCrop","Forest","HerbaceousVeg","Highway",
    "Industrial","Pasture","PermanentCrop","Residential",
    "River","SeaLake"
]


class HybridPolicy:
    """
    Per-image offloading decision engine.
    Instantiate once and call decide() for each image.
    """

    def __init__(self, seed: int = 42):
        self.channel  = ChannelModel(seed=seed)
        self.topology = MECTopology(seed=seed)
        self.slicer   = SliceSimulator(seed=seed)
        self.encoder  = SemanticEncoder()
        self._log     = []

    def decide(
        self,
        class_idx:  int,
        confidence: float,
        all_probs:  np.ndarray,
        is_batch:   bool = False,
        cqi_override: int = None,
    ) -> dict:
        """
        Make a complete offloading + transmission decision for one image.
        Returns a rich dict suitable for display and logging.
        """
        t0          = time.perf_counter()
        class_name  = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else "Unknown"

        # 1. Channel state
        ch = self.channel.sample() if cqi_override is None else \
             self.channel.sample(condition=None)
        if cqi_override is not None:
            from network.channel import sinr_to_cqi, CQI_TABLE, cqi_to_semantic_level
            cqi  = cqi_override
            info = CQI_TABLE[cqi]
            sl, sb, sc = cqi_to_semantic_level(cqi)
        else:
            cqi = ch.cqi

        # 2. Slice selection (semantic-aware network slicing)
        slice_dec = select_slice(class_idx, class_name, is_batch)

        # 3. QCI profile
        qci_profile = get_qci(class_idx, is_batch)

        # 4. MEC routing via Dijkstra
        routing = self.topology.select_processing_node(
            confidence=confidence,
            slice_name=slice_dec.slice_name,
            cqi=cqi,
            source="GS",
        )

        # 5. Semantic encoding
        semantic = self.encoder.encode(class_idx, confidence, all_probs, cqi)

        # 6. Slice transmission simulation
        tx_result = self.slicer.transmit(slice_dec, semantic["bytes"])

        # 7. Final decision label
        node_type = routing["node_type"]
        if routing["node"] == "GS":
            decision = "LOCAL"
        elif node_type == "mec":
            decision = "MEC"
        else:
            decision = "CLOUD"

        # 8. Total pipeline latency
        pipeline_ms = routing["total_ms"] + tx_result["total_ms"]
        decision_ms = (time.perf_counter() - t0) * 1000

        result = {
            # Core prediction
            "class_idx":   class_idx,
            "class_name":  class_name,
            "confidence":  round(confidence, 4),
            # Channel
            "cqi":         cqi,
            "cqi_condition": ch.condition if cqi_override is None else "manual",
            "modulation":  ch.modulation if cqi_override is None else "-",
            # Slice
            "slice":       slice_dec.slice_name,
            "slice_reason":slice_dec.reason,
            # QCI
            "qci":         qci_profile.qci_id,
            "qci_delay_budget": qci_profile.delay_ms,
            # Routing
            "decision":    decision,
            "node":        routing["node"],
            "path":        routing["path"],
            "link_ms":     routing["link_ms"],
            "compute_ms":  routing["compute_ms"],
            "routing_reason": routing["reason"],
            # Semantic payload
            "payload_bytes":    semantic["bytes"],
            "payload_content":  semantic["content"],
            "payload_level":    semantic["level"],
            "bw_saving_pct":    semantic["bw_saving_pct"],
            # Transmission
            "tx_ms":       tx_result["total_ms"],
            "pipeline_ms": round(pipeline_ms, 2),
            "decision_ms": round(decision_ms, 3),
        }
        self._log.append(result)
        return result

    def get_summary(self) -> dict:
        if not self._log: return {}
        decisions   = [r["decision"] for r in self._log]
        bw_savings  = [r["bw_saving_pct"] for r in self._log]
        latencies   = [r["pipeline_ms"] for r in self._log]
        slices      = [r["slice"] for r in self._log]
        return {
            "n":               len(self._log),
            "local_pct":       round(decisions.count("LOCAL")/len(decisions)*100, 1),
            "mec_pct":         round(decisions.count("MEC")/len(decisions)*100, 1),
            "cloud_pct":       round(decisions.count("CLOUD")/len(decisions)*100, 1),
            "avg_bw_saving":   round(float(np.mean(bw_savings)), 1),
            "avg_latency_ms":  round(float(np.mean(latencies)), 1),
            "p95_latency_ms":  round(float(np.percentile(latencies, 95)), 1),
            "urllc_pct":       round(slices.count("URLLC")/len(slices)*100, 1),
            "embb_pct":        round(slices.count("eMBB")/len(slices)*100, 1),
        }

    def get_log(self, n: int = 50):
        return list(self._log[-n:])

    def save_results(self):
        summary = self.get_summary()
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump({"summary": summary, "log": self._log[-200:]}, f, indent=2)

    def sweep_thresholds(self, class_idxs, confidences, all_probs_list):
        """Run threshold sweep on a batch of predictions."""
        thresholds = np.arange(0.0, 1.01, 0.05).tolist()
        results = []
        for thresh in thresholds:
            local, mec, cloud, bw_total = 0, 0, 0, 0
            correct = 0
            n = len(class_idxs)
            for i in range(n):
                c = confidences[i]
                probs = all_probs_list[i]
                pred_idx = int(np.argmax(probs))
                if c >= thresh:
                    local += 1
                    if pred_idx == class_idxs[i]: correct += 1
                    bw_total += 32
                else:
                    cloud += 1
                    bw_total += 12288
            edge_acc = correct / local * 100 if local > 0 else 0
            bw_red   = (1 - bw_total / (n * 12288)) * 100
            results.append({
                "threshold":    round(float(thresh), 2),
                "edge_acc":     round(edge_acc, 2),
                "local_pct":    round(local/n*100, 1),
                "cloud_pct":    round(cloud/n*100, 1),
                "bw_reduction": round(bw_red, 2),
            })
        return results
