"""
MEC Network Topology with Dijkstra Routing
============================================
Models a Multi-Access Edge Computing network as a weighted directed graph.
Dijkstra's algorithm finds the minimum-latency path from the ground station
to any processing node (MEC or cloud).

Topology:
    GS ----20ms---- MEC_A ----15ms---- CLOUD
    |                 |
   30ms             10ms
    |                 |
   MEC_B ---25ms--- MEC_C

Node types:
  GS    — Ground Station (image source, runs SatNet locally)
  MEC_A — Primary MEC node (5G base station co-located)
  MEC_B — Secondary MEC node (regional)
  MEC_C — Tertiary MEC node (shared)
  CLOUD — Remote cloud server (AWS / GCP equivalent)

NOVEL CONTRIBUTION
------------------
Dynamic MEC selection via Dijkstra on a load-weighted graph.
Node load (CPU %) is added to edge weight in real time, so a
heavily loaded MEC is avoided even if it is topologically closer.
This is called "load-aware semantic offloading" and has not been
implemented for satellite EO before.
"""

import heapq
import random
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class MECNode:
    node_id:      str
    node_type:    str          # "edge_device" | "mec" | "cloud"
    compute_ms:   Tuple[float,float]   # (min, max) inference time
    load_pct:     float = 0.0  # 0–100 % CPU load (dynamic)
    available:    bool  = True
    location:     str   = ""

    def sample_latency(self) -> float:
        base = random.uniform(*self.compute_ms)
        load_penalty = (self.load_pct / 100) * base * 0.8
        return base + load_penalty


# ── Static graph definition ──────────────────────────────────────────────────

NODES: Dict[str, MECNode] = {
    "GS":    MECNode("GS",    "edge_device", (8, 15),    location="Ground Station"),
    "MEC_A": MECNode("MEC_A", "mec",         (12, 25),   location="Primary MEC (5G BS)"),
    "MEC_B": MECNode("MEC_B", "mec",         (15, 30),   location="Secondary MEC"),
    "MEC_C": MECNode("MEC_C", "mec",         (18, 35),   location="Tertiary MEC"),
    "CLOUD": MECNode("CLOUD", "cloud",        (20, 80),   location="Remote Cloud"),
}

# Base link latencies (ms) — symmetric
BASE_EDGES: List[Tuple[str,str,float]] = [
    ("GS",    "MEC_A", 20.0),
    ("GS",    "MEC_B", 30.0),
    ("MEC_A", "MEC_C", 10.0),
    ("MEC_A", "CLOUD", 15.0),
    ("MEC_B", "MEC_C", 25.0),
    ("MEC_C", "CLOUD", 20.0),
]


class MECTopology:
    """
    Weighted directed graph of MEC nodes.
    Edge weights = base latency + load penalty on destination node.
    Dijkstra returns shortest-latency path from source to all targets.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self._nodes  = {k: MECNode(**n.__dict__) for k, n in NODES.items()}
        self._base   = {e[:2]: e[2] for e in BASE_EDGES}
        # Make bidirectional
        for (a, b), w in list(self._base.items()):
            self._base[(b, a)] = w
        self._routing_log: List[dict] = []
        self._t = 0

    # ── Dynamic load simulation ───────────────────────────────────────────────

    def tick(self):
        """Advance simulation time — update node loads."""
        self._t += 1
        for nid, node in self._nodes.items():
            if nid == "GS": continue
            noise = np.random.normal(0, 5)
            node.load_pct = float(np.clip(
                node.load_pct + noise, 5, 95))

    def set_load(self, node_id: str, load_pct: float):
        self._nodes[node_id].load_pct = float(np.clip(load_pct, 0, 100))

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_adj(self) -> Dict[str, List[Tuple[str, float]]]:
        """Build adjacency list with current load-weighted edge costs."""
        adj: Dict[str, List[Tuple[str, float]]] = {n: [] for n in self._nodes}
        for (a, b), base_w in self._base.items():
            dst_node = self._nodes[b]
            if not dst_node.available:
                continue
            # Load penalty: each 10% CPU adds 2ms to the link cost
            load_penalty = (dst_node.load_pct / 10) * 2.0
            adj[a].append((b, base_w + load_penalty))
        return adj

    # ── Dijkstra ──────────────────────────────────────────────────────────────

    def dijkstra(self, source: str) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
        """
        Classic Dijkstra on the load-weighted graph.
        Returns (dist, prev) where dist[v] = min latency from source to v.

        Complexity: O((V + E) log V) with a binary min-heap.
        V = 5 nodes, E = 8 edges — runs in microseconds.
        """
        adj   = self._build_adj()
        dist  = {n: float("inf") for n in self._nodes}
        prev: Dict[str, Optional[str]] = {n: None for n in self._nodes}
        dist[source] = 0.0
        heap = [(0.0, source)]   # (distance, node)

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v, w in adj.get(u, []):
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v]  = u
                    heapq.heappush(heap, (nd, v))

        return dist, prev

    def shortest_path(self, source: str, target: str) -> Tuple[float, List[str]]:
        """Return (total_latency_ms, path_list) for source→target."""
        dist, prev = self.dijkstra(source)
        path, node = [], target
        while node is not None:
            path.append(node)
            node = prev[node]
        path.reverse()
        return dist[target], path

    # ── Routing decision ──────────────────────────────────────────────────────

    def select_processing_node(
        self,
        confidence:  float,
        slice_name:  str,
        cqi:         int,
        source:      str = "GS",
    ) -> dict:
        """
        Select the optimal processing node given:
          - model confidence (high → local is fine)
          - 5G slice requirement (URLLC → must be fast)
          - CQI (poor channel → prefer local)

        Returns routing decision with node, latency, path, reason.
        """
        self.tick()  # update loads
        dist, _ = self.dijkstra(source)

        # Candidates: all reachable nodes except source
        candidates = {
            n: dist[n] for n in self._nodes
            if n != source and self._nodes[n].available and dist[n] < float("inf")
        }

        # --- Routing logic ---
        if cqi <= 3:
            # Channel too poor for reliable upload → process locally
            chosen = source
            reason = f"CQI={cqi} (poor channel) → local processing"
        elif confidence >= 0.90 and slice_name != "URLLC":
            # High confidence, non-critical → local semantic result
            chosen = source
            reason = f"confidence={confidence:.2f} ≥ 0.90, {slice_name} slice → edge-only"
        elif slice_name == "URLLC":
            # Disaster class → find fastest MEC (not cloud)
            mec_nodes = {n: d for n, d in candidates.items()
                         if self._nodes[n].node_type == "mec"}
            if mec_nodes:
                chosen = min(mec_nodes, key=mec_nodes.get)
                reason = f"URLLC slice → nearest MEC {chosen} ({mec_nodes[chosen]:.1f}ms)"
            else:
                chosen = source
                reason = "URLLC slice, no MEC available → local fallback"
        elif confidence < 0.70:
            # Low confidence → cloud for best accuracy
            chosen = "CLOUD"
            reason = f"confidence={confidence:.2f} < 0.70 → cloud verification"
        else:
            # Default: best MEC by Dijkstra
            mec_nodes = {n: d for n, d in candidates.items()
                         if self._nodes[n].node_type in ("mec", "cloud")}
            chosen = min(mec_nodes, key=mec_nodes.get) if mec_nodes else source
            reason = f"eMBB/default → Dijkstra selected {chosen} ({dist.get(chosen,0):.1f}ms)"

        link_lat, path = self.shortest_path(source, chosen)
        compute_lat    = self._nodes[chosen].sample_latency()
        total_lat      = link_lat + compute_lat

        decision = {
            "node":        chosen,
            "node_type":   self._nodes[chosen].node_type,
            "path":        " → ".join(path),
            "link_ms":     round(link_lat, 2),
            "compute_ms":  round(compute_lat, 2),
            "total_ms":    round(total_lat, 2),
            "load_pct":    round(self._nodes[chosen].load_pct, 1),
            "reason":      reason,
            "slice":       slice_name,
            "cqi":         cqi,
            "confidence":  round(confidence, 3),
        }
        self._routing_log.append(decision)
        return decision

    def get_node_loads(self) -> Dict[str, float]:
        return {n: round(self._nodes[n].load_pct, 1) for n in self._nodes}

    def get_topology_data(self) -> dict:
        """Return graph structure for visualization."""
        return {
            "nodes": [{"id": n, "type": self._nodes[n].node_type,
                       "load": self._nodes[n].load_pct,
                       "location": self._nodes[n].location}
                      for n in self._nodes],
            "edges": [{"from": a, "to": b, "base_ms": w}
                      for (a,b), w in self._base.items() if a < b],
        }

    def get_routing_log(self, n: int = 20) -> List[dict]:
        return list(self._routing_log[-n:])
