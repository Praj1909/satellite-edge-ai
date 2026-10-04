"""Baseline model comparison: SatNet vs MobileNetV2, EfficientNet-B0, ResNet-18, ResNet-50."""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import SatNet

RESULTS_PATH = os.path.join(os.path.dirname(__file__),"..","results","baselines.json")
NUM_CLASSES  = 10

def _mobilenet():
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, NUM_CLASSES); return m

def _efficientnet():
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES); return m

def _resnet18():
    m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES); return m

def _resnet50():
    m = models.resnet50(weights=None); m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES); return m

REGISTRY = {
    "SatNet (ours)":   lambda: SatNet(),
    "MobileNetV2":     _mobilenet,
    "EfficientNet-B0": _efficientnet,
    "ResNet-18":       _resnet18,
    "ResNet-50":       _resnet50,
}

def measure_latency(model, n=30, device="cpu"):
    model.eval(); model.to(device)
    x = torch.randn(1,3,64,64).to(device)
    with torch.no_grad():
        for _ in range(5): model(x)   # warmup
    times = []
    with torch.no_grad():
        for _ in range(n):
            t0 = time.perf_counter(); model(x)
            times.append((time.perf_counter()-t0)*1000)
    return {"mean_ms":round(float(np.mean(times)),2),
            "min_ms": round(float(np.min(times)),2),
            "p95_ms": round(float(np.percentile(times,95)),2)}

def run_baseline_comparison(val_loader=None, trained_paths=None, device="cpu"):
    results = []
    for name, factory in REGISTRY.items():
        print(f"  {name}...")
        m = factory()
        n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        path = (trained_paths or {}).get(name)
        if path and os.path.exists(path):
            m.load_state_dict(torch.load(path, map_location=device))
        m.eval(); m.to(device)
        lat = measure_latency(m, device=device)
        val_acc = None
        if val_loader:
            c, t = 0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    p = m(imgs).argmax(1)
                    c += (p==labels).sum().item(); t += labels.size(0)
            val_acc = round(c/t*100, 2)
        results.append({
            "model":     name,
            "params_M":  round(n_params/1e6, 2),
            "mean_ms":   lat["mean_ms"],
            "p95_ms":    lat["p95_ms"],
            "val_acc":   val_acc,
            "edge_ok":   n_params < 10_000_000 and lat["mean_ms"] < 100,
        })
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH,"w") as f: json.dump(results, f, indent=2)
    return results

def load_baseline_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f: return json.load(f)
    return None
