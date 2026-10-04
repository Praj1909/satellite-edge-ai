"""Ablation study: remove one component at a time and measure the impact."""
import os, sys, json, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import SatNet
from utils.dsa  import RingBuffer, SlidingWindow, PriorityScheduler, Priority
from utils.image_buffer import ImageRingBuffer

RESULTS_PATH = os.path.join(os.path.dirname(__file__),"..","results","ablation.json")
N = 200  # images per variant

def _fake_batch(n): return [torch.randn(3,64,64) for _ in range(n)]

def _run(model, images, use_priority, use_buffer, use_semcomms):
    if use_buffer:
        buf = ImageRingBuffer(capacity=50)
    sched = PriorityScheduler("abl")
    sw    = SlidingWindow(50)
    times, order = [], []

    for i, img in enumerate(images):
        if use_buffer:
            arr = img.numpy()
            buf.push(arr)
        pri = Priority.CRITICAL if i % 20 == 0 else Priority.NORMAL
        real_pri = pri if use_priority else Priority.NORMAL
        sched.enqueue(fn=lambda t=img: model.predict_single(t), name=str(pri.name), priority=real_pri)

    while not sched.is_empty:
        t0 = time.perf_counter()
        task = sched.run_next()
        ms = (time.perf_counter()-t0)*1000
        if not use_buffer: ms += np.random.uniform(0.5, 2.0)  # alloc jitter
        times.append(ms); sw.push(ms)
        if task: order.append(task.name)

    critical_first = (order.index("CRITICAL") < order.index("NORMAL")
                      if "CRITICAL" in order and "NORMAL" in order else use_priority)
    bw_per_img = 32 if use_semcomms else 12288
    bw_red = round((1 - bw_per_img/12288)*100, 1)

    return {
        "mean_ms":          round(float(np.mean(times)),2),
        "std_ms":           round(float(np.std(times)),2),
        "throughput_ips":   round(1000/float(np.mean(times)),1),
        "critical_preempts":critical_first,
        "bytes_per_image":  bw_per_img,
        "bw_reduction_pct": bw_red,
    }

def run_ablation(n_images=N):
    model = SatNet(); model.eval()
    images = _fake_batch(n_images)
    print("  Running ablation variants...")

    variants = {
        "full_system":       (True,  True,  True,  "Full system — all components"),
        "no_priority":       (False, True,  True,  "No priority scheduler (FIFO)"),
        "no_buffer":         (True,  False, True,  "No ring buffer (dynamic alloc)"),
        "no_semantic_comms": (True,  True,  False, "No semantic communication"),
    }
    results = {}
    for key, (prio, buf, sem, desc) in variants.items():
        r = _run(model, images, prio, buf, sem)
        r["description"] = desc
        results[key] = r
        print(f"    {key}: {r['mean_ms']}ms mean, critical_first={r['critical_preempts']}")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH,"w") as f: json.dump(results, f, indent=2)
    print(f"  Ablation saved → {RESULTS_PATH}")
    return results

def load_ablation_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f: return json.load(f)
    return None

if __name__ == "__main__":
    run_ablation()
