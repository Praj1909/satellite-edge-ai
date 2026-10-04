"""Full benchmark suite — DSA timings, inference, 5G simulation."""
import os, sys, time, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dsa          import RingBuffer, SlidingWindow, PriorityScheduler, Priority
from utils.image_buffer import ImageRingBuffer
from core.model         import SatNet
from network.slicing    import SliceSimulator, select_slice
from network.channel    import ChannelModel
from network.mec_graph  import MECTopology

RESULTS_PATH = os.path.join(os.path.dirname(__file__),"..","results","benchmarks.json")

def run_all_benchmarks(progress_cb=None):
    def tick(lbl, pct):
        if progress_cb: progress_cb(lbl, pct)

    results = {}

    # 1. Ring Buffer
    tick("Ring Buffer...", 10)
    rb = ImageRingBuffer(capacity=50)
    img = np.random.randn(3,64,64).astype(np.float32)
    N = 500
    t0 = time.perf_counter()
    for _ in range(N): rb.push(img)
    push_us = (time.perf_counter()-t0)*1e6/N
    t0 = time.perf_counter()
    for _ in range(N//2): rb.pop()
    pop_us = (time.perf_counter()-t0)*1e6/(N//2)
    results["ring_buffer"] = {"push_us":round(push_us,2),"pop_us":round(pop_us,2),
                               "overflows":rb.stats()["overflows"],"complexity":"O(1)"}

    # 2. Sliding Window
    tick("Sliding Window...", 25)
    sw = SlidingWindow(50)
    samples = np.random.randn(10000).tolist()
    t0 = time.perf_counter()
    for v in samples: sw.push(v)
    sw_ms = (time.perf_counter()-t0)*1000
    t0 = time.perf_counter()
    for _ in range(10000): _ = sw.mean; _ = sw.window_max
    stat_us = (time.perf_counter()-t0)*1e6/10000
    results["sliding_window"] = {"ten_k_ms":round(sw_ms,2),"stat_us":round(stat_us,3),"complexity":"O(1) amortised"}

    # 3. Priority Queue
    tick("Priority Queue...", 40)
    sched = PriorityScheduler("bench")
    tasks = [("wildfire",Priority.CRITICAL),("archive",Priority.BACKGROUND),
             ("classify",Priority.NORMAL),("ndvi",Priority.HIGH),
             ("log",Priority.LOW),("flood",Priority.CRITICAL),("compress",Priority.BACKGROUND)]
    t0 = time.perf_counter()
    for n,p in tasks: sched.enqueue(fn=lambda:None, name=n, priority=p)
    enq_ms = (time.perf_counter()-t0)*1000
    order = []
    while not sched.is_empty:
        t = sched.run_next()
        if t: order.append(f"P{t.priority}:{t.name}")
    drain_ms = (time.perf_counter()-t0)*1000
    results["priority_queue"] = {"enq_ms":round(enq_ms,3),"drain_ms":round(drain_ms,3),
                                  "order":order,"heap_correct":order[0].startswith("P0"),
                                  "complexity":"O(log n)"}

    # 4. Model inference
    tick("CNN Inference...", 55)
    model = SatNet(); model.eval()
    times = []
    for _ in range(30):
        x = torch.randn(1,3,64,64)
        t0 = time.perf_counter()
        with torch.no_grad(): model(x)
        times.append((time.perf_counter()-t0)*1000)
    results["inference"] = {"mean_ms":round(np.mean(times),2),"min_ms":round(min(times),2),
                             "p95_ms":round(np.percentile(times,95),2),"params":model.count_params()}

    # 5. Dijkstra MEC routing
    tick("Dijkstra MEC routing...", 70)
    topo = MECTopology()
    times_d = []
    for _ in range(100):
        t0 = time.perf_counter()
        topo.dijkstra("GS")
        times_d.append((time.perf_counter()-t0)*1000)
    results["dijkstra"] = {"mean_ms":round(np.mean(times_d),3),"p95_ms":round(np.percentile(times_d,95),3),
                            "nodes":5,"edges":8,"complexity":"O((V+E)log V)"}

    # 6. 5G slice simulation
    tick("5G Slice simulation...", 85)
    ch = ChannelModel(); slicer = SliceSimulator()
    from network.qos import SemanticEncoder
    enc = SemanticEncoder()
    class_names = ["AnnualCrop","Forest","HerbaceousVeg","Highway","Industrial",
                   "Pasture","PermanentCrop","Residential","River","SeaLake"]
    slice_counts = {"URLLC":0,"eMBB":0,"mMTC":0}
    bw_saved = []
    for i in range(200):
        cls = i % 10
        cname = class_names[cls]
        sd = select_slice(cls, cname)
        state = ch.sample()
        sem = enc.encode(cls, 0.9, None, state.cqi)
        tx  = slicer.transmit(sd, sem["bytes"])
        slice_counts[sd.slice_name] += 1
        bw_saved.append(sem["bw_saving_pct"])
    results["network_5g"] = {
        "slice_counts": slice_counts,
        "avg_bw_saving_pct": round(float(np.mean(bw_saved)),1),
        "slice_simulator": slicer.get_stats(),
    }

    tick("Done!", 100)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH,"w") as f: json.dump(results, f, indent=2, default=str)
    return results

def load_benchmark_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f: return json.load(f)
    return None

if __name__ == "__main__":
    print("\nRunning benchmarks...\n")
    r = run_all_benchmarks(progress_cb=lambda l,p: print(f"  [{p:3d}%] {l}"))
    print(f"\nRing Buffer   push: {r['ring_buffer']['push_us']}µs  pop: {r['ring_buffer']['pop_us']}µs")
    print(f"Sliding Win   10k: {r['sliding_window']['ten_k_ms']}ms  stat: {r['sliding_window']['stat_us']}µs")
    print(f"Priority Q    drain: {r['priority_queue']['drain_ms']}ms  correct: {r['priority_queue']['heap_correct']}")
    print(f"CNN Inference mean: {r['inference']['mean_ms']}ms  p95: {r['inference']['p95_ms']}ms")
    print(f"Dijkstra      mean: {r['dijkstra']['mean_ms']}ms")
    print(f"5G Slices     bw_saved: {r['network_5g']['avg_bw_saving_pct']}%\n")
