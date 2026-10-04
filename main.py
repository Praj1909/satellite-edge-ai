"""
Satellite Edge AI — 5G-Aware Dashboard
Run: streamlit run main.py
"""
import streamlit as st
import numpy as np
import time, os, sys, json
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.model         import SatNet, load_model, MODEL_PATH, CLASS_NAMES
from core.hybrid_policy import HybridPolicy
from core.baselines     import run_baseline_comparison, load_baseline_results
from core.ablation      import run_ablation, load_ablation_results
from network.slicing    import SliceSimulator, select_slice, SLICE_CONFIG
from network.channel    import ChannelModel, CQI_TABLE, SEMANTIC_LEVELS
from network.mec_graph  import MECTopology
from network.qos        import SemanticEncoder, get_qci, QCI_PROFILES
from network.semantic_6g import GoalOrientedController, SemanticInformationValue, SemanticSimilarityFilter, GOALS
from utils.dsa          import RingBuffer, SlidingWindow, PriorityScheduler, Priority
from utils.image_buffer import ImageRingBuffer
from benchmarks.run_benchmark import run_all_benchmarks, load_benchmark_results

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Satellite Edge AI", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#0f1117;}
[data-testid="stSidebar"]{background:#0a0c12 !important;border-right:1px solid #1e2230;}
[data-testid="metric-container"]{background:#161b27;border:1px solid #1e2230;border-radius:8px;padding:16px !important;}
[data-testid="metric-container"] [data-testid="stMetricLabel"]{color:#6b7a99 !important;font-size:10px !important;letter-spacing:1px;text-transform:uppercase;font-family:'IBM Plex Mono',monospace !important;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:#e2e8f0 !important;font-size:22px !important;font-weight:600;font-family:'IBM Plex Mono',monospace !important;}
h1,h2,h3{color:#e2e8f0 !important;font-weight:600;}
[data-testid="stTabs"] button{font-family:'IBM Plex Mono',monospace !important;font-size:11px !important;color:#4a5568 !important;letter-spacing:1px;text-transform:uppercase;}
[data-testid="stTabs"] button[aria-selected="true"]{color:#3b82f6 !important;border-bottom-color:#3b82f6 !important;}
.stButton>button{background:#3b82f6 !important;color:#fff !important;font-weight:500 !important;border:none !important;border-radius:6px !important;padding:8px 20px !important;}
.stButton>button:hover{background:#2563eb !important;}
.card{background:#161b27;border:1px solid #1e2230;border-left:3px solid #3b82f6;border-radius:6px;padding:12px 16px;margin:5px 0;font-family:'IBM Plex Mono',monospace;font-size:12px;color:#6b7a99;line-height:1.9;}
.card-green{background:#161b27;border:1px solid #1e2230;border-left:3px solid #10b981;border-radius:6px;padding:12px 16px;margin:5px 0;font-size:13px;color:#d1fae5;}
.card-amber{background:#161b27;border:1px solid #1e2230;border-left:3px solid #f59e0b;border-radius:6px;padding:10px 14px;margin:5px 0;font-size:12px;color:#6b7a99;}
.card-red{background:#161b27;border:1px solid #1e2230;border-left:3px solid #ef4444;border-radius:6px;padding:10px 14px;margin:5px 0;font-size:12px;color:#6b7a99;}
.sec{font-size:10px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#4a5568;font-family:'IBM Plex Mono',monospace;margin:14px 0 6px;padding-bottom:5px;border-bottom:1px solid #1e2230;}
.badge{display:inline-block;background:#1e2230;border:1px solid #2d3748;border-radius:4px;padding:2px 8px;font-family:'IBM Plex Mono',monospace;font-size:10px;color:#6b7a99;margin:2px;}
hr{border-color:#1e2230 !important;}
.stProgress>div>div{background:#3b82f6 !important;}
</style>
""", unsafe_allow_html=True)

# ── Theme helpers ──────────────────────────────────────────────────────────────
BG="#0f1117"; BG2="#161b27"; GRID="#1e2230"; TEXT="#6b7a99"
BLUE="#3b82f6"; GREEN="#10b981"; AMBER="#f59e0b"; RED="#ef4444"; MUTE="#4a5568"

def sfig(fig, title="", height=280):
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG2,
        font=dict(family="IBM Plex Mono", color=TEXT, size=11),
        title=dict(text=title, font=dict(color="#c5cedd", size=12), x=0),
        xaxis=dict(gridcolor=GRID, color=MUTE, zeroline=False),
        yaxis=dict(gridcolor=GRID, color=MUTE, zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
        margin=dict(l=8,r=8,t=36,b=8), height=height,
    )
    return fig

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in {
    "model": None, "policy": HybridPolicy(),
    "img_buffer": ImageRingBuffer(50),
    "scheduler": PriorityScheduler("main"),
    "sw": SlidingWindow(50),
    "channel": ChannelModel(),
    "topology": MECTopology(),
    "encoder": SemanticEncoder(),
    "goal_ctrl": GoalOrientedController(),
    "siv": SemanticInformationValue(),
    "sim_filter": SemanticSimilarityFilter(),
    "history": [], "bench": None,
}.items():
    if k not in st.session_state: st.session_state[k] = v

INFER_TF = transforms.Compose([
    transforms.Resize((64,64)), transforms.ToTensor(),
    transforms.Normalize([0.3444,0.3803,0.4078],[0.2025,0.1364,0.1153]),
])

def get_model():
    if st.session_state.model is None:
        if os.path.exists(MODEL_PATH):
            st.session_state.model = load_model(MODEL_PATH)
        else:
            m = SatNet(); m.eval(); st.session_state.model = m
    return st.session_state.model

get_model()
trained = os.path.exists(MODEL_PATH)

SLICE_COLORS = {"URLLC": RED, "eMBB": BLUE, "mMTC": AMBER}
CLASS_COLORS  = [AMBER,GREEN,"#22d3ee",MUTE,RED,AMBER,GREEN,RED,BLUE,"#1e2230"]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Satellite Edge AI")
    st.markdown('<div class="sec">Model</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="card">SatNet &nbsp;&nbsp; {'Trained ✓' if trained else 'Demo mode'}<br>
Dataset &nbsp; EuroSAT (27k)<br>Classes &nbsp; 10 land-use<br>Params &nbsp;&nbsp; ~1.2 M</div>""", unsafe_allow_html=True)
    if not trained:
        st.markdown('<div class="card-amber">Train first:<br><code>python train_satnet.py</code></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec">5G / 6G Layer</div>', unsafe_allow_html=True)
    st.markdown("""<div class="card">Network Slicing — URLLC/eMBB/mMTC<br>
CQI Channel Model — 3GPP TS 36.213<br>
MEC Routing — Dijkstra O((V+E)log V)<br>
Semantic Encoder — CQI-adaptive<br>
QCI Profiles — 3GPP TS 23.203</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec">DSA Components</div>', unsafe_allow_html=True)
    st.markdown("""<div class="card">Ring Buffer — O(1)<br>
Sliding Window — O(1)<br>
Monotonic Deque — O(1) max<br>
Min-Heap Scheduler — O(log n)<br>
Dijkstra — O((V+E)log V)</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec">Semantic Comms</div>', unsafe_allow_html=True)
    st.markdown("""<div class="card">CQI 1–3 &nbsp; 1 byte (class only)<br>
CQI 4–6 &nbsp; 8 bytes<br>
CQI 7–10 &nbsp; 20 bytes<br>
CQI 11–14 32 bytes<br>
vs raw image: 12 288 bytes</div>""", unsafe_allow_html=True)

st.markdown("## Satellite Edge AI — 5G-Aware Pipeline")
st.markdown('<div style="color:#4a5568;font-size:13px;margin-bottom:18px;">EuroSAT classification · 5G Network Slicing · CQI-Adaptive Semantic Communication · MEC Dijkstra Routing · DSA Pipeline</div>', unsafe_allow_html=True)
st.divider()

tabs = st.tabs(["CLASSIFY","5G NETWORK","MEC ROUTING","CHANNEL QUALITY","6G SEMANTIC","DSA VISUALIZER","BENCHMARKS","BASELINES"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — CLASSIFY
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("## Live Classification + 5G Offloading Decision")
    st.markdown('<div style="color:#4a5568;font-size:13px;margin-bottom:12px;">Upload a satellite image → SatNet classifies → 5G slice selected → MEC node chosen via Dijkstra → semantic payload encoded by CQI</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload satellite image", type=["jpg","jpeg","png","tif"])
    if uploaded:
        img_pil    = Image.open(uploaded).convert("RGB")
        img_tensor = INFER_TF(img_pil)
        arr = np.array(img_pil.resize((64,64))).transpose(2,0,1).astype(np.float32)/255.0
        st.session_state.img_buffer.push(arr)

        model = get_model()
        sched = st.session_state.scheduler
        sched.enqueue(fn=lambda t=img_tensor: model.predict_single(t),
                      name="classify", priority=Priority.HIGH)
        task = sched.run_next()
        class_idx, confidence, infer_ms, all_probs = task.result

        # Full 5G pipeline decision
        decision = st.session_state.policy.decide(class_idx, confidence, all_probs)
        st.session_state.sw.push(infer_ms)
        st.session_state.history.append({**decision, "infer_ms": infer_ms})

        st.divider()
        # ── Metrics row
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Predicted Class",   CLASS_NAMES[class_idx])
        c2.metric("Confidence",        f"{confidence:.1%}")
        c3.metric("Inference",         f"{infer_ms:.1f} ms")
        c4.metric("5G Slice",          decision["slice"])
        c5.metric("Decision",          decision["decision"])

        st.divider()
        col_img, col_probs = st.columns([1,2])
        with col_img:
            st.markdown('<div class="sec">Input Image (64×64)</div>', unsafe_allow_html=True)
            st.image(img_pil.resize((256,256)), width=256)
            slice_col = SLICE_COLORS.get(decision["slice"], BLUE)
            st.markdown(f'<div class="card" style="border-left-color:{slice_col}">'
                        f'<b style="color:#e2e8f0">{CLASS_NAMES[class_idx]}</b><br>'
                        f'Confidence: {confidence:.1%}<br>'
                        f'Slice: {decision["slice"]}<br>'
                        f'QCI: {decision["qci"]}</div>', unsafe_allow_html=True)

        with col_probs:
            st.markdown('<div class="sec">Class Probabilities</div>', unsafe_allow_html=True)
            fig_b = go.Figure()
            fig_b.add_trace(go.Bar(x=all_probs.tolist(), y=CLASS_NAMES, orientation="h",
                marker_color=CLASS_COLORS, text=[f"{p:.1%}" for p in all_probs],
                textposition="outside"))
            fig_b.update_xaxes(title_text="Probability", range=[0,1.15])
            sfig(fig_b, "SatNet softmax output", height=320)
            st.plotly_chart(fig_b, use_container_width=True, config={"displayModeBar":False})

        st.divider()
        # ── 5G decision breakdown
        st.markdown('<div class="sec">5G Offloading Decision Breakdown</div>', unsafe_allow_html=True)
        d1,d2,d3,d4 = st.columns(4)
        d1.markdown(f'<div class="card">Channel (CQI)<br><br>'
                    f'CQI: {decision["cqi"]}<br>'
                    f'Condition: {decision["cqi_condition"]}<br>'
                    f'Modulation: {decision["modulation"]}</div>', unsafe_allow_html=True)
        d2.markdown(f'<div class="card" style="border-left-color:{SLICE_COLORS.get(decision["slice"],BLUE)}">5G Slice<br><br>'
                    f'Slice: {decision["slice"]}<br>'
                    f'QCI: {decision["qci"]}<br>'
                    f'Delay budget: {decision["qci_delay_budget"]} ms</div>', unsafe_allow_html=True)
        d3.markdown(f'<div class="card">MEC Routing<br><br>'
                    f'Node: {decision["node"]}<br>'
                    f'Path: {decision["path"]}<br>'
                    f'Link: {decision["link_ms"]} ms</div>', unsafe_allow_html=True)
        d4.markdown(f'<div class="card-green">Semantic Payload<br><br>'
                    f'{decision["payload_bytes"]} bytes<br>'
                    f'{decision["payload_content"]}<br>'
                    f'BW saved: {decision["bw_saving_pct"]}%</div>', unsafe_allow_html=True)

        st.markdown('<div class="sec">Routing Reason</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card">{decision["routing_reason"]}<br>'
                    f'Slice reason: {decision["slice_reason"]}</div>', unsafe_allow_html=True)

    if st.session_state.history:
        st.divider()
        st.markdown('<div class="sec">Session History</div>', unsafe_allow_html=True)
        for i, h in enumerate(reversed(st.session_state.history[-5:])):
            n = len(st.session_state.history)-i
            st.markdown(f'<div class="badge">#{n}</div>'
                        f'<div class="badge">{h["class_name"]}</div>'
                        f'<div class="badge">{h["confidence"]:.0%}</div>'
                        f'<div class="badge">Slice:{h["slice"]}</div>'
                        f'<div class="badge">{h["decision"]}</div>'
                        f'<div class="badge">{h["payload_bytes"]}B</div>'
                        f'<div class="badge">{h["bw_saving_pct"]}% BW saved</div>',
                        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — 5G NETWORK SLICING
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("## 5G Network Slicing — Semantic-Aware Slice Selection")
    st.markdown('<div style="color:#4a5568;font-size:13px;margin-bottom:12px;">The predicted EuroSAT class drives 5G slice selection. Disaster-risk classes (Forest=wildfire, River/SeaLake=flood, Highway/Industrial=change) → URLLC. Routine classes → eMBB. Background tasks → mMTC. This closed loop between AI output and network resources is the core 6G novelty.</div>', unsafe_allow_html=True)

    # Slice config cards
    for sname, sp in SLICE_CONFIG.items():
        col_ = SLICE_COLORS.get(sname, BLUE)
        st.markdown(f'<div class="card" style="border-left-color:{col_}">'
                    f'<b style="color:#e2e8f0">{sname}</b> — {sp.description}<br>'
                    f'Latency budget: {sp.latency_ms} ms &nbsp;·&nbsp; '
                    f'Reliability: {sp.reliability*100:.3f}% &nbsp;·&nbsp; '
                    f'Max payload: {sp.max_payload_B} bytes &nbsp;·&nbsp; '
                    f'Bandwidth: {sp.bandwidth_Mbps} Mbps &nbsp;·&nbsp; '
                    f'Priority: {sp.priority}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="sec">Run 5G Slice Simulation</div>', unsafe_allow_html=True)
    n_req = st.slider("Requests to simulate", 20, 300, 100)

    if st.button("Run 5G Slice Simulation"):
        slicer  = SliceSimulator()
        ch      = st.session_state.channel
        encoder = st.session_state.encoder
        slice_log, bw_log, lat_log = [], [], []

        for i in range(n_req):
            cls   = i % 10
            cname = CLASS_NAMES[cls]
            sd    = select_slice(cls, cname)
            state = ch.sample()
            sem   = encoder.encode(cls, 0.85, None, state.cqi)
            tx    = slicer.transmit(sd, sem["bytes"])
            slice_log.append(sd.slice_name)
            bw_log.append(sem["bw_saving_pct"])
            lat_log.append(tx["total_ms"])

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("URLLC requests",  f"{slice_log.count('URLLC')}/{n_req}")
        m2.metric("eMBB requests",   f"{slice_log.count('eMBB')}/{n_req}")
        m3.metric("Avg BW saved",    f"{np.mean(bw_log):.1f}%")
        m4.metric("Avg latency",     f"{np.mean(lat_log):.1f} ms")

        fig_s = make_subplots(rows=1, cols=2,
            subplot_titles=["Slice distribution", "Latency by slice (ms)"])
        sc = {"URLLC":slice_log.count("URLLC"),"eMBB":slice_log.count("eMBB"),"mMTC":slice_log.count("mMTC")}
        fig_s.add_trace(go.Bar(x=list(sc.keys()), y=list(sc.values()),
            marker_color=[RED,BLUE,AMBER]), row=1,col=1)

        for sn, col_ in [("URLLC",RED),("eMBB",BLUE),("mMTC",AMBER)]:
            lats = [lat_log[i] for i,s in enumerate(slice_log) if s==sn]
            if lats:
                fig_s.add_trace(go.Box(y=lats, name=sn, marker_color=col_,
                    line_color=col_), row=1,col=2)
        fig_s.update_layout(paper_bgcolor=BG, plot_bgcolor=BG2,
            font=dict(family="IBM Plex Mono",color=TEXT,size=11),
            showlegend=False, margin=dict(l=8,r=8,t=40,b=8), height=300)
        fig_s.update_xaxes(gridcolor=GRID,color=MUTE,zeroline=False)
        fig_s.update_yaxes(gridcolor=GRID,color=MUTE,zeroline=False)
        st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar":False})

# ══════════════════════════════════════════════════════════════
# TAB 3 — MEC ROUTING (Dijkstra)
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("## MEC Topology — Dijkstra Routing")
    st.markdown('<div style="color:#4a5568;font-size:13px;margin-bottom:12px;">A weighted directed graph of MEC nodes. Dijkstra finds the minimum-latency path from the Ground Station to any processing node. Node loads (CPU %) are added to edge weights dynamically, so a heavily loaded MEC is avoided even if topologically closer.</div>', unsafe_allow_html=True)

    topo = st.session_state.topology
    topo_data = topo.get_topology_data()
    loads = topo.get_node_loads()

    # Node load bars
    st.markdown('<div class="sec">Current Node Loads</div>', unsafe_allow_html=True)
    cols_ = st.columns(5)
    for i, node in enumerate(topo_data["nodes"]):
        with cols_[i]:
            load = loads.get(node["id"], 0)
            color = GREEN if load < 40 else (AMBER if load < 70 else RED)
            cols_[i].metric(node["id"], f"{load:.0f}%",
                             help=f"{node['location']} — {node['type']}")

    st.divider()
    st.markdown('<div class="sec">Dijkstra Shortest Paths from GS</div>', unsafe_allow_html=True)
    if st.button("Run Dijkstra from GS"):
        dist, prev = topo.dijkstra("GS")
        for target in ["MEC_A","MEC_B","MEC_C","CLOUD"]:
            lat, path = topo.shortest_path("GS", target)
            n_type = [n for n in topo_data["nodes"] if n["id"]==target][0]["type"]
            card_c = GREEN if n_type == "mec" else RED
            st.markdown(f'<div class="card" style="border-left-color:{card_c}">'
                        f'<b style="color:#e2e8f0">GS → {target}</b> &nbsp;·&nbsp; '
                        f'{lat:.1f} ms total &nbsp;·&nbsp; '
                        f'Path: {" → ".join(path)}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="sec">Dynamic Routing Simulation (50 requests)</div>', unsafe_allow_html=True)
    if st.button("Simulate 50 Routing Decisions"):
        node_counts = {"GS":0,"MEC_A":0,"MEC_B":0,"MEC_C":0,"CLOUD":0}
        latencies   = []
        for i in range(50):
            conf  = float(np.random.uniform(0.5, 1.0))
            cls   = i % 10
            cname = CLASS_NAMES[cls]
            sd    = select_slice(cls, cname)
            state = st.session_state.channel.sample()
            r = topo.select_processing_node(conf, sd.slice_name, state.cqi)
            node_counts[r["node"]] = node_counts.get(r["node"],0) + 1
            latencies.append(r["total_ms"])

        m1,m2,m3 = st.columns(3)
        m1.metric("Avg total latency", f"{np.mean(latencies):.1f} ms")
        m2.metric("P95 latency",       f"{np.percentile(latencies,95):.1f} ms")
        m3.metric("Most used node",    max(node_counts, key=node_counts.get))

        fig_r = go.Figure()
        fig_r.add_trace(go.Bar(
            x=list(node_counts.keys()), y=list(node_counts.values()),
            marker_color=[GREEN if k.startswith("MEC") else (BLUE if k=="GS" else RED)
                          for k in node_counts]))
        fig_r.update_xaxes(title_text="Node")
        fig_r.update_yaxes(title_text="Requests routed")
        sfig(fig_r, "Routing decisions across MEC topology", height=260)
        st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar":False})

        log = topo.get_routing_log(n=10)
        st.markdown('<div class="sec">Last 10 Routing Decisions</div>', unsafe_allow_html=True)
        for r in reversed(log):
            col_ = GREEN if r["node"].startswith("MEC") else (BLUE if r["node"]=="GS" else RED)
            st.markdown(f'<div class="card" style="border-left-color:{col_}">'
                        f'conf={r["confidence"]:.2f} | CQI={r["cqi"]} | slice={r["slice"]} | '
                        f'→ {r["node"]} ({r["total_ms"]}ms) | {r["reason"]}</div>',
                        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 4 — CHANNEL QUALITY (CQI)
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("## Channel Quality Indicator (CQI) — Adaptive Semantic Depth")
    st.markdown('<div style="color:#4a5568;font-size:13px;margin-bottom:12px;">Based on 3GPP TS 36.213. CQI 1–15 maps to modulation scheme and spectral efficiency. The CQI level determines how much semantic information can be reliably transmitted — from 1 byte (class index only) to 32 bytes (class + confidence + top-3 + geo-hash). This CQI-adaptive semantic depth is the novel link between the physical 5G channel and the AI output.</div>', unsafe_allow_html=True)

    # CQI table
    st.markdown('<div class="sec">3GPP CQI Table (TS 36.213)</div>', unsafe_allow_html=True)
    rows_cqi = []
    for cqi, info in CQI_TABLE.items():
        from network.channel import cqi_to_semantic_level
        lvl, byt, cont = cqi_to_semantic_level(cqi)
        rows_cqi.append({"CQI":cqi,"SINR(dB)":f"≥{info['sinr_min']}",
                          "Modulation":info["modulation"],
                          "Efficiency":info["efficiency"],
                          "Semantic payload":f"{byt}B — {cont}"})

    fig_cqi_table = go.Figure(data=[go.Table(
        header=dict(values=["CQI","SINR (dB)","Modulation","Efficiency (bps/Hz)","Semantic Payload"],
                    fill_color="#1e2230", font=dict(color="#e2e8f0",size=11,family="IBM Plex Mono"),
                    line_color="#2d3748", align="center"),
        cells=dict(values=[[r["CQI"] for r in rows_cqi],
                            [r["SINR(dB)"] for r in rows_cqi],
                            [r["Modulation"] for r in rows_cqi],
                            [r["Efficiency"] for r in rows_cqi],
                            [r["Semantic payload"] for r in rows_cqi]],
                   fill_color="#161b27", font=dict(color=TEXT,size=10,family="IBM Plex Mono"),
                   line_color="#1e2230", align="center")
    )])
    fig_cqi_table.update_layout(paper_bgcolor=BG, margin=dict(l=0,r=0,t=8,b=8), height=420)
    st.plotly_chart(fig_cqi_table, use_container_width=True, config={"displayModeBar":False})

    st.divider()
    st.markdown('<div class="sec">Simulate LEO Satellite Pass (Channel Variation)</div>', unsafe_allow_html=True)
    n_pass = st.slider("Pass length (samples)", 20, 200, 80)
    if st.button("Simulate Satellite Pass"):
        ch  = st.session_state.channel
        enc = st.session_state.encoder
        states = ch.simulate_pass(n_pass)
        cqis   = [s.cqi for s in states]
        conds  = [s.condition for s in states]
        bw_sav = [enc.encode(0,0.9,None,s.cqi)["bytes"] for s in states]

        fig_ch = make_subplots(rows=2, cols=1,
            subplot_titles=["CQI over LEO pass", "Semantic payload bytes (lower = more savings)"],
            shared_xaxes=True)
        cond_color = {"poor":RED,"moderate":AMBER,"good":GREEN,"excellent":BLUE}
        colors_list = [cond_color.get(c, BLUE) for c in conds]
        fig_ch.add_trace(go.Scatter(x=list(range(n_pass)), y=cqis,
            mode="lines+markers", line=dict(color=BLUE,width=2),
            marker=dict(color=colors_list,size=6)), row=1,col=1)
        fig_ch.add_hline(y=7,  line_color=GREEN, line_dash="dot", row=1,col=1,
                          annotation_text="eMBB threshold")
        fig_ch.add_trace(go.Scatter(x=list(range(n_pass)), y=bw_sav,
            fill="tozeroy", line=dict(color=AMBER,width=1.5),
            fillcolor="rgba(245,158,11,0.15)"), row=2,col=1)
        fig_ch.update_layout(paper_bgcolor=BG, plot_bgcolor=BG2,
            font=dict(family="IBM Plex Mono",color=TEXT,size=11),
            showlegend=False, margin=dict(l=8,r=8,t=40,b=8), height=380)
        fig_ch.update_xaxes(gridcolor=GRID,color=MUTE,zeroline=False)
        fig_ch.update_yaxes(gridcolor=GRID,color=MUTE,zeroline=False)
        st.plotly_chart(fig_ch, use_container_width=True, config={"displayModeBar":False})

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Mean CQI",    f"{np.mean(cqis):.1f}")
        m2.metric("Min CQI",     f"{min(cqis)}")
        m3.metric("Max CQI",     f"{max(cqis)}")
        m4.metric("Avg payload", f"{np.mean(bw_sav):.0f} B")

# ══════════════════════════════════════════════════════════════
# TAB 4 — 6G SEMANTIC COMMUNICATION
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("## 6G Semantic Communication")
    st.markdown('<div style="color:#4a5568;font-size:13px;margin-bottom:12px;">6G moves beyond 5G\'s QoS (bits/latency) to Goal-Oriented Communication and Semantic Information Value — the network asks not "did the bits arrive?" but "did the receiver achieve its goal?" These are core IMT-2030 (6G) concepts implemented here on real satellite data for the first time.</div>', unsafe_allow_html=True)

    # Explain the three 6G concepts
    st.markdown('<div class="sec">Three 6G Concepts Implemented</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="card" style="border-left-color:#a855f7">
        <b style="color:#e2e8f0">Goal-Oriented Comms</b><br><br>
        3GPP TR 22.874<br><br>
        Not "did the bits arrive?"<br>
        But "was the goal achieved?"<br><br>
        Goal: detect disasters ≥95% accuracy in ≤50ms<br>
        System tracks success rate per goal
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card" style="border-left-color:#06b6d4">
        <b style="color:#e2e8f0">Semantic Info Value (SIV)</b><br><br>
        Qin et al. arXiv:2112.10125<br><br>
        SIV = confidence × (1 − P_prior)<br><br>
        If we already know the area is Forest,<br>
        the 10th "Forest" result adds nothing.<br>
        Low SIV → skip the transmission
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="card" style="border-left-color:#10b981">
        <b style="color:#e2e8f0">Semantic Similarity Filter</b><br><br>
        Temporal semantic compression<br><br>
        If last 3 results were all "Forest",<br>
        the 4th "Forest" is redundant.<br>
        Filter it out — save bandwidth<br>
        without losing any new information
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Goal-Oriented Simulation
    st.markdown('<div class="sec">Goal-Oriented Communication — Simulation</div>', unsafe_allow_html=True)
    n_goal = st.slider("Number of images to evaluate", 20, 200, 80, key="n_goal")
    if st.button("Run Goal-Oriented Simulation"):
        gc   = st.session_state.goal_ctrl
        np.random.seed(42)
        classes     = np.random.choice(list(range(10)), size=n_goal)
        confidences = np.random.uniform(0.60, 0.99, n_goal)
        latencies   = np.random.uniform(10,  600, n_goal)

        achieved_log, action_log = [], []
        for cls, conf, lat in zip(classes, confidences, latencies):
            r = gc.evaluate(int(cls), float(conf), float(lat))
            achieved_log.append(int(r["achieved"]))
            action_log.append(r["action"])

        goal_stats = gc.get_goal_stats()
        disaster   = goal_stats.get("disaster_detection",    {})
        routine    = goal_stats.get("land_use_monitoring",   {})

        g1,g2,g3,g4 = st.columns(4)
        g1.metric("Disaster goal success",  f'{disaster.get("success_rate","—")}%')
        g2.metric("Disaster avg delay",     f'{disaster.get("avg_delay_ms","—")} ms')
        g3.metric("Routine goal success",   f'{routine.get("success_rate","—")}%')
        g4.metric("Routine avg delay",      f'{routine.get("avg_delay_ms","—")} ms')

        # Actions pie
        action_counts = {a: action_log.count(a) for a in ["ACCEPT","ESCALATE","RETRANSMIT","ALERT"]}
        fig_goal = go.Figure(go.Pie(
            labels=list(action_counts.keys()),
            values=list(action_counts.values()),
            marker_colors=[GREEN, RED, AMBER, BLUE],
            hole=0.4,
        ))
        fig_goal.update_layout(paper_bgcolor=BG,
            font=dict(family="IBM Plex Mono",color=TEXT,size=11),
            margin=dict(l=8,r=8,t=36,b=8), height=260,
            title=dict(text="Goal-oriented action distribution",
                       font=dict(color="#c5cedd",size=12),x=0))
        st.plotly_chart(fig_goal, use_container_width=True, config={"displayModeBar":False})

        # Goal log
        st.markdown('<div class="sec">Last 8 Goal Evaluations</div>', unsafe_allow_html=True)
        for r in list(gc.get_log())[-8:]:
            col_ = GREEN if r["achieved"] else (RED if r["action"]=="ESCALATE" else AMBER)
            st.markdown(
                f'<div class="card" style="border-left-color:{col_}">'
                f'{r["class_name"]} | conf={r["confidence"]:.0%} | '
                f'lat={r["latency_ms"]:.0f}ms | goal={r["goal_id"]} | '
                f'<b style="color:#e2e8f0">{r["action"]}</b> — {r["reason"]}'
                f'</div>', unsafe_allow_html=True)

    st.divider()

    # ── Semantic Information Value
    st.markdown('<div class="sec">Semantic Information Value (SIV) — Redundancy Elimination</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">SIV = confidence × (1 − P_prior) where P_prior = fraction of recent history that was the same class. A satellite flying over a forest sends "Forest" repeatedly — after the first few, the remaining transmissions have near-zero SIV. The system defers them, saving bandwidth with no information loss.</div>', unsafe_allow_html=True)

    if st.button("Run SIV Simulation (100 images, mostly forest area)"):
        siv_engine = SemanticInformationValue(history_size=20, defer_threshold=0.15)
        results_siv = siv_engine.simulate_batch(100)
        stats_siv   = siv_engine.get_stats()

        sv1,sv2,sv3 = st.columns(3)
        sv1.metric("Transmitted",  stats_siv["transmitted"])
        sv2.metric("Deferred",     stats_siv["deferred"])
        sv3.metric("Extra BW saved (on top of semantic comms)", f'{stats_siv["defer_rate"]}%')

        # SIV over time
        siv_vals  = [r["siv"]  for r in results_siv]
        actions   = [r["action"] for r in results_siv]
        colors_sv = [GREEN if a=="TRANSMIT" else RED for a in actions]

        fig_siv = go.Figure()
        fig_siv.add_trace(go.Scatter(
            x=list(range(len(siv_vals))), y=siv_vals,
            mode="lines+markers",
            line=dict(color=BLUE, width=1.5),
            marker=dict(color=colors_sv, size=7),
            name="SIV"
        ))
        fig_siv.add_hline(y=0.15, line_color=AMBER, line_dash="dash",
                          annotation_text="defer threshold",
                          annotation_font_color=AMBER)
        fig_siv.update_xaxes(title_text="Image index")
        fig_siv.update_yaxes(title_text="Semantic Information Value")
        sfig(fig_siv, "SIV over satellite pass — green=transmit, red=deferred", height=280)
        st.plotly_chart(fig_siv, use_container_width=True, config={"displayModeBar":False})

        st.markdown(f'<div class="card-green">Result: {stats_siv["defer_rate"]}% of transmissions eliminated as semantically redundant. Combined with semantic payload encoding (99.2% BW saving), total bandwidth reduction exceeds 99.5%.</div>', unsafe_allow_html=True)

    st.divider()

    # ── Semantic Similarity Filter
    st.markdown('<div class="sec">Semantic Similarity Filter — Temporal Compression</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">If the same class is predicted K times in a row, subsequent identical results are filtered as redundant. Critical classes (wildfire/flood) are always passed through regardless.</div>', unsafe_allow_html=True)

    repeat_k = st.slider("Repeat threshold K", 2, 6, 3, key="repeat_k")
    n_filter  = st.slider("Images to filter", 20, 100, 50, key="n_filter")
    if st.button("Run Similarity Filter"):
        ssf = SemanticSimilarityFilter(window=5, repeat_threshold=repeat_k)
        np.random.seed(7)
        # Simulate slow-moving satellite over homogeneous terrain
        classes_f = np.random.choice([1,1,1,0,2,2,3,8], size=n_filter)
        filter_log = [ssf.filter(int(c), 0.9) for c in classes_f]
        stats_f = ssf.get_stats()

        f1,f2,f3 = st.columns(3)
        f1.metric("Passed",      stats_f["passed"])
        f2.metric("Filtered",    stats_f["filtered"])
        f3.metric("Filter rate", f'{stats_f["filter_rate"]}%')

        pass_vals   = [1 if r["action"]=="PASS" else 0 for r in filter_log]
        class_names_ = [r["class_name"][:6] for r in filter_log]
        colors_f = [GREEN if p else RED for p in pass_vals]

        fig_f = go.Figure()
        fig_f.add_trace(go.Bar(
            x=list(range(n_filter)), y=pass_vals,
            marker_color=colors_f,
            text=class_names_, textposition="outside",
        ))
        fig_f.update_xaxes(title_text="Image index")
        fig_f.update_yaxes(title_text="Transmitted (1) / Filtered (0)", range=[-0.2,1.5])
        sfig(fig_f, "Green = transmitted, Red = filtered as redundant", height=260)
        st.plotly_chart(fig_f, use_container_width=True, config={"displayModeBar":False})

    st.divider()
    st.markdown('<div class="sec">5G vs 6G — What is Different</div>', unsafe_allow_html=True)
    rows_cmp = [
        ["What the network cares about",   "Latency, throughput, reliability",     "Did the receiver achieve its goal?"],
        ["Transmission decision",          "Always transmit if QoS is met",        "Only transmit if SIV is high enough"],
        ["Resource allocation",            "Based on QoS Class (QCI)",             "Based on semantic content + goal"],
        ["Redundancy handling",            "None — every packet forwarded",         "Semantic similarity filter skips repeats"],
        ["Payload size",                   "Fixed (full image or nothing)",         "CQI-adaptive: 1 byte to 32 bytes"],
        ["Standard reference",             "3GPP Release 15/16 (5G NR)",            "IMT-2030, 3GPP TR 22.874 (6G)"],
    ]
    fig_cmp = go.Figure(data=[go.Table(
        header=dict(
            values=["Aspect","5G Approach","6G Approach (this system)"],
            fill_color="#1e2230",
            font=dict(color="#e2e8f0", size=11, family="IBM Plex Mono"),
            line_color="#2d3748", align="left"),
        cells=dict(
            values=[[r[0] for r in rows_cmp],
                    [r[1] for r in rows_cmp],
                    [r[2] for r in rows_cmp]],
            fill_color=["#161b27","#161b27","#0f1a0f"],
            font=dict(color=[TEXT, TEXT, "#86efac"], size=10, family="IBM Plex Mono"),
            line_color="#1e2230", align="left")
    )])
    fig_cmp.update_layout(paper_bgcolor=BG, margin=dict(l=0,r=0,t=8,b=8), height=280)
    st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar":False})

# ══════════════════════════════════════════════════════════════
# TAB 5 — DSA VISUALIZER
# ══════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("## DSA Structure Visualizer")
    choice = st.selectbox("Select structure", [
        "Image Ring Buffer","Min-Heap Priority Queue",
        "Sliding Window + Monotonic Deque","Dijkstra on MEC Graph"])
    st.divider()

    if choice == "Image Ring Buffer":
        st.markdown("### Image Ring Buffer — O(1) Push/Pop")
        st.markdown('<div class="card">Fixed-capacity array. HEAD = write pointer, TAIL = read pointer. Both advance with modulo — wrap around at end. Oldest image overwritten when full. Zero memory allocation after startup.</div>', unsafe_allow_html=True)
        n_push = st.slider("Images pushed", 5, 60, 30)
        cap=50; size=min(n_push,cap); head=n_push%cap; tail=(head-size)%cap
        filled=[(tail+i)%cap for i in range(size)]
        colors_=[BLUE if i in filled else BG2 for i in range(cap)]
        fig_rb=go.Figure()
        fig_rb.add_trace(go.Bar(x=list(range(cap)),y=[1]*cap,marker_color=colors_,showlegend=False))
        fig_rb.add_vline(x=head,line_color=AMBER,line_dash="dash",line_width=2,
                         annotation_text="HEAD (write)",annotation_font_color=AMBER)
        fig_rb.add_vline(x=tail,line_color=GREEN,line_dash="dash",line_width=2,
                         annotation_text="TAIL (read)",annotation_font_color=GREEN)
        fig_rb.update_yaxes(showticklabels=False)
        fig_rb.update_xaxes(title_text="Buffer slot index")
        sfig(fig_rb,f"Ring buffer — {size}/{cap} slots occupied",height=220)
        st.plotly_chart(fig_rb,use_container_width=True,config={"displayModeBar":False})
        r1,r2,r3=st.columns(3)
        r1.metric("Occupied",f"{size}/{cap}"); r2.metric("HEAD",head); r3.metric("Fill",f"{size/cap:.0%}")

    elif choice == "Min-Heap Priority Queue":
        st.markdown("### Min-Heap Priority Queue — O(log n)")
        st.markdown('<div class="card">Binary tree where every parent ≤ its children. Most urgent task always at root — O(1) peek. O(log n) insert/remove. CRITICAL-class detections (wildfire/flood) always preempt routine land-use classification.</div>', unsafe_allow_html=True)
        sd = PriorityScheduler("vis")
        for n,p in [("archive",Priority.BACKGROUND),("classify_land",Priority.NORMAL),
                    ("detect_wildfire",Priority.CRITICAL),("measure_ndvi",Priority.HIGH),
                    ("log_result",Priority.LOW),("detect_flood",Priority.CRITICAL),
                    ("compress",Priority.BACKGROUND)]:
            sd.enqueue(fn=lambda:None, name=n, priority=p)
        exe=[]
        while not sd.is_empty:
            t=sd.run_next()
            if t: exe.append((t.priority,t.name))
        pris=[e[0] for e in exe]; names=[e[1] for e in exe]
        pcols={0:RED,1:AMBER,2:BLUE,3:GREEN,4:MUTE}
        fig_hp=go.Figure()
        fig_hp.add_trace(go.Bar(x=list(range(len(names))),y=[4-p for p in pris],
            marker_color=[pcols[p] for p in pris],text=names,textposition="outside"))
        fig_hp.update_layout(yaxis=dict(tickvals=[0,1,2,3,4],
            ticktext=["Background","Low","Normal","High","Critical"]),
            xaxis_title="Execution order (left = first)")
        sfig(fig_hp,"Min-heap — critical tasks always run first",height=280)
        st.plotly_chart(fig_hp,use_container_width=True,config={"displayModeBar":False})

    elif choice == "Sliding Window + Monotonic Deque":
        st.markdown("### Sliding Window + Monotonic Deque — O(1)")
        st.markdown('<div class="card">Running sum gives O(1) mean per sample. Monotonic deque gives O(1) window max/min — maintains sorted candidates, each element enters and exits once. Used to track rolling inference latency in real time.</div>', unsafe_allow_html=True)
        ws=st.slider("Window size",5,50,20); np2=st.slider("Images",30,150,80)
        sw2=SlidingWindow(ws); np.random.seed(42)
        lats=20+10*np.abs(np.sin(np.linspace(0,4*np.pi,np2)))+5*np.random.randn(np2)
        ms_,mxs,mns=[],[],[]
        for v in lats:
            sw2.push(float(v)); ms_.append(sw2.mean); mxs.append(sw2.window_max); mns.append(sw2.window_min)
        fig_sw=go.Figure()
        fig_sw.add_trace(go.Scatter(y=lats.tolist(),name="Raw latency",line=dict(color=MUTE,width=1)))
        fig_sw.add_trace(go.Scatter(y=ms_,name=f"Rolling mean O(1)",line=dict(color=BLUE,width=2)))
        fig_sw.add_trace(go.Scatter(y=mxs,name="Window max (deque)",line=dict(color=AMBER,width=1.5,dash="dash")))
        fig_sw.add_trace(go.Scatter(y=mns,name="Window min (deque)",line=dict(color=GREEN,width=1.5,dash="dot")))
        fig_sw.update_xaxes(title_text="Image index"); fig_sw.update_yaxes(title_text="Latency (ms)")
        sfig(fig_sw,f"Sliding window size={ws} — all stats in O(1)",height=300)
        st.plotly_chart(fig_sw,use_container_width=True,config={"displayModeBar":False})
        s1,s2,s3=st.columns(3)
        s1.metric("Mean",f"{sw2.mean:.2f} ms"); s2.metric("Max",f"{sw2.window_max:.2f} ms"); s3.metric("Min",f"{sw2.window_min:.2f} ms")

    else:
        st.markdown("### Dijkstra on MEC Graph — O((V+E) log V)")
        st.markdown('<div class="card">MEC nodes modeled as a weighted directed graph. Edge weights = base link latency + load penalty on destination node. Dijkstra finds minimum-latency path. Runs in microseconds (5 nodes, 8 edges).</div>', unsafe_allow_html=True)
        topo2 = MECTopology()
        dist, _ = topo2.dijkstra("GS")
        nodes_ = ["MEC_A","MEC_B","MEC_C","CLOUD"]
        lats_  = [dist[n] for n in nodes_]
        cols_  = [GREEN,GREEN,GREEN,RED]
        fig_d = go.Figure()
        fig_d.add_trace(go.Bar(x=nodes_,y=lats_,marker_color=cols_,
            text=[f"{v:.1f}ms" for v in lats_],textposition="outside"))
        fig_d.update_xaxes(title_text="Destination node")
        fig_d.update_yaxes(title_text="Dijkstra distance from GS (ms)")
        sfig(fig_d,"Shortest path latency from Ground Station",height=260)
        st.plotly_chart(fig_d,use_container_width=True,config={"displayModeBar":False})
        for n,lat in zip(nodes_,lats_):
            _,path = topo2.shortest_path("GS",n)
            st.markdown(f'<span class="badge">GS → {n}: {lat:.1f}ms via {" → ".join(path)}</span>',
                        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 6 — BENCHMARKS
# ══════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("## Performance Benchmarks")
    st.markdown('<div style="color:#4a5568;font-size:13px;margin-bottom:12px;">Empirical timing of every DSA operation, CNN inference, Dijkstra routing, and 5G slice simulation. Confirms all complexity claims.</div>', unsafe_allow_html=True)

    existing_b = load_benchmark_results()
    if existing_b:
        st.markdown('<div class="card-green">Results loaded from previous run.</div>', unsafe_allow_html=True)
    if st.button("Run All Benchmarks"):
        bar = st.progress(0, text="Starting...")
        def cb(lbl,pct): bar.progress(1.0 if pct==100 else pct/100, text=lbl)
        existing_b = run_all_benchmarks(progress_cb=cb)
        st.success("Done!")

    if existing_b:
        st.divider()
        st.markdown('<div class="sec">DSA Structure Timings</div>', unsafe_allow_html=True)
        b1,b2,b3 = st.columns(3)
        rb_=existing_b.get("ring_buffer",{})
        b1.markdown(f'<div class="card"><b style="color:#e2e8f0">Ring Buffer</b> — O(1)<br><br>'
                    f'Push: {rb_.get("push_us","—")} µs<br>'
                    f'Pop: {rb_.get("pop_us","—")} µs<br>'
                    f'Overflows: {rb_.get("overflows","—")}</div>', unsafe_allow_html=True)
        sw_=existing_b.get("sliding_window",{})
        b2.markdown(f'<div class="card"><b style="color:#e2e8f0">Sliding Window</b> — O(1)<br><br>'
                    f'10k pushes: {sw_.get("ten_k_ms","—")} ms<br>'
                    f'Stats lookup: {sw_.get("stat_us","—")} µs</div>', unsafe_allow_html=True)
        pq_=existing_b.get("priority_queue",{})
        b3.markdown(f'<div class="card"><b style="color:#e2e8f0">Priority Queue</b> — O(log n)<br><br>'
                    f'Drain 7 tasks: {pq_.get("drain_ms","—")} ms<br>'
                    f'Heap correct: {"Yes ✓" if pq_.get("heap_correct") else "No"}</div>', unsafe_allow_html=True)

        st.markdown('<div class="sec">Inference + Routing</div>', unsafe_allow_html=True)
        i1,i2,i3,i4 = st.columns(4)
        inf_=existing_b.get("inference",{})
        dij_=existing_b.get("dijkstra",{})
        i1.metric("CNN mean",  f'{inf_.get("mean_ms","—")} ms')
        i2.metric("CNN P95",   f'{inf_.get("p95_ms","—")} ms')
        i3.metric("Dijkstra mean", f'{dij_.get("mean_ms","—")} ms')
        i4.metric("Dijkstra P95",  f'{dij_.get("p95_ms","—")} ms')

        st.markdown('<div class="sec">5G Network Simulation</div>', unsafe_allow_html=True)
        net_=existing_b.get("network_5g",{})
        n1,n2,n3 = st.columns(3)
        sc=net_.get("slice_counts",{})
        n1.metric("URLLC requests", sc.get("URLLC","—"))
        n2.metric("eMBB requests",  sc.get("eMBB","—"))
        n3.metric("Avg BW saved",   f'{net_.get("avg_bw_saving_pct","—")}%')

        pq_order = pq_.get("order",[])
        if pq_order:
            st.markdown('<div class="sec">Priority Queue Execution Order</div>', unsafe_allow_html=True)
            for item in pq_order:
                st.markdown(f'<span class="badge">{item}</span>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 7 — BASELINES
# ══════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown("## Baseline Model Comparison")
    st.markdown('<div style="color:#4a5568;font-size:13px;margin-bottom:12px;">SatNet vs standard edge-deployable models. Compares parameter count, inference latency, and edge deployability.</div>', unsafe_allow_html=True)

    existing_bl = load_baseline_results()
    if existing_bl:
        st.markdown('<div class="card-green">Results loaded from previous run.</div>', unsafe_allow_html=True)
    if st.button("Run Baseline Comparison"):
        with st.spinner("Measuring latency for all models (30 runs each)..."):
            existing_bl = run_baseline_comparison()
        st.success("Done!")

    if existing_bl:
        st.divider()
        names_  = [r["model"]    for r in existing_bl]
        params_ = [r["params_M"] for r in existing_bl]
        lats_   = [r["mean_ms"]  for r in existing_bl]
        p95_    = [r["p95_ms"]   for r in existing_bl]
        cols_bl = [BLUE if "SatNet" in n else MUTE for n in names_]

        fig_bl = make_subplots(rows=1, cols=2,
            subplot_titles=["Parameters (M) — smaller = better for edge",
                            "Mean inference latency (ms) — lower = better"])
        fig_bl.add_trace(go.Bar(x=names_,y=params_,marker_color=cols_bl,
            text=[f"{p:.1f}M" for p in params_],textposition="outside"), row=1,col=1)
        fig_bl.add_trace(go.Bar(x=names_,y=lats_,marker_color=cols_bl,
            text=[f"{l:.0f}ms" for l in lats_],textposition="outside"), row=1,col=2)
        fig_bl.update_layout(paper_bgcolor=BG,plot_bgcolor=BG2,
            font=dict(family="IBM Plex Mono",color=TEXT,size=11),
            showlegend=False,margin=dict(l=8,r=8,t=40,b=8),height=300)
        fig_bl.update_xaxes(gridcolor=GRID,color=MUTE,zeroline=False)
        fig_bl.update_yaxes(gridcolor=GRID,color=MUTE,zeroline=False)
        st.plotly_chart(fig_bl, use_container_width=True, config={"displayModeBar":False})

        for r in existing_bl:
            edge_str = "Yes ✓" if r["edge_ok"] else "No ✗"
            acc_str  = f"{r['val_acc']}%" if r.get("val_acc") else "—"
            hl = "card-green" if "SatNet" in r["model"] else "card"
            st.markdown(
                f'<div class="{hl}">'
                f'<b style="color:#e2e8f0">{r["model"]}</b> &nbsp;·&nbsp; '
                f'{r["params_M"]}M params &nbsp;·&nbsp; '
                f'{r["mean_ms"]}ms mean &nbsp;·&nbsp; '
                f'{r["p95_ms"]}ms P95 &nbsp;·&nbsp; '
                f'Acc: {acc_str} &nbsp;·&nbsp; Edge: {edge_str}'
                f'</div>', unsafe_allow_html=True)
