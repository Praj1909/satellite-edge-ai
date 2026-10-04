# 🛰️ Satellite Edge AI

### 5G/6G-Aware Satellite Vision and Edge Intelligence Pipeline

A Python-based research prototype that connects **satellite image classification** with **network-aware edge intelligence**.

The system uses a lightweight CNN, **SatNet**, to classify EuroSAT satellite imagery and then uses the predicted land-use class to drive a chain of communication and edge-computing decisions:

**Satellite Image → AI Classification → Network Slice → Channel/CQI → MEC Routing → Semantic Communication**

Rather than treating AI inference as the final output, this project explores how an AI decision can influence **communication resource allocation and edge processing**.

---

## 🚀 Why This Project?

Satellite systems can generate large volumes of imagery while operating under communication and computational constraints.

This project explores a simple question:

> **What if the result of an AI model could directly influence how satellite information is transported and processed across an edge network?**

The system therefore combines:

- 🧠 Deep learning for satellite image classification
- 📡 5G/6G-inspired network slicing
- 📶 CQI-based channel simulation
- 🌐 Multi-access Edge Computing (MEC)
- 🧭 Shortest-path routing using Dijkstra's algorithm
- 🗜️ Semantic communication and payload reduction
- ⚡ Data structures for time-sensitive edge processing
- 📊 Benchmarking and performance analysis
- 🖥️ An interactive Streamlit dashboard

This makes the repository more than a standalone image-classification model. It is an experimental **AI + communication + edge decision pipeline**.

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │   Satellite Image   │
                    │      EuroSAT        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      SatNet CNN     │
                    │ Image Classification│
                    └──────────┬──────────┘
                               │
                        Land-use Class
                               │
                               ▼
                 ┌──────────────────────────┐
                 │   Edge Decision Engine   │
                 └────────────┬─────────────┘
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
      ┌────────────┐   ┌────────────┐   ┌──────────────┐
      │ 5G Slice   │   │ CQI /      │   │  Semantic    │
      │ Selection   │   │ Channel    │   │ Communication│
      │             │   │ Simulation │   │              │
      └──────┬─────┘   └──────┬─────┘   └──────┬───────┘
             │                │                │
             └────────────────┼────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │     MEC Network     │
                    │  Dijkstra Routing   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Edge Processing /   │
                    │ Communication Path  │
                    └─────────────────────┘
```

---

# 🧠 Core AI Pipeline

## SatNet CNN

The project uses a lightweight convolutional neural network called **SatNet** for EuroSAT image classification.

The model:

- Is implemented using PyTorch
- Performs satellite land-use classification
- Supports 10 EuroSAT classes
- Saves the trained model as `models/satnet.pth`

### Classification Classes

```text
AnnualCrop
Forest
HerbaceousVegetation
Highway
Industrial
Pasture
PermanentCrop
Residential
River
SeaLake
```

The model implementation is located in:

```text
core/model.py
```

Training logic is implemented in:

```text
core/train.py
```

The main training entry point is:

```text
train_satnet.py
```

---

# 📡 Network-Aware Intelligence

The key idea of the project is that the AI prediction is not treated as the end of the pipeline.

The predicted land-use class is passed into a network decision layer.

The system then considers:

1. **Class importance**
2. **Network slice requirements**
3. **Channel quality**
4. **MEC routing**
5. **Semantic payload reduction**

This creates a connection between **machine learning inference and communication decisions**.

---

# 📶 5G Network Slicing

The project models three network slice types:

| Slice | Intended Use |
|---|---|
| **URLLC** | Sensitive or urgent information |
| **eMBB** | Normal / high-bandwidth traffic |
| **mMTC** | Background or non-critical tasks |

The selected slice is influenced by the classification result and the associated class importance.

Network slicing logic is implemented under:

```text
network/slicing.py
```

---

# 📊 CQI and Channel Simulation

The project includes a simplified communication-channel model using **Channel Quality Indicator (CQI)** concepts.

The channel simulation is implemented in:

```text
network/channel.py
```

The system uses simulated link quality to influence communication decisions within the edge pipeline.

This allows the project to explore how AI decisions can interact with changing communication conditions.

---

# 🌐 MEC Routing

The project models a Multi-access Edge Computing topology.

The MEC graph is implemented in:

```text
network/mec_graph.py
```

Routing uses **Dijkstra's shortest-path algorithm** to determine an appropriate path through the simulated MEC topology.

This provides the network layer with a routing decision after the AI inference stage.

---

# 🗜️ Semantic Communication

Instead of always treating the complete raw image as the communication payload, the project explores a semantic communication approach.

Relevant modules include:

```text
network/qos.py
network/semantic_6g.py
```

The system estimates semantic payload reduction compared with raw image transmission.

The goal is to demonstrate how communication can focus on the **information relevant to the downstream decision**, rather than blindly transmitting all available data.

---

# ⚡ Edge Processing and DSA

The project also includes custom data structures designed around edge-processing workflows.

Located in:

```text
utils/dsa.py
```

The utilities include:

- `RingBuffer`
- `SlidingWindow`
- `PriorityScheduler`

These are used to model buffering, time-windowed processing and priority-aware scheduling.

An image buffering utility is also provided:

```text
utils/image_buffer.py
```

---

# 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit application.

Launch it using:

```bash
streamlit run main.py
```

The dashboard provides functionality for:

- 📤 Satellite image upload
- 🧠 Live class prediction
- 📈 Prediction confidence
- 📡 5G slice recommendation
- 📶 Channel quality estimation
- 🌐 MEC routing path
- 🗜️ Semantic payload size
- 💾 Estimated bandwidth savings
- 📊 Benchmark views
- 📜 Historical decisions

Main application:

```text
main.py
```

---

# 📊 Benchmarking

A dedicated benchmark suite is included to evaluate different parts of the system.

Run:

```bash
python benchmarks/run_benchmark.py
```

The benchmark suite measures:

- Ring buffer performance
- Sliding-window statistics
- Priority queue scheduling
- CNN inference timing
- Dijkstra routing timing
- 5G slice simulation
- Bandwidth-saving behavior

Results are stored in:

```text
results/benchmarks.json
```

Additional baseline results are stored in:

```text
results/baselines.json
```

---

# 📁 Repository Structure

```text
satellite-edge-ai/
│
├── benchmarks/
│   └── run_benchmark.py
│
├── core/
│   ├── model.py
│   ├── train.py
│   ├── baselines.py
│   ├── hybrid_policy.py
│   └── ablation.py
│
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   └── README.md
│
├── experiments/
│
├── models/
│   ├── satnet.pth
│   └── train_log.json
│
├── network/
│   ├── channel.py
│   ├── mec_graph.py
│   ├── qos.py
│   ├── semantic_6g.py
│   └── slicing.py
│
├── results/
│   ├── baselines.json
│   └── benchmarks.json
│
├── utils/
│   ├── dsa.py
│   └── image_buffer.py
│
├── main.py
├── train_satnet.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/Praj1909/satellite-edge-ai.git
cd satellite-edge-ai
```

---

## 2. Create a virtual environment

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

If required:

```bash
pip install --upgrade pip
```

---

# 📦 Dataset Setup

This project uses the **EuroSAT dataset** for satellite image classification.

The raw dataset is intentionally **not included in this repository** because of its size.

The dataset loader is located at:

```text
data/dataset.py
```

The expected dataset location is:

```text
data/eurosat_data/eurosat/2750/
```

After downloading and extracting EuroSAT, place the dataset under:

```text
data/eurosat_data/
```

> The dataset is required for training the SatNet model.

---

# 🏋️ Training

The easiest way to train the model is:

```bash
python train_satnet.py
```

You can also specify training parameters:

```bash
python train_satnet.py --epochs 10 --lr 0.001 --batch-size 32
```

Alternatively, run the training module directly:

```bash
python core/train.py --epochs 10 --lr 0.001 --batch-size 32
```

The best trained model is saved to:

```text
models/satnet.pth
```

Training history is stored in:

```text
models/train_log.json
```

---

# ▶️ Running the Dashboard

After installing dependencies and preparing the dataset:

```bash
streamlit run main.py
```

Streamlit will provide a local browser address, typically:

```text
http://localhost:8501
```

You can then upload a satellite image and inspect the complete decision pipeline.

---

# 📊 Running Benchmarks

To benchmark the system:

```bash
python benchmarks/run_benchmark.py
```

Results are generated in:

```text
results/benchmarks.json
```

---

# 🔬 Project Components

| Component | Implementation |
|---|---|
| Satellite classification | PyTorch / SatNet CNN |
| Dataset | EuroSAT |
| Network slicing | Python simulation |
| Channel modeling | CQI-based simulation |
| MEC routing | Dijkstra shortest path |
| Semantic communication | Payload reduction / semantic logic |
| Edge scheduling | RingBuffer, SlidingWindow, PriorityScheduler |
| Dashboard | Streamlit |
| Benchmarking | Custom benchmark suite |

---

# 🛠️ Technologies

```text
Python
PyTorch
NumPy
Pandas
Scikit-learn
Streamlit
Deep Learning
Computer Vision
Edge AI
MEC
5G / 6G Networking
Network Slicing
QoS
Semantic Communication
Graph Algorithms
Data Structures
```

---

# 🎯 What This Project Demonstrates

The project demonstrates a complete experimental pipeline in which:

```text
AI
 ↓
understands the satellite image

Network intelligence
 ↓
determines how the information should be transported

MEC
 ↓
determines where edge processing should occur

Semantic communication
 ↓
reduces unnecessary information transfer

Scheduling / buffering
 ↓
supports time-sensitive edge processing
```

The overall concept can be summarized as:

> **AI decides what matters. The network decides how to transport it. Edge computing decides where to process it.**

---

# ⚠️ Project Scope

This repository is an **experimental research/demo system**, not a production telecom stack.

The 5G/6G components are simulations and conceptual implementations intended to demonstrate interactions between:

- AI inference
- communication conditions
- network slicing
- edge routing
- semantic information transfer
- resource-aware decision making

The project is therefore best viewed as a **research-oriented systems prototype**.

---

# 🔭 Potential Applications

The concepts explored in this project are relevant to areas such as:

- 🛰️ Satellite image intelligence
- 🤖 Edge AI
- 📡 5G / 6G communication
- 🌐 Network-aware machine learning
- 🧠 Distributed intelligence
- 🚀 Space and satellite systems
- 📶 Communication-efficient AI
- ⚡ Multi-access Edge Computing
- 🗜️ Semantic communication

---

# 📌 Key Takeaway

This project explores an end-to-end satellite edge intelligence pipeline rather than treating image classification as an isolated machine-learning task.

A satellite image is classified, the resulting information influences network decisions, communication conditions affect the simulated system, and the information is routed toward an edge-processing environment.

The project therefore combines:

**Computer Vision + Machine Learning + Communication Systems + Edge Computing + Algorithms**

into a single experimental platform.

---

# 👨‍💻 Author

**Prajwal Jathanna**

Electronics and Communication Engineering  
RV College of Engineering, Bengaluru

---

## ⭐ If you find this project interesting

Feel free to explore the implementation, benchmark results and individual modules in the repository.

```text
Satellite Vision
      +
     AI
      +
5G / 6G Networking
      +
     MEC
      +
Semantic Communication
      =
Satellite Edge Intelligence
```
