SATELLITE EDGE AI - 5G-AWARE SATELLITE VISION PIPELINE

This project is a full end-to-end satellite image classification and edge networking demo built in Python.
It combines:
- a lightweight CNN classifier called SatNet
- EuroSAT land-use classification
- 5G/6G-inspired network slicing decisions
- CQI-based channel modeling
- MEC routing using shortest-path logic
- semantic communication and bandwidth-saving logic
- a Streamlit dashboard for visual analysis and simulation

What this project is really doing:

A satellite image is classified using a CNN. Based on the detected land-use class, the system decides which 5G network slice to use (URLLC, eMBB, or mMTC), which MEC node to route traffic through, and how much semantic information can be compressed while still preserving the decision quality. This turns AI inference into a real-time edge-network orchestration decision.

In simple words:
This is not just a model training repo. It is a research-style demonstration of an AI + communication + edge decision pipeline for satellite systems.

---------------------------------------------------------------------
PROJECT GOAL
---------------------------------------------------------------------
The goal is to build and demonstrate a satellite edge intelligence system that:
- classifies remote sensing images
- picks a suitable wireless network slice based on the class importance
- sends semantic information instead of raw image data when possible
- routes traffic through a MEC topology
- simulates realistic 5G/6G communication conditions
- benchmarks performance and resource efficiency

This is especially relevant for future 6G and edge intelligence use cases where AI inference decisions must be connected to communication constraints.

---------------------------------------------------------------------
WHAT YOU BUILT HERE
---------------------------------------------------------------------
1. SatNet CNN
   - Model file: core/model.py
   - Lightweight convolutional neural network for EuroSAT classification
   - Trained on 10 land-use classes
   - Output classes include:
     AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial,
     Pasture, PermanentCrop, Residential, River, SeaLake

2. EuroSAT Dataset Integration
   - Dataset loader lives in data/dataset.py
   - The project expects the EuroSAT dataset under the folder:
     data/eurosat_data/eurosat/2750/

3. Training Pipeline
   - Script: train_satnet.py
   - Training module: core/train.py
   - Trains SatNet using PyTorch
   - Saves best model to models/satnet.pth

4. Streamlit Dashboard
   - Main app: main.py
   - Launches a web dashboard for:
     - image upload
     - live class prediction
     - confidence scores
     - 5G slice selection
     - channel quality estimation
     - MEC routing path
     - semantic payload size and bandwidth savings
     - benchmark views and historical decisions

5. 5G Network Slicing
   - Network logic lives under network/
   - The project models three slice types:
     - URLLC: for sensitive or urgent classes
     - eMBB: for normal/standard traffic
     - mMTC: for background or non-critical tasks
   - This is driven by the model output and class importance

6. CQI and Channel Simulation
   - network/channel.py contains channel behavior and CQI modeling
   - The app simulates link quality and maps it to communication decisions

7. MEC Routing
   - network/mec_graph.py models a MEC topology and uses Dijkstra shortest path routing
   - The system chooses the best edge node according to the situation

8. Semantic Communication
   - network/qos.py and network/semantic_6g.py
   - Semantic compression reduces transmitted payload size versus raw image data
   - This is part of the 6G-style communication reasoning in the project

9. DSA Utilities
   - utils/dsa.py contains data structures such as:
     - RingBuffer
     - SlidingWindow
     - PriorityScheduler
   - These support time-sensitive edge-processing workflows and buffering

10. Benchmark Suite
    - benchmarks/run_benchmark.py
    - Measures:
      - ring buffer performance
      - sliding window statistics
      - priority queue scheduling
      - CNN inference time
      - Dijkstra routing timing
      - 5G slice simulation and bandwidth savings
    - Results saved under results/benchmarks.json

---------------------------------------------------------------------
REPOSITORY STRUCTURE
---------------------------------------------------------------------
main.py                     - main Streamlit dashboard
train_satnet.py             - easy training entry point
requirements.txt            - Python dependencies
core/
  model.py                  - SatNet CNN and saved model paths
  train.py                  - training logic
  baselines.py              - baseline comparisons
  hybrid_policy.py          - AI + network decision policy
  ablation.py               - ablation logic
network/
  channel.py                - channel/CQI simulation
  mec_graph.py              - MEC topology and routing
  qos.py                    - QoS and semantic encoding
  semantic_6g.py            - semantic decision logic
  slicing.py                - 5G slice selection
utils/
  dsa.py                    - queue/buffer scheduling logic
  image_buffer.py           - image queue buffer
benchmarks/
  run_benchmark.py          - benchmark suite
results/
  baselines.json            - baseline result saved data
  benchmarks.json           - benchmark result saved data
models/
  satnet.pth                - trained model file
  train_log.json            - training history
data/
  dataset.py
  eurosat_data/eurosat/2750/ - dataset folder

---------------------------------------------------------------------
HOW TO RUN THIS PROJECT
---------------------------------------------------------------------
STEP 1: Open a terminal in the project folder

Example:
cd "U:\DSA PROJECT\Trial New Hopefully works\sat_edge"

STEP 2: Create and activate a virtual environment

Windows PowerShell:
python -m venv venv
.\venv\Scripts\Activate.ps1

Windows Command Prompt:
python -m venv venv
venv\Scripts\activate.bat

Linux / Mac:
python3 -m venv venv
source venv/bin/activate

STEP 3: Install dependencies

pip install -r requirements.txt

If needed, upgrade pip first:
pip install --upgrade pip

STEP 4: Train the model

Option A:
python train_satnet.py

Option B:
python train_satnet.py --epochs 10 --lr 0.001 --batch-size 32

Option C (direct training script):
python core/train.py --epochs 10 --lr 0.001 --batch-size 32

Notes:
- Training uses the EuroSAT dataset.
- The project saves the best model to models/satnet.pth
- If the dataset is missing, you need to place the dataset under the expected folder structure.

STEP 5: Run the dashboard

streamlit run main.py

Then open the browser URL displayed in the terminal, usually:
http://localhost:8501

STEP 6: Use the app
- Upload a satellite image
- View the model prediction
- See the 5G slice recommendation
- See the MEC node and route chosen
- Inspect semantic payload size and bandwidth savings
- Explore network and benchmarking tabs

STEP 7: Run benchmarks (optional)

python benchmarks/run_benchmark.py

This will generate results in:
results/benchmarks.json

---------------------------------------------------------------------
IMPORTANT NOTES
---------------------------------------------------------------------
- If the model has not been trained yet, the dashboard may still run in demo mode, but you should train it first for the full experience.
- The app is designed as an experimental research/demo system, not a production telecom stack.
- This project combines AI, communication simulation, routing, and edge scheduling into one interactive pipeline.
- The dataset is important: without the EuroSAT data, training will not work properly.

---------------------------------------------------------------------
WHY THIS IS A STRONG PROJECT
---------------------------------------------------------------------
This repository is more than a simple classifier. It demonstrates a modern edge-AI concept:
- AI decides what matters
- the network chooses how to transport that decision
- the system compresses and routes information efficiently
- resource usage is optimized based on class priority and channel quality

That makes it a good research prototype or GitHub portfolio project for:
- computer vision
- satellite imagery analysis
- 5G/6G communication
- edge intelligence
- network-aware AI systems
- applied ML + systems design

---------------------------------------------------------------------
SUGGESTED GITHUB DESCRIPTION
---------------------------------------------------------------------
Satellite Edge AI is a PyTorch-based satellite image classification and 5G-aware edge decision pipeline. It combines EuroSAT image classification with network slicing, CQI-aware channel simulation, MEC routing, semantic communication, and benchmarking tools inside a Streamlit dashboard. The project demonstrates how AI inference can be integrated with communication resource allocation in a realistic edge-network setting.

---------------------------------------------------------------------
QUICK START SUMMARY
---------------------------------------------------------------------
1. python -m venv venv
2. .\venv\Scripts\Activate.ps1
3. pip install -r requirements.txt
4. python train_satnet.py
5. streamlit run main.py

If you want to benchmark the system:
python benchmarks/run_benchmark.py

---------------------------------------------------------------------
FINAL NOTE
---------------------------------------------------------------------
This repo is a working prototype that fuses satellite image understanding with intelligent communication decisions. It is designed to show how AI models can be connected to 5G/6G-style network decisions in a single experimental system.

If you are publishing this on GitHub, the main story is:
"A satellite edge AI system for EuroSAT classification, network slicing, MEC routing, and semantic-aware communication simulation."

---------------------------------------------------------------------
