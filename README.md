# WATCHMEN

**Neuro-Symbolic Temporal Risk-Accumulation Framework**

---

## Overview

WATCHMEN is an AI-powered retail surveillance platform designed to assist security personnel by continuously monitoring shoppers throughout their entire visit. Unlike traditional CCTV analytics that classify isolated video clips, WATCHMEN maintains a behavioral memory for every shopper, accumulates risk over time, applies explainable symbolic reasoning, and verifies high-risk events before generating alerts.

The system aims to reduce false alarms while providing explainable, real-time threat detection for retail environments such as supermarkets, shopping malls, jewelry stores, and showrooms.

---

## Key Features

- Continuous shopper tracking across the entire visit
- Multi-stage behavioral analysis
- Adaptive Risk Accumulator
- Neuro-Symbolic Decision Engine
- Vision-Language Model (VLM) verification
- Explainable AI decisions
- Human-readable evidence timeline
- Cost-aware intelligent alerting
- Modular architecture for future research

---

## Project Architecture

```
WATCHMEN
│
├── ai_core/
│   ├── configs/
│   ├── datasets/
│   ├── models/
│   ├── src/
│   │   ├── layer1_ingestion/
│   │   ├── layer2_perception/
│   │   ├── layer3_behaviour/
│   │   ├── layer4_reasoning/
│   │   ├── layer5_vlm_verdict/
│   │   ├── layer6_explainability/
│   │   └── utils/
│   │
│   ├── tests/
│   └── requirements.txt
│
├── backend/
├── frontend/
└── README.md
```

---

## Technology Stack

### AI

- Python
- OpenCV
- YOLO
- NumPy
- PyTorch

### Backend

- Node.js
- Express.js

### Frontend

- React

### Database

- MongoDB

---

## AI Pipeline

```
Camera Feed
      │
      ▼
Layer 1 — Video Ingestion
      │
      ▼
Layer 2 — Object & Person Perception
      │
      ▼
Layer 3 — Behaviour Understanding
      │
      ▼
Layer 4 — Neuro-Symbolic Reasoning
      │
      ▼
Layer 5 — VLM Verification
      │
      ▼
Layer 6 — Explainability & Evidence Generation
      │
      ▼
Backend API
      │
      ▼
Frontend Dashboard
```

---

## Current Status

- ✅ Research completed
- ✅ Architecture finalized
- ✅ Repository restructured
- 🚧 AI implementation in progress
- 🚧 Backend integration in progress
- 🚧 Frontend development in progress

---

## Team

- Dhruv Sharma — AI Research & Neuro-Symbolic Reasoning
- Manisha — Perception & Behaviour Analysis
- Mayank — Backend Development
- Aditi — Frontend Development

---

## License

This repository is part of a Final Year B.Tech research project.
