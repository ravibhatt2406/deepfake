# TruthScan — AI Detection & Fact Verification Platform

A production-grade platform for detecting AI-generated content, verifying fake news, and analysing image/audio originality.

## 🏗️ Professional Architecture

The platform has been restructured for scalable, containerised deployment:

```
truthscan-platform/
├── frontend/
│   └── index.html               ← SPA Frontend (Static Assets)
├── backend/
│   ├── main.py                  ← FastAPI Production Entry Point
│   ├── shared/                  ← Unified Schemas & Config
│   ├── ai_detection/            ← AI Detector Module
│   ├── fake_news/               ← RAG Fact-Check Engine
│   ├── image_forensics/         ← Image Analysis Module
│   └── audio_forensics/         ← Audio Analysis Module
├── nginx/
│   └── nginx.conf               ← Reverse Proxy & Load Balancer
└── docker-compose.yml           ← Multi-container Orchestration
```

## 🚀 Deployment (Docker)

The fastest way to deploy the entire stack:

```bash
docker-compose up --build -d
```
Access the platform at `http://localhost`.

## 🛠️ Local Development

### Backend
```bash
cd backend
$env:PYTHONPATH="."
uvicorn main:app --port 8000 --reload
```

### Frontend
Serve the `frontend/` directory using any static web server:
```bash
cd frontend
python -m http.server 80
```

## 🎧 New: Audio Authenticity Module
Detect synthetic voices, cloning, and manual edits (splicing/noise anomalies) in WAV/MP3 files.

## ⚖️ License
Proprietary — All rights reserved.
