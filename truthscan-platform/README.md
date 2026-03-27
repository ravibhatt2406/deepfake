# 🛡️ TruthScan — AI Detection & Fact Verification Platform

[![Architecture: Modular](https://img.shields.io/badge/Architecture-Modular-blueviolet)](https://github.com/ravibhatt2406/deepfake)
[![Backend: FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![Frontend: SPA](https://img.shields.io/badge/Frontend-Vanilla_JS-f7df1e)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![Deployment: Docker](https://img.shields.io/badge/Deployment-Docker-2496ed)](https://www.docker.com/)

**TruthScan** is a production-grade ecosystem designed for content integrity, authenticity analysis, and verifiable fact-checking. It combines advanced deep learning ensembles with Retrieval-Augmented Generation (RAG) to provide an end-to-end solution for combating misinformation and synthetic media.

---

## 🏗️ System Architecture

The platform uses a modular, containerized architecture for high scalability and reliability.

```mermaid
graph TD
    User((User)) -->|HTTP/80| Nginx[Nginx Reverse Proxy]
    Nginx -->|/ | Frontend[SPA Frontend]
    Nginx -->|/api/v1/| Backend[FastAPI Backend]
    
    subgraph "Backend Services"
        Backend --> AI[AI Detector Ensemble]
        Backend --> FC[Fact-Check RAG Engine]
        Backend --> IMG[Image Forensics]
        Backend --> AUD[Audio Authenticity]
        Backend --> DNA[Content DNA Fingerprinting]
    end
    
    FC --> LLM[Claude AI / GPT-4]
    FC --> Search[News Search APIs]
    DNA --> VDB[(ChromaDB Vector Store)]
    AI --> HF[HuggingFace / ONNX Models]
```

---

## 🌟 Core Modules

### 🤖 1. AI Content Detection
An ensemble of state-of-the-art models (EfficientNet, Xception, CLIP) capable of detecting:
- **Synthetic Images/Video**: Perceptual and frequency-level anomaly detection.
- **AI-Generated Text**: Large-scale language model signature detection.
- **Deepfake URLs**: Real-time crawling and analysis of suspicious links.

### 📰 2. RAG-based Fact-Checking
Moving beyond simple pattern matching, our fact-checker uses **Retrieval-Augmented Generation**:
- **Zero Hallucination**: Grounding all verdicts in retrieved snippets from trusted news sources (Reuters, BBC, AP).
- **Explainable AI**: Providing line-by-line evidence, source links, and credibility scores.
- **Cross-Lingual support**: Detection of disinformation across multiple languages.

### 🔬 3. Multi-Layer Image Forensics
Deep-level analysis of image authenticity:
- **ELA (Error Level Analysis)**: Identifying compression inconsistencies.
- **Metadata Scrubbing**: Recovering hidden EXIF/GPS/Software signatures.
- **GAN Fingerprinting**: Specialized detection for StyleGAN and DALL-E signatures.

### 🎧 4. Audio Authenticity (New)
Sophisticated analysis for the "Deepfake Audio" era:
- **Synthetic Voice Detection**: Identifying robotic prosody and phoneme repetition.
- **Voice Cloning Analysis**: Matching spectral signatures against known human profiles.
- **Tamper Detection**: Detecting splicing, editing, and background noise floor mismatches.

---

## 🚀 Deployment & Installation

### Option A: One-Click (Docker Compose)
This is the recommended way to run the full stack (Nginx + Backend + Frontend).

```bash
cd truthscan-platform
docker-compose up --build -d
```
Access the platform at **`http://localhost`**.

### Option B: Local Development

**Backend Setup:**
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env       # Add your API keys
$env:PYTHONPATH="."
uvicorn main:app --port 8000 --reload
```

**Frontend Setup:**
Simply serve the `frontend/` folder:
```bash
cd frontend
python -m http.server 80
```

---

## 🛠️ Technology Stack

| Layer | Technologies |
|-------|--------------|
| **Core** | Python 3.10+, FastAPI, Pydantic v2 |
| **AI/ML** | PyTorch, ONNX, HuggingFace, OpenAI CLIP, EfficientNet |
| **Database** | ChromaDB (Vector), Redis (Cache) |
| **Frontend** | Vanilla JS, CSS3, HTML5 (SPA Architecture) |
| **Infrastructure** | Docker, Nginx, GitHub Actions |

---

## 📺 Project Previews

| Dashboard | Result Analysis |
|-----------|-----------------|
| ![Dashboard](file:///C:/Users/User/.gemini/antigravity/brain/e73e041c-9c79-439b-aa92-3389022a1b7c/home_page_clean_ui_1774621256780.png) | ![Results](file:///C:/Users/User/.gemini/antigravity/brain/e73e041c-9c79-439b-aa92-3389022a1b7c/ai_detector_clean_ui_1774621246237.png) |

---

## 📜 License
Developed as part of the **TruthScan Initiative**. Proprietary — All rights reserved.
