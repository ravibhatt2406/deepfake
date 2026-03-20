# TruthScan — AI Detection & Fact Verification Platform

A complete platform for detecting AI-generated content, verifying fake news, and analysing image originality.

## Project Structure

```
TruthScan-Project/
├── frontend/
│   └── truthscan-platform.html   ← Open this in a browser to use the app
└── backend/
    ├── main.py                   ← FastAPI app entry point
    ├── requirements.txt          ← Python dependencies
    ├── Dockerfile                ← Docker setup
    ├── .env.example              ← Copy to .env and fill in your keys
    ├── README.md                 ← Full backend setup guide
    ├── ai_detection/             ← Module 1: AI/Deepfake detector
    ├── fake_news/                ← Module 2: Fake news RAG pipeline
    ├── shared/                   ← Schemas, cache, config
    ├── scripts/                  ← Model setup & knowledge base indexer
    └── tests/                    ← Accuracy benchmarks
```

## Quick Start (Frontend Only)

Just open `frontend/truthscan-platform.html` in any browser.
No server needed. All 3 modules work out of the box inside claude.ai.

## Modules

| Module | Description |
|--------|-------------|
| 🎞️ AI Detector | Detects AI-generated images, video, text, documents, URLs |
| 📰 Fake News | Verifies claims against Reuters, BBC, AP, Snopes, PolitiFact etc. |
| 🔬 Image Originality | ELA heatmap, EXIF metadata, tamper detection, GAN analysis |

## Running the Backend (Optional)

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env        # add your ANTHROPIC_API_KEY
uvicorn main:app --port 8000
```

See `backend/README.md` for full setup including model downloads and ChromaDB indexing.
