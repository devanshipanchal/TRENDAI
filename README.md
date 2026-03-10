# TrendAI — GenAI Fashion Trend Analyzer

> End-to-end pipeline that ingests multimodal fashion data, detects emerging trends using vision-language models, and generates actionable style forecasts.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          TrendAI System                              │
│                                                                       │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────────┐ │
│  │  Ingest  │──▶│  Vision  │──▶│  Trend   │──▶│   Generative     │ │
│  │  Layer   │   │  Encoder │   │  Engine  │   │   Forecast API   │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────────────┘ │
│       │               │              │                   │           │
│  Social/Web      CLIP + BLIP    Clustering +        LLM Synthesis   │
│  Scraper API    Embeddings      Time-Series         + Style Cards    │
└─────────────────────────────────────────────────────────────────────┘
```

## Pipeline Stages

1. **Ingest** — Scrape + normalize fashion images & metadata from Instagram hashtags, Pinterest boards, runway feeds, and retail catalogues
2. **Vision Encode** — CLIP embeds images; BLIP-2 generates captions; attribute classifier tags garments
3. **Trend Detection** — HDBSCAN clusters embeddings; temporal drift scoring ranks rising/falling trends
4. **Forecast** — LLM synthesizes trend narratives, generates style cards, outputs structured JSON forecasts
5. **Serve** — FastAPI exposes `/analyze`, `/forecast`, `/style-card` endpoints

---

## Project Structure

```
trendai/
├── src/
│   ├── agents/          # Orchestration agents (LangGraph)
│   ├── pipeline/        # Each stage as a composable module
│   ├── models/          # Model wrappers (CLIP, BLIP, LLM)
│   ├── api/             # FastAPI app
│   └── utils/           # Shared helpers
├── data/
│   ├── raw/             # Scraped images + metadata
│   ├── processed/       # Embeddings, captions, attributes
│   └── synthetic/       # Generated dataset (see below)
├── notebooks/           # EDA + prototyping
├── tests/
└── docs/
```

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env          # add API keys
python -m src.pipeline.generate_dataset   # build synthetic dataset
python -m src.pipeline.run_pipeline       # full end-to-end run
uvicorn src.api.app:app --reload          # start API
```

## Dataset

See `data/synthetic/` — 5000 synthetic fashion items with images paths, metadata, timestamps, and pre-labeled trend categories. Real deployment connects to `src/pipeline/ingest.py` scrapers.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Vision Encoding | OpenAI CLIP (`ViT-L/14`), BLIP-2 |
| Clustering | HDBSCAN, UMAP |
| LLM Synthesis | Claude claude-sonnet-4-20250514 (via Anthropic API) |
| Orchestration | LangGraph |
| Serving | FastAPI + Redis |
| Tracking | MLflow |
| Storage | S3-compatible (MinIO local) |
