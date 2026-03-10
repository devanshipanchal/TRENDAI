"""
src/api/app.py
───────────────
TrendAI REST API — FastAPI application.

Endpoints:
    POST /analyze           — Analyze a single image or URL
    POST /forecast          — Run full trend forecast on dataset
    GET  /trends            — Get latest cached trend list
    GET  /trends/{id}       — Get single trend details
    POST /style-card        — Generate a style card for a trend name
    GET  /health            — Health check

Run:
    uvicorn src.api.app:app --reload --port 8000
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv


from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, HttpUrl

# ── App setup ─────────────────────────────────────────────────────────────────

load_dotenv()

app = FastAPI(
    title="TrendAI",
    description="GenAI Fashion Trend Analyzer API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# In-memory cache (replace with Redis in production)
_cache: dict[str, dict] = {}


# ── Request / Response models ─────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    image_url: Optional[str] = Field(None, description="Public URL of fashion image")
    image_base64: Optional[str] = Field(None, description="Base64-encoded image")
    caption: Optional[str] = Field(None, description="Optional existing caption")

    class Config:
        json_schema_extra = {
            "example": {
                "image_url": "https://example.com/outfit.jpg",
                "caption": "Oversized linen blazer with wide-leg trousers",
            }
        }


class AnalyzeResponse(BaseModel):
    item_id: str
    detected_trends: list[str]
    primary_trend: str
    trend_velocity: str
    attributes: dict
    caption: str
    embedding_preview: list[float]
    confidence: float
    processed_at: str


class ForecastRequest(BaseModel):
    dataset_path: Optional[str] = Field(
        None, description="Path to parquet/CSV dataset. Uses synthetic if omitted."
    )
    analysis_window_days: int = Field(30, ge=7, le=365)
    top_k_trends: int = Field(10, ge=1, le=50)
    include_style_cards: bool = True

    class Config:
        json_schema_extra = {
            "example": {
                "analysis_window_days": 30,
                "top_k_trends": 10,
                "include_style_cards": True,
            }
        }


class ForecastResponse(BaseModel):
    forecast_id: str
    generated_at: str
    status: str
    executive_summary: str
    macro_direction: str
    total_trends_analyzed: int
    rising_count: int
    declining_count: int
    style_cards: list[dict]


class StyleCardRequest(BaseModel):
    trend_name: str
    trend_data: Optional[dict] = None   # optional pre-computed trend data

    class Config:
        json_schema_extra = {
            "example": {
                "trend_name": "gorpcore",
                "trend_data": {
                    "velocity": "rising",
                    "keywords": ["fleece", "trail runners", "Gore-Tex"],
                    "item_count": 342,
                    "dominant_platform": "instagram",
                }
            }
        }


class StyleCardResponse(BaseModel):
    trend_name: str
    headline: str
    narrative: str
    key_pieces: list[str]
    styling_tips: list[str]
    color_story: str
    target_demographic: str
    price_entry_point: str
    where_to_shop: list[str]
    forecast_horizon: str
    confidence: float
    generated_at: str


class TrendListResponse(BaseModel):
    trends: list[dict]
    total: int
    generated_at: str
    cache_age_seconds: float


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    services: dict


# ── Helper: get encoder (lazy init) ──────────────────────────────────────────

_encoder = None

def get_encoder():
    global _encoder
    if _encoder is None:
        from src.models.vision_encoder import VisionEncoder
        _encoder = VisionEncoder(mock=True)
    return _encoder


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "TrendAI API running"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat() + "Z",
        services={
            "anthropic": "configured" if os.getenv("ANTHROPIC_API_KEY") else "missing_key",
            "redis": "not_configured",
            "s3": "not_configured",
        },
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(req: AnalyzeRequest):
    """
    Analyze a single fashion image.
    Returns trend attribution, garment attributes, and CLIP embedding.
    """
    if not req.image_url and not req.image_base64 and not req.caption:
        raise HTTPException(400, "Provide image_url, image_base64, or caption")

    encoder = get_encoder()
    item_id = str(uuid.uuid4())[:12]

    # Load image
    from PIL import Image
    import io, base64, httpx

    img = None
    if req.image_base64:
        try:
            img_bytes = base64.b64decode(req.image_base64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            raise HTTPException(400, "Invalid base64 image")
    elif req.image_url:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(req.image_url, timeout=10)
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))

    if img is None:
        img = Image.new("RGB", (224, 224), color=(200, 180, 160))

    result = encoder.encode_image(img, prompt=req.caption or "")
    caption = result.caption if result.caption != "Fashion item." else (req.caption or "Fashion item.")

    # Trend attribution (simplified — in production uses nearest cluster centroid)
    import random
    trends = ["gorpcore", "quiet_luxury", "dopamine_dressing", "soft_minimalism", "biker_aesthetic"]
    primary_trend = random.choice(trends)
    velocities = {"gorpcore": "rising", "quiet_luxury": "declining", "dopamine_dressing": "peak",
                  "soft_minimalism": "rising", "biker_aesthetic": "rising"}

    return AnalyzeResponse(
        item_id=item_id,
        detected_trends=random.sample(trends, k=3),
        primary_trend=primary_trend,
        trend_velocity=velocities.get(primary_trend, "stable"),
        attributes=result.attributes,
        caption=caption,
        embedding_preview=result.embedding[:8].tolist(),
        confidence=round(random.uniform(0.72, 0.95), 3),
        processed_at=datetime.utcnow().isoformat() + "Z",
    )


@app.post("/forecast", response_model=ForecastResponse)
async def run_forecast_endpoint(req: ForecastRequest, background_tasks: BackgroundTasks):
    """
    Run full trend forecast pipeline.
    Loads dataset → clusters → LLM synthesis.
    Returns a ForecastResponse with style cards.
    """
    forecast_id = str(uuid.uuid4())[:8]

    # Check cache
    cache_key = f"forecast_{req.analysis_window_days}_{req.top_k_trends}"
    if cache_key in _cache and (time.time() - _cache[cache_key]["_ts"]) < 3600:
        cached = _cache[cache_key]
        return ForecastResponse(**{k: v for k, v in cached.items() if not k.startswith("_")})

    # Load dataset
    import pandas as pd
    from pathlib import Path

    dataset_path = req.dataset_path or "data/synthetic/fashion_dataset.parquet"
    try:
        if dataset_path.endswith(".parquet"):
            df = pd.read_parquet(dataset_path)
        else:
            df = pd.read_csv(dataset_path)
    except Exception:
        raise HTTPException(404, f"Dataset not found: {dataset_path}. Run generate_dataset.py first.")

    # Detect trends
    from src.pipeline.trend_engine import TrendEngine
    engine = TrendEngine(
        min_cluster_size=10,
        top_k=req.top_k_trends,
        trend_window_days=req.analysis_window_days,
    )
    trend_reports = engine.detect_trends(df)

    # LLM synthesis
    style_cards = []
    executive_summary = f"Analysis of {len(df):,} items revealed {len(trend_reports)} distinct trend clusters."
    macro_direction = "The market is moving toward elevated utility and expressive minimalism."

    if req.include_style_cards and os.getenv("ANTHROPIC_API_KEY"):
        from src.agents.forecast_agent import run_forecast
        forecast = run_forecast(trend_reports, analysis_window_days=req.analysis_window_days)
        import dataclasses
        style_cards = [dataclasses.asdict(c) for c in forecast.style_cards]
        executive_summary = forecast.executive_summary or executive_summary
        macro_direction = forecast.macro_direction or macro_direction

    rising = sum(1 for r in trend_reports if r.velocity == "rising")
    declining = sum(1 for r in trend_reports if r.velocity == "declining")

    response_data = {
        "forecast_id": forecast_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "status": "complete",
        "executive_summary": executive_summary,
        "macro_direction": macro_direction,
        "total_trends_analyzed": len(trend_reports),
        "rising_count": rising,
        "declining_count": declining,
        "style_cards": style_cards,
    }

    _cache[cache_key] = {**response_data, "_ts": time.time()}
    return ForecastResponse(**response_data)


@app.get("/trends", response_model=TrendListResponse)
async def list_trends():
    """Return latest cached trend list."""
    import pandas as pd
    import numpy as np
    import dataclasses
    from pathlib import Path
    from src.pipeline.trend_engine import TrendEngine

    dataset_path = Path("data/synthetic/fashion_dataset.parquet")
    if not dataset_path.exists():
        raise HTTPException(404, "No dataset found. Run generate_dataset.py first.")

    df = pd.read_parquet(dataset_path)
    engine = TrendEngine(min_cluster_size=10, top_k=20)
    reports = engine.detect_trends(df)

    
    import dataclasses
    trends_data = []
    for r in reports:
        d = dataclasses.asdict(r)
        d.pop("raw_signals", None)
        trends_data.append(d)

         # convert numpy types
        for k, v in d.items():
            if isinstance(v, np.integer):
                d[k] = int(v)
            elif isinstance(v, np.floating):
                d[k] = float(v)
        trends_data.append(d)

    return TrendListResponse(
        trends=trends_data,
        total=len(trends_data),
        generated_at=datetime.utcnow().isoformat() + "Z",
        cache_age_seconds=0.0,
    )


    return jsonable_encoder(
        TrendListResponse(
            trends=trends_data,
            total=len(trends_data),
            generated_at=datetime.utcnow().isoformat() + "Z",
            cache_age_seconds=0.0,
        )
    )


@app.get("/trends/{cluster_id}")
async def get_trend(cluster_id: int):
    """Get details for a specific trend cluster."""
    result = await list_trends()
    for trend in result.trends:
        if trend.get("cluster_id") == cluster_id:
            return trend
    raise HTTPException(404, f"Trend cluster {cluster_id} not found")


@app.post("/style-card", response_model=StyleCardResponse)
async def generate_style_card(req: StyleCardRequest):
    """
    Generate a style card for a named trend using Claude.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "ANTHROPIC_API_KEY not configured")

    import anthropic
    import json, re

    client = anthropic.Anthropic()

    trend_context = f"Trend: {req.trend_name}"
    if req.trend_data:
        trend_context += f"\nData: {json.dumps(req.trend_data, indent=2)}"

    prompt = f"""Generate a comprehensive style card for this fashion trend.

{trend_context}

Return ONLY valid JSON matching this structure:
{{
  "headline": "punchy 10-word headline",
  "narrative": "2-3 paragraph trend story",
  "key_pieces": ["item 1", "item 2", "item 3", "item 4", "item 5"],
  "styling_tips": ["tip 1", "tip 2", "tip 3"],
  "color_story": "1 sentence color palette description",
  "target_demographic": "primary audience",
  "price_entry_point": "price range description",
  "where_to_shop": ["brand 1", "brand 2", "brand 3"],
  "forecast_horizon": "timing prediction",
  "confidence": 0.85
}}"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )

    text = message.content[0].text
    text = re.sub(r"```json|```", "", text).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"LLM returned invalid JSON: {e}")

    return StyleCardResponse(
        trend_name=req.trend_name,
        headline=data.get("headline", ""),
        narrative=data.get("narrative", ""),
        key_pieces=data.get("key_pieces", []),
        styling_tips=data.get("styling_tips", []),
        color_story=data.get("color_story", ""),
        target_demographic=data.get("target_demographic", ""),
        price_entry_point=data.get("price_entry_point", ""),
        where_to_shop=data.get("where_to_shop", []),
        forecast_horizon=data.get("forecast_horizon", ""),
        confidence=float(data.get("confidence", 0.8)),
        generated_at=datetime.utcnow().isoformat() + "Z",
    )


# ── Error handlers ────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": type(exc).__name__,
            "detail": str(exc),
            "path": str(request.url),
        },
    )

   
