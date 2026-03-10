"""
tests/test_pipeline.py
───────────────────────
Unit and integration tests for TrendAI pipeline.

Run: pytest tests/ -v
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """50-item sample dataframe with mock embeddings."""
    rng = np.random.default_rng(42)
    n = 50
    trends = ["gorpcore", "quiet_luxury", "dopamine_dressing", "biker_aesthetic", "soft_minimalism"]
    records = []
    for i in range(n):
        trend = trends[i % len(trends)]
        records.append({
            "item_id": f"item_{i:04d}",
            "trend_cluster": trend,
            "trend_velocity": ["rising", "stable", "peak", "rising", "declining"][i % 5],
            "engagement_score": float(rng.uniform(0.1, 0.95)),
            "post_date": (datetime(2024, 1, 1) + timedelta(days=i * 5)).isoformat(),
            "platform": ["instagram", "pinterest", "tiktok"][i % 3],
            "caption": f"A {trend} fashion item with interesting styling",
            "keywords": "cashmere|tailored|neutral",
            "color_primary": "#C4B7A6",
            "embedding": rng.standard_normal(768).tolist(),
        })
    return pd.DataFrame(records)


@pytest.fixture
def mock_trend_report():
    """A synthetic TrendReport-like dict."""
    return {
        "cluster_id": 0,
        "trend_name": "fleece_0",
        "velocity": "rising",
        "velocity_score": 0.42,
        "item_count": 85,
        "engagement_mean": 0.67,
        "engagement_trend": 0.12,
        "dominant_platform": "instagram",
        "top_keywords": ["fleece", "gorpcore", "trail", "utility"],
        "color_palette": ["#2D5016", "#FF6600"],
        "representative_items": ["item_0001", "item_0042"],
        "date_first_seen": "2024-01-15",
        "date_peak": "2024-11-20",
        "umap_centroid": [1.23, -0.45],
    }


# ── Dataset generation tests ──────────────────────────────────────────────────

class TestDatasetGeneration:

    def test_build_dataset_shape(self):
        from src.pipeline.generate_dataset import build_dataset
        df = build_dataset(n=100)
        assert len(df) == 100
        assert "item_id" in df.columns
        assert "trend_cluster" in df.columns
        assert "embedding" in df.columns
        assert "engagement_score" in df.columns

    def test_embedding_is_unit_normed(self):
        from src.pipeline.generate_dataset import build_dataset
        df = build_dataset(n=10)
        for emb in df["embedding"]:
            norm = np.linalg.norm(emb)
            assert abs(norm - 1.0) < 1e-5, f"Embedding not unit-normed: norm={norm}"

    def test_trend_distribution_reasonable(self):
        from src.pipeline.generate_dataset import build_dataset
        df = build_dataset(n=500)
        # Each trend should have at least some items (roughly uniform)
        counts = df["trend_cluster"].value_counts()
        assert counts.min() >= 20, f"Some trend has too few items: {counts.min()}"
        assert counts.max() <= 200, f"Some trend has too many items: {counts.max()}"

    def test_engagement_scores_in_range(self):
        from src.pipeline.generate_dataset import build_dataset
        df = build_dataset(n=200)
        assert df["engagement_score"].between(0, 1).all()

    def test_save_and_load(self, tmp_path):
        from src.pipeline.generate_dataset import build_dataset, save_dataset
        df = build_dataset(n=50)
        save_dataset(df, tmp_path)
        assert (tmp_path / "fashion_dataset.parquet").exists()
        assert (tmp_path / "fashion_dataset.csv").exists()
        assert (tmp_path / "dataset_card.json").exists()

        # Reload and verify
        df2 = pd.read_parquet(tmp_path / "fashion_dataset.parquet")
        assert len(df2) == len(df)


# ── Vision encoder tests ──────────────────────────────────────────────────────

class TestVisionEncoder:

    def test_mock_embedding_shape(self):
        from src.models.vision_encoder import VisionEncoder
        enc = VisionEncoder(mock=True)
        emb = enc._mock_embedding()
        assert emb.shape == (768,)

    def test_mock_embedding_is_normalized(self):
        from src.models.vision_encoder import VisionEncoder
        enc = VisionEncoder(mock=True)
        emb = enc._mock_embedding()
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-5

    def test_encode_image_returns_result(self):
        from src.models.vision_encoder import VisionEncoder, EncodingResult
        from PIL import Image
        enc = VisionEncoder(mock=True)
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        result = enc.encode_image(img)
        assert isinstance(result, EncodingResult)
        assert result.embedding.shape == (768,)
        assert isinstance(result.caption, str)
        assert isinstance(result.attributes, dict)

    def test_encode_batch(self):
        from src.models.vision_encoder import VisionEncoder
        from PIL import Image
        enc = VisionEncoder(mock=True)
        images = [Image.new("RGB", (224, 224)) for _ in range(5)]
        result = enc.encode_batch(images, batch_size=3)
        assert result.embeddings.shape == (5, 768)
        assert len(result.captions) == 5
        assert len(result.attributes) == 5

    def test_attribute_extraction(self):
        from src.models.vision_encoder import extract_attributes
        attrs = extract_attributes("A tailored cashmere blazer in beige, paired with wide-leg trousers")
        assert attrs["garment_category"] == "outerwear"
        assert "cashmere" in attrs.get("fabrics", [])
        assert attrs["color_family"] == "neutrals"


# ── Trend engine tests ────────────────────────────────────────────────────────

class TestTrendEngine:

    def test_detect_trends_returns_reports(self, sample_df):
        from src.pipeline.trend_engine import TrendEngine
        engine = TrendEngine(min_cluster_size=5, top_k=5)
        reports = engine.detect_trends(sample_df)
        assert len(reports) > 0
        assert len(reports) <= 5

    def test_trend_report_has_required_fields(self, sample_df):
        from src.pipeline.trend_engine import TrendEngine, TrendReport
        engine = TrendEngine(min_cluster_size=5, top_k=3)
        reports = engine.detect_trends(sample_df)
        for r in reports:
            assert isinstance(r, TrendReport)
            assert isinstance(r.velocity_score, float)
            assert r.velocity in ("rising", "peak", "stable", "declining")
            assert r.item_count > 0
            assert isinstance(r.top_keywords, list)

    def test_reports_sorted_by_velocity(self, sample_df):
        from src.pipeline.trend_engine import TrendEngine
        engine = TrendEngine(min_cluster_size=5, top_k=10)
        reports = engine.detect_trends(sample_df)
        scores = [r.velocity_score for r in reports]
        assert scores == sorted(scores, reverse=True), "Reports not sorted by velocity score"

    def test_semantic_search(self):
        from src.pipeline.trend_engine import find_similar_items
        rng = np.random.default_rng(0)
        embeddings = rng.standard_normal((20, 768)).astype(np.float32)
        query = embeddings[0].copy()  # should be its own nearest neighbor
        item_ids = [f"item_{i}" for i in range(20)]
        results = find_similar_items(query, embeddings, item_ids, top_k=5)
        assert results[0][0] == "item_0", "Query item should be its own nearest neighbor"
        assert len(results) == 5
        assert all(0 <= score <= 1.01 for _, score in results)


# ── Ingest tests ──────────────────────────────────────────────────────────────

class TestIngest:

    def test_mock_scraper_yields_items(self):
        from src.pipeline.ingest import MockScraper
        scraper = MockScraper()
        items = list(scraper.scrape(limit=20))
        assert len(items) == 20

    def test_fashion_item_engagement(self):
        from src.pipeline.ingest import FashionItem
        item = FashionItem(
            item_id="test",
            source="instagram",
            image_url="https://example.com/img.jpg",
            caption="test",
            post_date="2024-01-01",
            platform="instagram",
            author_handle="@test",
            likes=50000,
            comments=1000,
            hashtags=[],
        )
        score = item.engagement_score(max_likes=100_000)
        assert 0 < score <= 1.0

    def test_ingest_pipeline_runs(self, tmp_path):
        from src.pipeline.ingest import IngestPipeline, MockScraper
        pipeline = IngestPipeline(
            scrapers=[MockScraper()],
            output_dir=tmp_path,
            download_images=False,
        )
        items = pipeline.run(limit=10)
        assert len(items) == 10
        assert (tmp_path / "manifest.jsonl").exists()


# ── API tests ─────────────────────────────────────────────────────────────────

class TestAPI:

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from src.api.app import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_analyze_with_caption(self, client):
        resp = client.post("/analyze", json={
            "caption": "Cashmere turtleneck in cream with tailored trousers"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "item_id" in data
        assert "primary_trend" in data
        assert "attributes" in data
        assert "embedding_preview" in data
        assert len(data["embedding_preview"]) == 8

    def test_analyze_missing_input(self, client):
        resp = client.post("/analyze", json={})
        assert resp.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
