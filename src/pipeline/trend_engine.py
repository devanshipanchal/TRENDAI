"""
src/pipeline/trend_engine.py
─────────────────────────────
Trend detection: takes a dataframe of embeddings + timestamps, runs
UMAP → HDBSCAN clustering, scores temporal velocity, and returns
ranked TrendReport objects.

Pipeline:
    embeddings (N×768)
        └─ UMAP (N×50)
              └─ HDBSCAN clusters
                    └─ temporal scoring
                          └─ ranked TrendReport list
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

# ── Optional heavy imports ────────────────────────────────────────────────────

def _import_umap():
    try:
        import umap
        return umap.UMAP
    except ImportError:
        warnings.warn("umap-learn not installed. Skipping dimensionality reduction.", stacklevel=2)
        return None

def _import_hdbscan():
    try:
        import hdbscan
        return hdbscan.HDBSCAN
    except ImportError:
        warnings.warn("hdbscan not installed. Using KMeans fallback.", stacklevel=2)
        return None


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class TrendSignal:
    """A single data point contributing to a trend."""
    item_id: str
    cluster_id: int
    embedding: np.ndarray
    post_date: datetime
    engagement_score: float
    platform: str
    caption: str
    attributes: dict


@dataclass
class TrendReport:
    """Fully scored trend with metadata ready for LLM synthesis."""
    cluster_id: int
    trend_name: str                     # auto-named or user-labeled
    velocity: str                       # rising / peak / stable / declining
    velocity_score: float               # [-1, 1]  positive = rising
    item_count: int
    engagement_mean: float
    engagement_trend: float             # slope of engagement over time
    dominant_platform: str
    top_keywords: list[str]
    color_palette: list[str]
    representative_items: list[str]     # item_ids
    date_first_seen: str
    date_peak: str
    umap_centroid: list[float]          # 2D for visualization
    raw_signals: list[TrendSignal] = field(default_factory=list, repr=False)

    def to_llm_context(self) -> str:
        """Serialize for LLM prompt injection."""
        return (
            f"Trend: {self.trend_name}\n"
            f"Velocity: {self.velocity} (score: {self.velocity_score:+.2f})\n"
            f"Items seen: {self.item_count}\n"
            f"Avg engagement: {self.engagement_mean:.3f}\n"
            f"Dominant platform: {self.dominant_platform}\n"
            f"Top keywords: {', '.join(self.top_keywords)}\n"
            f"Colors: {', '.join(self.color_palette)}\n"
            f"Active since: {self.date_first_seen}\n"
        )


# ── Trend Engine ─────────────────────────────────────────────────────────────

class TrendEngine:
    """
    Full trend detection pipeline.

    Args:
        umap_n_components: UMAP output dims before clustering (default 50)
        umap_n_neighbors: UMAP local structure (default 15)
        min_cluster_size: HDBSCAN minimum cluster size
        min_samples: HDBSCAN core point density
        trend_window_days: lookback window for velocity calculation
        top_k: max trends to return
    """

    def __init__(
        self,
        umap_n_components: int = 50,
        umap_n_neighbors: int = 15,
        min_cluster_size: int = 10,
        min_samples: int = 5,
        trend_window_days: int = 30,
        top_k: int = 20,
    ):
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.trend_window_days = trend_window_days
        self.top_k = top_k

        self._umap_model = None
        self._clusterer = None
        self._umap_2d = None   # for visualization

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_trends(self, df: pd.DataFrame) -> list[TrendReport]:
        """
        Main entry point.

        Args:
            df: DataFrame with columns:
                - item_id (str)
                - embedding (list[float] or np.ndarray, dim 768)
                - post_date (str ISO8601 or datetime)
                - engagement_score (float)
                - platform (str)
                - caption (str)
                - keywords (str, pipe-separated)
                - color_primary (str hex)

        Returns:
            List of TrendReport sorted by velocity_score desc.
        """
        df = df.copy()
        df["post_date"] = pd.to_datetime(df["post_date"])

        # 1. Extract embedding matrix
        embeddings = self._extract_embeddings(df)

        # 2. Dimensionality reduction
        reduced = self._run_umap(embeddings, n_components=self.umap_n_components)
        reduced_2d = self._run_umap(embeddings, n_components=2)
        self._umap_2d = reduced_2d

        # 3. Clustering
        labels = self._run_clustering(reduced)
        df["cluster_id"] = labels

        # 4. Score each cluster
        unique_clusters = [c for c in np.unique(labels) if c != -1]
        reports = []
        for cid in unique_clusters:
            cluster_df = df[df["cluster_id"] == cid]
            cluster_2d = reduced_2d[df["cluster_id"] == cid]
            report = self._build_trend_report(cid, cluster_df, cluster_2d)
            reports.append(report)

        # 5. Sort by velocity score, return top_k
        reports.sort(key=lambda r: r.velocity_score, reverse=True)
        return reports[: self.top_k]

    # ── Stage implementations ─────────────────────────────────────────────────

    def _extract_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        if "embedding" in df.columns:
            first = df["embedding"].iloc[0]
            if isinstance(first, (list, np.ndarray)):
                return np.vstack(df["embedding"].values)
        # Fallback: generate random embeddings for testing
        warnings.warn("No 'embedding' column found. Generating random embeddings.", stacklevel=2)
        return np.random.default_rng(42).standard_normal((len(df), 768)).astype(np.float32)

    def _run_umap(self, embeddings: np.ndarray, n_components: int) -> np.ndarray:
        UMAP = _import_umap()
        if UMAP is None:
            # PCA fallback
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(n_components, embeddings.shape[1]))
            return pca.fit_transform(embeddings)

        model = UMAP(
            n_components=n_components,
            n_neighbors=self.umap_n_neighbors,
            metric="cosine",
            random_state=42,
            low_memory=True,
        )
        return model.fit_transform(embeddings)

    def _run_clustering(self, reduced: np.ndarray) -> np.ndarray:
        HDBSCAN = _import_hdbscan()
        if HDBSCAN is None:
            from sklearn.cluster import KMeans
            k = max(5, len(reduced) // 100)
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            return km.fit_predict(reduced)

        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        self._clusterer = clusterer
        return clusterer.fit_predict(reduced)

    def _build_trend_report(
        self, cluster_id: int, df: pd.DataFrame, coords_2d: np.ndarray
    ) -> TrendReport:
        now = df["post_date"].max()
        window_start = now - timedelta(days=self.trend_window_days)

        recent = df[df["post_date"] >= window_start]
        older = df[df["post_date"] < window_start]

        # Engagement velocity
        recent_eng = recent["engagement_score"].mean() if len(recent) > 0 else 0
        older_eng = older["engagement_score"].mean() if len(older) > 0 else recent_eng
        velocity_score = float(np.clip((recent_eng - older_eng) / (older_eng + 1e-6), -1, 1))

        velocity_label = (
            "rising" if velocity_score > 0.15
            else "declining" if velocity_score < -0.15
            else "peak" if recent_eng > 0.7
            else "stable"
        )

        # Keywords from captions
        all_keywords: list[str] = []
        if "keywords" in df.columns:
            for kw_str in df["keywords"].dropna():
                all_keywords.extend(str(kw_str).split("|"))
        from collections import Counter
        top_kw = [k for k, _ in Counter(all_keywords).most_common(8)]

        # Colors
        colors = df["color_primary"].dropna().value_counts().head(5).index.tolist() \
            if "color_primary" in df.columns else []

        # Platform
        dom_platform = df["platform"].value_counts().idxmax() \
            if "platform" in df.columns else "unknown"

        # Representative items (highest engagement)
        rep_items = df.nlargest(5, "engagement_score")["item_id"].tolist() \
            if "item_id" in df.columns else []

        # Centroid in 2D UMAP space
        centroid_2d = coords_2d.mean(axis=0).tolist() if len(coords_2d) > 0 else [0.0, 0.0]

        # Auto-name from keywords
        trend_name = (
            f"trend_{cluster_id}"
            if not top_kw
            else f"{top_kw[0].replace(' ', '_')}_{cluster_id}"
        )

        return TrendReport(
            cluster_id=cluster_id,
            trend_name=trend_name,
            velocity=velocity_label,
            velocity_score=round(velocity_score, 4),
            item_count=len(df),
            engagement_mean=round(float(df["engagement_score"].mean()), 4),
            engagement_trend=round(float(recent_eng - older_eng), 4),
            dominant_platform=dom_platform,
            top_keywords=top_kw,
            color_palette=colors,
            representative_items=rep_items,
            date_first_seen=df["post_date"].min().strftime("%Y-%m-%d"),
            date_peak=df.loc[df["engagement_score"].idxmax(), "post_date"].strftime("%Y-%m-%d"),
            umap_centroid=centroid_2d,
        )


# ── Utility: semantic search ──────────────────────────────────────────────────

def find_similar_items(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    item_ids: list[str],
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """
    Cosine similarity search.
    Returns list of (item_id, similarity_score) sorted desc.
    """
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    normed = embeddings / norms
    scores = normed @ query_norm
    top_indices = np.argsort(-scores)[:top_k]
    return [(item_ids[i], float(scores[i])) for i in top_indices]
