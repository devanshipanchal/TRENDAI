"""
src/pipeline/generate_dataset.py
─────────────────────────────────
Generates a synthetic fashion dataset of 5000 items with realistic
metadata, trend labels, timestamps, and embedding-ready fields.

Run:  python -m src.pipeline.generate_dataset
Output: data/synthetic/fashion_dataset.parquet
         data/synthetic/fashion_dataset.csv
         data/synthetic/dataset_card.json
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Seed for reproducibility ──────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Taxonomy ──────────────────────────────────────────────────────────────────

TREND_CLUSTERS = {
    "quiet_luxury": {
        "description": "Understated, high-quality basics; neutral palette; no logos",
        "keywords": ["cashmere", "cream", "tailored", "minimalist", "beige", "wool coat"],
        "color_palette": ["#F5F0EB", "#D4C5B0", "#8B7D6B", "#3D3530", "#1A1512"],
        "velocity": "declining",  # was peak 2023
    },
    "gorpcore": {
        "description": "Outdoor performance gear worn as streetwear; technical fabrics",
        "keywords": ["fleece", "trail runners", "Gore-Tex", "utility vest", "cargo"],
        "color_palette": ["#2D5016", "#8B4513", "#4A4A4A", "#FF6600", "#1C3A5E"],
        "velocity": "rising",
    },
    "dopamine_dressing": {
        "description": "Maximalist color-blocking; bold prints; mood-boosting hues",
        "keywords": ["neon", "color-block", "sequin", "oversized", "print-mixing"],
        "color_palette": ["#FF1744", "#FF9100", "#FFEA00", "#00E676", "#2979FF"],
        "velocity": "peak",
    },
    "coastal_grandmother": {
        "description": "Breezy linen, soft florals, sun-washed tones, relaxed silhouettes",
        "keywords": ["linen", "floral", "wide-leg", "straw hat", "espadrilles"],
        "color_palette": ["#E8DCC8", "#9EC6C0", "#D4956A", "#6B8F71", "#F2E8D9"],
        "velocity": "stable",
    },
    "dark_academia": {
        "description": "Victorian-inspired scholarly aesthetics; plaid, tweed, leather",
        "keywords": ["plaid", "tweed", "oxfords", "blazer", "turtleneck", "corduroy"],
        "color_palette": ["#2C1810", "#4A3728", "#8B6914", "#556B2F", "#1C1C2E"],
        "velocity": "stable",
    },
    "y2k_revival": {
        "description": "Early 2000s nostalgia; low-rise, butterfly clips, metallics",
        "keywords": ["low-rise", "metallic", "mini skirt", "crop top", "rhinestone"],
        "color_palette": ["#FF69B4", "#00CED1", "#9370DB", "#FFD700", "#FF4500"],
        "velocity": "declining",
    },
    "soft_minimalism": {
        "description": "Muted tones, clean lines, anti-trend; Japanese influence",
        "keywords": ["monochrome", "drape", "linen", "wide silhouette", "neutral"],
        "color_palette": ["#F7F3EE", "#E8E0D5", "#C4B7A6", "#9E9289", "#6B6058"],
        "velocity": "rising",
    },
    "biker_aesthetic": {
        "description": "Moto jackets, leather, hardware, edgy femininity",
        "keywords": ["leather jacket", "biker boots", "mesh", "hardware", "black"],
        "color_palette": ["#0D0D0D", "#2C2C2C", "#8B0000", "#B8860B", "#4A4A4A"],
        "velocity": "rising",
    },
    "cottagecore": {
        "description": "Romantic rural fantasy; prairie dresses, natural materials",
        "keywords": ["prairie dress", "puff sleeve", "crochet", "mushroom print", "ditsy"],
        "color_palette": ["#D4B896", "#8FAF7E", "#C17F5A", "#E8D5B7", "#6B8C5A"],
        "velocity": "declining",
    },
    "regencycore": {
        "description": "Bridgerton-inspired empire waists, corsets, pearls, pastels",
        "keywords": ["corset", "empire waist", "pearl", "ruffle", "gloves", "satin"],
        "color_palette": ["#E8C5D4", "#C5D4E8", "#D4C5E8", "#F0E6D3", "#A8C5B8"],
        "velocity": "stable",
    },
}

GARMENT_TYPES = [
    "dress", "blouse", "trousers", "jeans", "skirt", "jacket", "coat",
    "sweater", "hoodie", "cardigan", "top", "bodysuit", "jumpsuit",
    "shorts", "blazer", "vest", "shirt", "leggings", "boots", "sneakers",
    "loafers", "heels", "bag", "belt", "scarf", "hat", "sunglasses",
]

BRANDS = [
    "Zara", "H&M", "Mango", "& Other Stories", "COS", "Arket",
    "Reformation", "Aritzia", "Everlane", "Uniqlo", "Acne Studios",
    "Totême", "The Row", "Bottega Veneta", "Prada", "Loro Piana",
    "Arc'teryx", "Salomon", "Patagonia", "Columbia", "Stone Island",
    "Off-White", "Palm Angels", "Jacquemus", "Cecilie Bahnsen",
    "Simone Rocha", "Jonathan Anderson", "Maison Margiela",
]

PLATFORMS = ["instagram", "pinterest", "tiktok", "vogue_runway", "ssense_editorial", "farfetch"]

SEASONS = ["SS24", "FW24", "SS25", "FW25", "SS26"]

PRICE_RANGES = {
    "budget": (15, 80),
    "mid": (80, 300),
    "premium": (300, 800),
    "luxury": (800, 5000),
}


def generate_synthetic_image_url(item_id: str, trend: str) -> str:
    """Generate a plausible image URL (points to placeholder in real setup)."""
    return f"https://trendai-assets.s3.amazonaws.com/raw/{trend}/{item_id}.jpg"


def generate_caption(garment: str, keywords: list[str], brand: str) -> str:
    templates = [
        f"{brand} {garment} featuring {keywords[0]} and {keywords[1]} details. "
        f"Perfect for achieving the {keywords[2]} aesthetic.",
        f"A {keywords[0]} {garment} from {brand} with {keywords[1]} accents. "
        f"Styled with {keywords[2]} pieces.",
        f"{brand}'s take on the {garment}: {keywords[0]}, {keywords[1]}, and subtly {keywords[2]}.",
    ]
    return random.choice(templates)


def generate_engagement_score(velocity: str, days_ago: int) -> float:
    """Simulate realistic engagement decay/growth curves."""
    base = random.uniform(0.3, 0.9)
    if velocity == "rising":
        # grows over time, recent posts get more engagement
        trend_factor = max(0.2, 1.0 - days_ago / 180)
        return round(base * trend_factor * random.uniform(0.8, 1.4), 4)
    elif velocity == "declining":
        trend_factor = min(1.0, 0.3 + days_ago / 180)
        return round(base * trend_factor * random.uniform(0.5, 0.9), 4)
    elif velocity == "peak":
        return round(base * random.uniform(0.85, 1.0), 4)
    else:  # stable
        return round(base * random.uniform(0.6, 0.95), 4)


def generate_mock_embedding(trend_name: str, dim: int = 768) -> list[float]:
    """
    Generate a deterministic-ish mock CLIP embedding per trend cluster.
    In production this is replaced by actual CLIP inference.
    """
    cluster_seed = sum(ord(c) for c in trend_name)
    rng = np.random.default_rng(cluster_seed)
    center = rng.standard_normal(dim)
    noise = np.random.default_rng().standard_normal(dim) * 0.3
    embedding = center + noise
    # L2 normalize (CLIP outputs are unit-normed)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.tolist()


def build_dataset(n: int = 5000) -> pd.DataFrame:
    records = []
    trend_names = list(TREND_CLUSTERS.keys())
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 3, 1)
    date_range_days = (end_date - start_date).days

    print(f"Generating {n} synthetic fashion items...")

    for _ in tqdm(range(n)):
        trend_name = random.choice(trend_names)
        trend_meta = TREND_CLUSTERS[trend_name]

        item_id = str(uuid.uuid4())[:12]
        garment = random.choice(GARMENT_TYPES)
        brand = random.choice(BRANDS)
        keywords = trend_meta["keywords"]
        platform = random.choice(PLATFORMS)
        season = random.choice(SEASONS)
        price_tier = random.choice(list(PRICE_RANGES.keys()))
        price_lo, price_hi = PRICE_RANGES[price_tier]
        price = round(random.uniform(price_lo, price_hi), 2)

        days_ago = random.randint(0, date_range_days)
        post_date = start_date + timedelta(days=(date_range_days - days_ago))

        caption = generate_caption(garment, keywords, brand)
        engagement = generate_engagement_score(trend_meta["velocity"], days_ago)
        image_url = generate_synthetic_image_url(item_id, trend_name)
        embedding = generate_mock_embedding(trend_name)

        records.append({
            "item_id": item_id,
            "image_url": image_url,
            "garment_type": garment,
            "brand": brand,
            "trend_cluster": trend_name,
            "trend_velocity": trend_meta["velocity"],
            "trend_description": trend_meta["description"],
            "caption": caption,
            "keywords": "|".join(random.sample(keywords, k=min(3, len(keywords)))),
            "platform": platform,
            "season": season,
            "price": price,
            "price_tier": price_tier,
            "post_date": post_date.isoformat(),
            "days_ago": days_ago,
            "engagement_score": engagement,
            "color_primary": random.choice(trend_meta["color_palette"]),
            # Store embedding as comma-separated string for CSV compat
            "embedding_preview": ",".join(f"{x:.4f}" for x in embedding[:8]) + ",...",
            # Full embedding stored separately in production (Parquet column)
            "embedding": embedding,
        })

    return pd.DataFrame(records)


def save_dataset(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parquet (full, including embedding vectors)
    parquet_path = out_dir / "fashion_dataset.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"✓ Saved Parquet: {parquet_path}")

    # CSV (no embedding column — too large)
    csv_df = df.drop(columns=["embedding"])
    csv_path = out_dir / "fashion_dataset.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"✓ Saved CSV: {csv_path}")

    # Dataset card
    card = {
        "name": "TrendAI Synthetic Fashion Dataset",
        "version": "1.0.0",
        "generated_at": datetime.now().isoformat(),
        "n_items": len(df),
        "n_trends": df["trend_cluster"].nunique(),
        "trend_distribution": df["trend_cluster"].value_counts().to_dict(),
        "velocity_distribution": df["trend_velocity"].value_counts().to_dict(),
        "platform_distribution": df["platform"].value_counts().to_dict(),
        "date_range": {
            "start": df["post_date"].min(),
            "end": df["post_date"].max(),
        },
        "embedding_dim": 768,
        "columns": list(df.columns),
        "description": (
            "Synthetic dataset simulating scraped fashion content from social media, "
            "runways, and retail platforms. Embeddings are mock CLIP-style vectors "
            "clustered around 10 trend archetypes. Replace with real CLIP inference "
            "in production."
        ),
    }
    card_path = out_dir / "dataset_card.json"
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2)
    print(f"✓ Saved dataset card: {card_path}")

    # Print summary
    print("\n── Dataset Summary ──────────────────────────────────")
    print(f"  Total items   : {len(df):,}")
    print(f"  Trend clusters: {df['trend_cluster'].nunique()}")
    print(f"  Date range    : {df['post_date'].min()[:10]} → {df['post_date'].max()[:10]}")
    print(f"  Platforms     : {', '.join(df['platform'].unique())}")
    print("\n  Trend distribution:")
    for trend, count in df["trend_cluster"].value_counts().items():
        bar = "█" * (count // 50)
        print(f"    {trend:<25} {count:>5}  {bar}")


if __name__ == "__main__":
    df = build_dataset(n=5000)
    save_dataset(df, Path("data/synthetic"))
