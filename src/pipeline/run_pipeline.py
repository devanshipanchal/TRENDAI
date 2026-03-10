"""
src/pipeline/run_pipeline.py
─────────────────────────────
End-to-end pipeline runner.

Stages:
    1. Ingest        — scrape / load data
    2. Encode        — CLIP embeddings + BLIP captions
    3. Detect        — UMAP + HDBSCAN trend clustering
    4. Forecast      — Claude LLM synthesis + style cards
    5. Persist       — save outputs to disk / S3

Run:
    python -m src.pipeline.run_pipeline [--mode synthetic|live] [--limit 1000]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

console = Console()


# ── Pipeline stages ───────────────────────────────────────────────────────────

def stage_ingest(mode: str, limit: int, output_dir: Path) -> pd.DataFrame:
    """Stage 1: Load or scrape fashion data."""
    from src.pipeline.ingest import IngestPipeline, MockScraper

    console.rule("[bold cyan]Stage 1 — Ingest")

    if mode == "synthetic":
        # Load pre-generated synthetic dataset
        synthetic_path = Path("data/synthetic/fashion_dataset.parquet")
        if not synthetic_path.exists():
            console.print("[yellow]Synthetic dataset not found. Generating...")
            from src.pipeline.generate_dataset import build_dataset, save_dataset
            df = build_dataset(n=limit)
            save_dataset(df, Path("data/synthetic"))
        else:
            df = pd.read_parquet(synthetic_path)
            if limit and limit < len(df):
                df = df.sample(n=limit, random_state=42).reset_index(drop=True)
        console.print(f"[green]✓ Loaded {len(df):,} items from synthetic dataset")
        return df

    elif mode == "mock":
        pipeline = IngestPipeline(
            scrapers=[MockScraper()],
            output_dir=output_dir / "raw",
            download_images=False,
        )
        items = pipeline.run(limit=limit)
        records = [item.to_dict() for item in items]
        df = pd.DataFrame(records)
        df["engagement_score"] = df.apply(
            lambda r: min(1.0, (r.get("likes", 0) + r.get("comments", 0) * 3) / 100_000),
            axis=1,
        )
        console.print(f"[green]✓ Mock-scraped {len(df):,} items")
        return df

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'synthetic' or 'mock'.")


def stage_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 2: Generate CLIP embeddings + captions (uses mock if GPU unavailable)."""
    from src.models.vision_encoder import VisionEncoder

    console.rule("[bold cyan]Stage 2 — Vision Encoding")

    # Check if embeddings already present (synthetic dataset has them)
    if "embedding" in df.columns:
        first = df["embedding"].iloc[0]
        if isinstance(first, (list, np.ndarray)) and len(first) > 100:
            console.print(f"[green]✓ Embeddings already present (dim={len(first)})")
            return df

    encoder = VisionEncoder(mock=True)  # switch mock=False with GPU + real images
    console.print("[yellow]⚡ Generating mock embeddings (set MOCK_VISION=0 for real CLIP)")

    embeddings = []
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task("Encoding images...", total=len(df))
        for _, row in df.iterrows():
            emb = encoder._mock_embedding()
            embeddings.append(emb.tolist())
            progress.advance(task)

    df = df.copy()
    df["embedding"] = embeddings
    console.print(f"[green]✓ Encoded {len(df):,} items → 768-dim embeddings")
    return df


def stage_detect(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list]:
    """Stage 3: Cluster embeddings → trend reports."""
    from src.pipeline.trend_engine import TrendEngine

    console.rule("[bold cyan]Stage 3 — Trend Detection")

    engine = TrendEngine(
        umap_n_components=config.get("umap_n_components", 50),
        min_cluster_size=config.get("min_cluster_size", 10),
        min_samples=config.get("min_samples", 5),
        trend_window_days=config.get("trend_window_days", 30),
        top_k=config.get("top_k", 15),
    )

    with console.status("Running UMAP + HDBSCAN..."):
        trend_reports = engine.detect_trends(df)

    console.print(f"[green]✓ Detected {len(trend_reports)} trend clusters")

    # Print summary table
    table = Table(title="Top Trend Clusters", show_header=True, header_style="bold magenta")
    table.add_column("Cluster", style="cyan", width=6)
    table.add_column("Name", width=25)
    table.add_column("Items", justify="right")
    table.add_column("Velocity", width=10)
    table.add_column("Score", justify="right")
    table.add_column("Platform")

    for r in trend_reports[:10]:
        velocity_color = {
            "rising": "green", "peak": "yellow",
            "stable": "blue", "declining": "red",
        }.get(r.velocity, "white")
        table.add_row(
            str(r.cluster_id),
            r.trend_name[:24],
            str(r.item_count),
            f"[{velocity_color}]{r.velocity}[/{velocity_color}]",
            f"{r.velocity_score:+.3f}",
            r.dominant_platform,
        )

    console.print(table)
    return df, trend_reports


def stage_forecast(trend_reports: list, config: dict) -> object:
    """Stage 4: LLM synthesis → ForecastReport."""
    from src.agents.forecast_agent import run_forecast

    console.rule("[bold cyan]Stage 4 — LLM Forecast Synthesis")

    if not config.get("anthropic_api_key") and not __import__("os").getenv("ANTHROPIC_API_KEY"):
        console.print("[yellow]⚠ ANTHROPIC_API_KEY not set. Skipping LLM synthesis.")
        console.print("[dim]Set ANTHROPIC_API_KEY in .env to enable forecast generation.")
        return None

    with console.status("[bold]Calling Claude for trend synthesis..."):
        report = run_forecast(
            trend_reports,
            analysis_window_days=config.get("trend_window_days", 30),
        )

    console.print(f"[green]✓ Generated {len(report.style_cards)} style cards")
    console.print(Panel(
        report.executive_summary or "No summary generated.",
        title="[bold]Executive Summary",
        border_style="green",
    ))
    return report


def stage_persist(
    df: pd.DataFrame,
    trend_reports: list,
    forecast_report: object,
    output_dir: Path,
) -> None:
    """Stage 5: Save all outputs."""
    console.rule("[bold cyan]Stage 5 — Persist Outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save enriched dataframe
    df_save = df.copy()
    if "embedding" in df_save.columns:
        df_save["embedding_preview"] = df_save["embedding"].apply(
            lambda e: ",".join(f"{x:.4f}" for x in e[:8]) + ",...") \
            if hasattr(df_save["embedding"].iloc[0], "__len__") else "n/a"
        df_save = df_save.drop(columns=["embedding"])
    df_save.to_csv(output_dir / "enriched_items.csv", index=False)
    console.print(f"[green]✓ Saved enriched_items.csv")

    # Save trend reports
    def _report_to_dict(r):
        import dataclasses
        if dataclasses.is_dataclass(r):
            d = dataclasses.asdict(r)
            d.pop("raw_signals", None)
            d.pop("embedding", None)
            return d
        return r

    trends_data = [_report_to_dict(r) for r in trend_reports]
    with open(output_dir / "trend_reports.json", "w") as f:
        json.dump(trends_data, f, indent=2, default=str)
    console.print(f"[green]✓ Saved trend_reports.json")

    # Save forecast
    if forecast_report:
        with open(output_dir / "forecast_report.json", "w") as f:
            json.dump(forecast_report.to_dict(), f, indent=2, default=str)
        console.print(f"[green]✓ Saved forecast_report.json")

    # Run summary
    console.print(Panel(
        f"Output directory: [bold]{output_dir.resolve()}[/bold]\n"
        f"Items processed: [bold]{len(df):,}[/bold]\n"
        f"Trend clusters: [bold]{len(trend_reports)}[/bold]\n"
        f"Style cards: [bold]{len(forecast_report.style_cards) if forecast_report else 0}[/bold]",
        title="[bold green]✓ Pipeline Complete",
        border_style="green",
    ))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TrendAI End-to-End Pipeline")
    parser.add_argument("--mode", choices=["synthetic", "mock", "live"], default="synthetic")
    parser.add_argument("--limit", type=int, default=2000, help="Max items to process")
    parser.add_argument("--output", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--skip-forecast", action="store_true", help="Skip LLM synthesis step")
    args = parser.parse_args()

    # Load .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    import os
    config = {
        "umap_n_components": int(os.getenv("UMAP_N_COMPONENTS", "50")),
        "min_cluster_size": int(os.getenv("MIN_CLUSTER_SIZE", "10")),
        "min_samples": int(os.getenv("MIN_SAMPLES", "5")),
        "trend_window_days": int(os.getenv("TREND_WINDOW_DAYS", "30")),
        "top_k": int(os.getenv("TOP_K_TRENDS", "15")),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
    }

    output_dir = Path(args.output)
    t0 = time.time()

    console.print(Panel(
        "[bold]TrendAI — GenAI Fashion Trend Analyzer[/bold]\n"
        f"Mode: {args.mode} | Limit: {args.limit:,} | Output: {output_dir}",
        border_style="magenta",
    ))

    # Run pipeline stages
    df = stage_ingest(args.mode, args.limit, output_dir)
    df = stage_encode(df)
    df, trend_reports = stage_detect(df, config)

    forecast_report = None
    if not args.skip_forecast:
        forecast_report = stage_forecast(trend_reports, config)

    stage_persist(df, trend_reports, forecast_report, output_dir)

    elapsed = time.time() - t0
    console.print(f"\n[dim]Total time: {elapsed:.1f}s[/dim]")


if __name__ == "__main__":
    main()
