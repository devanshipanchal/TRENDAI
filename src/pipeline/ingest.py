"""
src/pipeline/ingest.py
───────────────────────
Data ingestion layer. Supports:
  - Local filesystem (for dataset files)
  - S3/MinIO (for production)
  - Instagram hashtag scraping (requires instagrapi)
  - Pinterest board scraping (requires requests + API token)
  - Mock mode for testing

All scrapers normalize to the same FashionItem schema.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Iterator, Optional
import warnings

import httpx
from PIL import Image


# ── Normalized item schema ────────────────────────────────────────────────────

@dataclass
class FashionItem:
    item_id: str
    source: str                  # instagram | pinterest | runway | catalogue | local
    image_url: str
    caption: str
    post_date: str               # ISO8601
    platform: str
    author_handle: str
    likes: int
    comments: int
    hashtags: list[str]
    brand: Optional[str] = None
    product_url: Optional[str] = None
    price: Optional[float] = None
    season: Optional[str] = None
    local_image_path: Optional[str] = None

    def engagement_score(self, max_likes: int = 100_000) -> float:
        """Normalized engagement [0, 1]."""
        raw = self.likes + self.comments * 3
        return min(1.0, raw / max_likes)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Base scraper ──────────────────────────────────────────────────────────────

class BaseScraper:
    def scrape(self, limit: int = 100) -> Iterator[FashionItem]:
        raise NotImplementedError

    async def scrape_async(self, limit: int = 100) -> AsyncIterator[FashionItem]:
        raise NotImplementedError


# ── Mock scraper (for testing without credentials) ────────────────────────────

MOCK_HASHTAGS = [
    "#quietluxury", "#gorpcore", "#dopaminedressing",
    "#darkacademia", "#y2kfashion", "#cottagecore", "#regencycore",
    "#softminimalism", "#biker", "#coastalgrandmother",
]

MOCK_CAPTIONS = [
    "Effortless Sunday dressing — cashmere and tailored trousers",
    "Trail runners with everything this season 🏔️",
    "Color blocking is my love language 💛❤️💙",
    "Tweed blazer found at a vintage market, paired with oxfords",
    "Low rise jeans are back and I'm not complaining",
    "Linen sets for slow mornings ☀️",
    "Leather jacket era starts now 🖤",
    "Prairie dress for the win 🌸",
    "Corset + midi skirt — Bridgerton but make it 2025",
    "Monochrome grey everything",
]


class MockScraper(BaseScraper):
    """Returns synthetic FashionItem objects. Zero API calls."""

    def scrape(self, limit: int = 100) -> Iterator[FashionItem]:
        import random
        rng = random.Random(42)
        for i in range(limit):
            tag = rng.choice(MOCK_HASHTAGS)
            yield FashionItem(
                item_id=str(uuid.uuid4())[:12],
                source="mock",
                image_url=f"https://picsum.photos/seed/{i}/400/600",
                caption=rng.choice(MOCK_CAPTIONS),
                post_date=datetime.now().replace(
                    day=rng.randint(1, 28),
                    month=rng.randint(1, 12),
                ).isoformat(),
                platform="instagram",
                author_handle=f"@user_{rng.randint(1000, 9999)}",
                likes=rng.randint(100, 50000),
                comments=rng.randint(10, 2000),
                hashtags=[tag, "#fashion", "#ootd"],
            )


# ── Instagram scraper ─────────────────────────────────────────────────────────

class InstagramScraper(BaseScraper):
    """
    Scrapes hashtag feeds using instagrapi.
    Requires: pip install instagrapi
    Requires: INSTAGRAM_USERNAME + INSTAGRAM_PASSWORD env vars.
    """

    def __init__(self, hashtags: list[str]):
        self.hashtags = hashtags
        self._client = None

    def _get_client(self):
        if self._client:
            return self._client
        try:
            from instagrapi import Client
            cl = Client()
            cl.login(
                os.environ["INSTAGRAM_USERNAME"],
                os.environ["INSTAGRAM_PASSWORD"],
            )
            self._client = cl
            return cl
        except ImportError:
            raise RuntimeError("instagrapi not installed: pip install instagrapi")
        except KeyError:
            raise RuntimeError("Set INSTAGRAM_USERNAME and INSTAGRAM_PASSWORD env vars")

    def scrape(self, limit: int = 100) -> Iterator[FashionItem]:
        cl = self._get_client()
        per_hashtag = limit // len(self.hashtags)

        for tag in self.hashtags:
            try:
                medias = cl.hashtag_medias_recent(tag, amount=per_hashtag)
                for media in medias:
                    image_url = str(media.thumbnail_url or media.resources[0].thumbnail_url
                                    if media.resources else "")
                    yield FashionItem(
                        item_id=str(media.pk),
                        source="instagram",
                        image_url=image_url,
                        caption=media.caption_text or "",
                        post_date=media.taken_at.isoformat(),
                        platform="instagram",
                        author_handle=f"@{media.user.username}",
                        likes=media.like_count,
                        comments=media.comment_count,
                        hashtags=[f"#{tag}"] + [f"#{t}" for t in (media.caption_text or "").split("#")[1:]][:10],
                    )
            except Exception as e:
                warnings.warn(f"Instagram scrape error for #{tag}: {e}", stacklevel=2)


# ── Pinterest scraper ─────────────────────────────────────────────────────────

class PinterestScraper(BaseScraper):
    """
    Scrapes Pinterest boards/searches via REST API v5.
    Requires: PINTEREST_ACCESS_TOKEN env var.
    """

    BASE_URL = "https://api.pinterest.com/v5"

    def __init__(self, query: str):
        self.query = query
        self.token = os.getenv("PINTEREST_ACCESS_TOKEN", "")

    def scrape(self, limit: int = 100) -> Iterator[FashionItem]:
        if not self.token:
            warnings.warn("PINTEREST_ACCESS_TOKEN not set.", stacklevel=2)
            return

        headers = {"Authorization": f"Bearer {self.token}"}
        params = {"query": self.query, "page_size": min(limit, 25)}

        with httpx.Client() as client:
            fetched = 0
            bookmark = None
            while fetched < limit:
                if bookmark:
                    params["bookmark"] = bookmark
                try:
                    resp = client.get(
                        f"{self.BASE_URL}/pins/search",
                        headers=headers,
                        params=params,
                        timeout=10,
                    )
                    data = resp.json()
                except Exception as e:
                    warnings.warn(f"Pinterest API error: {e}", stacklevel=2)
                    break

                for pin in data.get("items", []):
                    img = pin.get("media", {}).get("images", {}).get("originals", {}).get("url", "")
                    yield FashionItem(
                        item_id=pin.get("id", str(uuid.uuid4())[:12]),
                        source="pinterest",
                        image_url=img,
                        caption=pin.get("description", ""),
                        post_date=pin.get("created_at", datetime.now().isoformat()),
                        platform="pinterest",
                        author_handle=pin.get("owner_name", ""),
                        likes=pin.get("save_count", 0),
                        comments=0,
                        hashtags=[],
                        product_url=pin.get("link", None),
                    )
                    fetched += 1
                    if fetched >= limit:
                        break

                bookmark = data.get("bookmark")
                if not bookmark:
                    break


# ── Image downloader ─────────────────────────────────────────────────────────

async def download_image(
    url: str, save_path: Path, client: httpx.AsyncClient
) -> Optional[Path]:
    """Download a single image asynchronously."""
    try:
        resp = await client.get(url, timeout=15, follow_redirects=True)
        resp.raise_for_status()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(resp.content)
        return save_path
    except Exception:
        return None


async def batch_download_images(
    items: list[FashionItem],
    output_dir: Path,
    concurrency: int = 10,
) -> dict[str, Path]:
    """Download images for a batch of FashionItems. Returns item_id → local path."""
    semaphore = asyncio.Semaphore(concurrency)
    results: dict[str, Path] = {}

    async def _download(item: FashionItem):
        ext = item.image_url.split(".")[-1].split("?")[0][:4] or "jpg"
        save_path = output_dir / f"{item.item_id}.{ext}"
        async with semaphore:
            async with httpx.AsyncClient() as client:
                path = await download_image(item.image_url, save_path, client)
                if path:
                    results[item.item_id] = path

    await asyncio.gather(*[_download(item) for item in items])
    return results


# ── Ingest pipeline entry point ───────────────────────────────────────────────

class IngestPipeline:
    """
    Orchestrates scraping → downloading → normalizing.

    Usage:
        pipeline = IngestPipeline(scrapers=[MockScraper()], output_dir=Path("data/raw"))
        items = pipeline.run(limit=500)
    """

    def __init__(
        self,
        scrapers: Optional[list[BaseScraper]] = None,
        output_dir: Path = Path("data/raw"),
        download_images: bool = True,
    ):
        self.scrapers = scrapers or [MockScraper()]
        self.output_dir = output_dir
        self.download_images = download_images

    def run(self, limit: int = 500) -> list[FashionItem]:
        """Synchronous run: scrape all sources, return normalized items."""
        all_items: list[FashionItem] = []
        per_scraper = limit // len(self.scrapers)

        for scraper in self.scrapers:
            for item in scraper.scrape(limit=per_scraper):
                all_items.append(item)

        if self.download_images:
            asyncio.run(
                batch_download_images(all_items, self.output_dir / "images")
            )

        # Save manifest
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.output_dir / "manifest.jsonl"
        with open(manifest_path, "w") as f:
            for item in all_items:
                f.write(json.dumps(item.to_dict()) + "\n")

        print(f"✓ Ingested {len(all_items)} items → {manifest_path}")
        return all_items
