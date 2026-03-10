"""
src/models/vision_encoder.py
─────────────────────────────
Wraps OpenAI CLIP (ViT-L/14) for image embeddings and BLIP-2 for
auto-captioning. Falls back to mock embeddings if GPU not available.

Usage:
    encoder = VisionEncoder()
    result = encoder.encode_image(PIL_image)
    # result.embedding: np.ndarray (768,)
    # result.caption: str
    # result.attributes: dict
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# ── Lazy imports (heavy libraries) ───────────────────────────────────────────

def _try_import_clip():
    try:
        import open_clip
        return open_clip
    except ImportError:
        warnings.warn("open_clip not installed. Using mock embeddings.", stacklevel=2)
        return None

def _try_import_transformers():
    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        return Blip2Processor, Blip2ForConditionalGeneration
    except ImportError:
        return None, None


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class EncodingResult:
    embedding: np.ndarray          # CLIP embedding (768-dim, L2-normalized)
    caption: str                   # BLIP-2 generated caption
    attributes: dict               # Parsed garment attributes
    confidence: float = 1.0
    model_version: str = "ViT-L-14"


@dataclass
class BatchEncodingResult:
    embeddings: np.ndarray         # (N, 768)
    captions: list[str]
    attributes: list[dict]
    item_ids: list[str] = field(default_factory=list)


# ── Attribute Extractor (rule-based from caption) ────────────────────────────

GARMENT_VOCAB = {
    "tops": ["blouse", "shirt", "top", "tee", "bodysuit", "corset", "sweater", "hoodie", "cardigan"],
    "bottoms": ["trousers", "pants", "jeans", "skirt", "shorts", "leggings"],
    "outerwear": ["jacket", "coat", "blazer", "vest", "anorak"],
    "dresses": ["dress", "gown", "jumpsuit", "romper"],
    "footwear": ["boots", "sneakers", "heels", "loafers", "sandals", "mules"],
    "accessories": ["bag", "hat", "scarf", "belt", "sunglasses", "jewelry"],
}

COLOR_VOCAB = {
    "neutrals": ["beige", "cream", "ivory", "white", "black", "grey", "gray", "tan", "nude"],
    "earth_tones": ["brown", "rust", "terracotta", "camel", "olive", "khaki"],
    "pastels": ["blush", "lavender", "mint", "powder blue", "lilac", "peach"],
    "bold": ["red", "cobalt", "emerald", "fuchsia", "orange", "yellow", "neon"],
    "metallics": ["gold", "silver", "bronze", "copper", "metallic"],
}

FABRIC_VOCAB = [
    "cotton", "linen", "silk", "satin", "wool", "cashmere", "leather",
    "denim", "velvet", "chiffon", "knit", "crochet", "mesh", "tweed",
    "fleece", "gore-tex", "nylon", "polyester", "viscose",
]


def extract_attributes(caption: str) -> dict:
    """Rule-based attribute extraction from generated caption."""
    caption_lower = caption.lower()

    garment_category = "other"
    for category, words in GARMENT_VOCAB.items():
        if any(w in caption_lower for w in words):
            garment_category = category
            break

    color_family = "other"
    for family, words in COLOR_VOCAB.items():
        if any(w in caption_lower for w in words):
            color_family = family
            break

    detected_fabrics = [f for f in FABRIC_VOCAB if f in caption_lower]

    return {
        "garment_category": garment_category,
        "color_family": color_family,
        "fabrics": detected_fabrics[:3],
        "is_layered": any(w in caption_lower for w in ["layered", "over", "under", "paired"]),
        "has_print": any(w in caption_lower for w in ["print", "pattern", "stripe", "check", "floral"]),
        "formality": (
            "formal" if any(w in caption_lower for w in ["gown", "suit", "blazer", "tailored"])
            else "casual" if any(w in caption_lower for w in ["hoodie", "jeans", "sneakers", "tee"])
            else "smart_casual"
        ),
    }


# ── Main Encoder ─────────────────────────────────────────────────────────────

class VisionEncoder:
    """
    Multimodal vision encoder combining CLIP + BLIP-2.

    In production: loads real models onto GPU.
    In CI/testing:  MOCK_VISION=1 env var returns deterministic fake outputs.
    """

    CLIP_MODEL = os.getenv("CLIP_MODEL", "ViT-L-14")
    CLIP_PRETRAINED = os.getenv("CLIP_PRETRAINED", "openai")
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

    def __init__(self, device: Optional[str] = None, mock: bool = False):
        self.mock = mock or os.getenv("MOCK_VISION", "0") == "1"
        self.device = device or self._auto_device()
        self._clip_model = None
        self._clip_preprocess = None
        self._blip_processor = None
        self._blip_model = None

        if not self.mock:
            self._load_clip()
            self._load_blip()

    def _auto_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _load_clip(self) -> None:
        open_clip = _try_import_clip()
        if open_clip is None:
            self.mock = True
            return
        try:
            import torch
            self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
                self.CLIP_MODEL, pretrained=self.CLIP_PRETRAINED
            )
            self._clip_model = self._clip_model.to(self.device).eval()
            print(f"✓ CLIP loaded: {self.CLIP_MODEL} on {self.device}")
        except Exception as e:
            warnings.warn(f"Failed to load CLIP: {e}. Using mock.", stacklevel=2)
            self.mock = True

    def _load_blip(self) -> None:
        Blip2Processor, Blip2Model = _try_import_transformers()
        if Blip2Processor is None:
            return
        try:
            self._blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self._blip_model = Blip2Model.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                load_in_8bit=True,   # requires bitsandbytes
                device_map="auto",
            )
            print("✓ BLIP-2 loaded (8-bit quantized)")
        except Exception as e:
            warnings.warn(f"BLIP-2 not loaded: {e}. Captions will be placeholder.", stacklevel=2)

    # ── Public API ────────────────────────────────────────────────────────────

    def encode_image(self, image: Image.Image, prompt: str = "") -> EncodingResult:
        """Encode a single PIL image → embedding + caption + attributes."""
        embedding = self._get_clip_embedding(image)
        caption = self._get_blip_caption(image) or prompt or "Fashion item."
        attributes = extract_attributes(caption)
        return EncodingResult(
            embedding=embedding,
            caption=caption,
            attributes=attributes,
        )

    def encode_batch(
        self,
        images: list[Image.Image],
        item_ids: Optional[list[str]] = None,
        batch_size: int = 32,
    ) -> BatchEncodingResult:
        """Encode a list of PIL images in batches."""
        all_embeddings = []
        all_captions = []
        all_attributes = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            embeddings = self._get_clip_batch(batch)
            captions = [self._get_blip_caption(img) or "Fashion item." for img in batch]
            attributes = [extract_attributes(c) for c in captions]
            all_embeddings.append(embeddings)
            all_captions.extend(captions)
            all_attributes.extend(attributes)

        return BatchEncodingResult(
            embeddings=np.vstack(all_embeddings),
            captions=all_captions,
            attributes=all_attributes,
            item_ids=item_ids or [],
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_clip_embedding(self, image: Image.Image) -> np.ndarray:
        if self.mock or self._clip_model is None:
            return self._mock_embedding()
        try:
            import torch
            tensor = self._clip_preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self._clip_model.encode_image(tensor)
                features = features / features.norm(dim=-1, keepdim=True)
            return features.squeeze().cpu().numpy()
        except Exception:
            return self._mock_embedding()

    def _get_clip_batch(self, images: list[Image.Image]) -> np.ndarray:
        if self.mock or self._clip_model is None:
            return np.stack([self._mock_embedding() for _ in images])
        try:
            import torch
            tensors = torch.stack([self._clip_preprocess(img) for img in images]).to(self.device)
            with torch.no_grad():
                features = self._clip_model.encode_image(tensors)
                features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy()
        except Exception:
            return np.stack([self._mock_embedding() for _ in images])

    def _get_blip_caption(self, image: Image.Image) -> Optional[str]:
        if self._blip_processor is None or self._blip_model is None:
            return None
        try:
            import torch
            inputs = self._blip_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated = self._blip_model.generate(**inputs, max_new_tokens=60)
            return self._blip_processor.decode(generated[0], skip_special_tokens=True)
        except Exception:
            return None

    def _mock_embedding(self) -> np.ndarray:
        """Reproducible random unit vector for testing."""
        vec = np.random.default_rng().standard_normal(self.EMBEDDING_DIM).astype(np.float32)
        return vec / np.linalg.norm(vec)


# ── Text Encoder (for trend queries) ─────────────────────────────────────────

class TextEncoder:
    """
    Encode text prompts with CLIP's text tower.
    Used for semantic trend search: 'show me quiet luxury'.
    """

    def __init__(self, vision_encoder: Optional[VisionEncoder] = None):
        self._encoder = vision_encoder or VisionEncoder()

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a text string → 768-dim embedding."""
        if self._encoder.mock or self._encoder._clip_model is None:
            return self._encoder._mock_embedding()
        try:
            import open_clip
            import torch
            tokenizer = open_clip.get_tokenizer(VisionEncoder.CLIP_MODEL)
            tokens = tokenizer([text]).to(self._encoder.device)
            with torch.no_grad():
                features = self._encoder._clip_model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
            return features.squeeze().cpu().numpy()
        except Exception:
            return self._encoder._mock_embedding()

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.encode_query(t) for t in texts])
