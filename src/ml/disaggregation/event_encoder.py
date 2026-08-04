"""
Sentence-encoder wrapper for event descriptions.

Converts a free-text event description (e.g. "Copa América 2024 Semifinal,
Argentina vs Canada at MetLife Stadium...") into a dense 384-dimensional
embedding using all-MiniLM-L6-v2.  These embeddings are consumed by the
FiLM conditioning layers in EventConditionedGNN.

Model properties (all-MiniLM-L6-v2):
  - ~80 MB on disk; runs in <50 ms per description on CPU
  - 384-dimensional output
  - Pre-trained for semantic similarity on short-to-medium English text

A JSON disk cache is maintained so repeated runs do not re-encode.
On first use the model is downloaded from HuggingFace (~91 MB) and saved
locally under outputs/nyc/all-MiniLM-L6-v2/; subsequent runs load from
disk without network access.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import numpy as np

_DEFAULT_CACHE  = (
    Path(__file__).resolve().parents[3]
    / "outputs" / "nyc" / "event_embeddings_cache.json"
)
_LOCAL_MODEL_DIR = (
    Path(__file__).resolve().parents[3]
    / "outputs" / "nyc" / "all-MiniLM-L6-v2"
)

EMBEDDING_DIM = 384
MODEL_NAME    = "all-MiniLM-L6-v2"


class EventEncoder:
    """
    Thin wrapper around sentence-transformers for event description encoding.

    Basic usage:
        encoder = EventEncoder()
        emb = encoder.encode("FIFA World Cup 2026 Final at MetLife Stadium")
        # emb.shape == (384,)

    Batch usage:
        embs = encoder.encode_batch(["event 1 description", "event 2 description"])
        # embs.shape == (2, 384)

    Embeddings are L2-normalised, so cosine similarity reduces to a dot product.
    """

    def __init__(
        self,
        model_name: str  = MODEL_NAME,
        cache_path: Path = _DEFAULT_CACHE,
        device:     str  = "cpu",
    ):
        self.model_name  = model_name
        self.cache_path  = Path(cache_path)
        self.device      = device
        self._model      = None
        self._cache: dict[str, list] = self._load_cache()

    def _load_model(self):
        """
        Load the sentence-transformers model, preferring the local copy.

        Search order:
          1. _LOCAL_MODEL_DIR (inside the project), no internet required
          2. HuggingFace cache (~/.cache/huggingface/)
          3. Download from HuggingFace Hub (first run only)

        After the first download, the model is saved to _LOCAL_MODEL_DIR so
        that subsequent runs are network-independent.
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. Install with:\n"
                    "    pip install sentence-transformers"
                )

            if _LOCAL_MODEL_DIR.exists():
                print(f"  Loading model from local copy ({_LOCAL_MODEL_DIR.name})...")
                self._model = SentenceTransformer(
                    str(_LOCAL_MODEL_DIR), device=self.device
                )
            else:
                print(f"  Downloading {self.model_name} (~91 MB, first run only)...")
                self._model = SentenceTransformer(self.model_name, device=self.device)
                _LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
                self._model.save(str(_LOCAL_MODEL_DIR))
                print(f"  Model saved to {_LOCAL_MODEL_DIR}")

            print(f"  Encoder ready. Output dimension: {EMBEDDING_DIM}")

        return self._model

    def _load_cache(self) -> dict:
        if self.cache_path.exists():
            with open(self.cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f)

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def encode(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Encode a single event description to a numpy vector.

        Args:
            text:      Free-text event description in English.
            use_cache: If True, return a cached embedding when available.

        Returns:
            np.ndarray of shape (384,), dtype float32, L2-normalised.
        """
        key = self._text_hash(text)

        if use_cache and key in self._cache:
            return np.array(self._cache[key], dtype=np.float32)

        model = self._load_model()
        emb   = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        emb   = emb.astype(np.float32)

        if use_cache:
            self._cache[key] = emb.tolist()
            self._save_cache()

        return emb

    def encode_batch(self, texts: list[str], use_cache: bool = True) -> np.ndarray:
        """
        Encode a list of descriptions, reusing the cache where possible.

        Returns:
            np.ndarray of shape (len(texts), 384), dtype float32.
        """
        results          = []
        to_encode_idx    = []
        to_encode_texts  = []

        for i, text in enumerate(texts):
            key = self._text_hash(text)
            if use_cache and key in self._cache:
                results.append(np.array(self._cache[key], dtype=np.float32))
            else:
                results.append(None)
                to_encode_idx.append(i)
                to_encode_texts.append(text)

        if to_encode_texts:
            model    = self._load_model()
            new_embs = model.encode(
                to_encode_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=len(to_encode_texts) > 5,
            ).astype(np.float32)

            for idx, text, emb in zip(to_encode_idx, to_encode_texts, new_embs):
                results[idx] = emb
                if use_cache:
                    self._cache[self._text_hash(text)] = emb.tolist()

            if use_cache:
                self._save_cache()

        return np.stack(results, axis=0)

    def encode_event_catalog(self, events: list[dict]) -> dict[str, np.ndarray]:
        """
        Encode all events in a catalog in one batch call.

        Args:
            events: List of dicts, each containing 'event_id' and 'description' keys.

        Returns:
            Dict mapping event_id -> embedding (384,).
        """
        texts = [ev["description"] for ev in events]
        ids   = [ev["event_id"]    for ev in events]
        embs  = self.encode_batch(texts)
        return {eid: emb for eid, emb in zip(ids, embs)}

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Cosine similarity between two embeddings.

        Because embeddings are L2-normalised on output, this is equivalent
        to a dot product and runs in O(d) time.
        """
        return float(np.dot(emb1, emb2))

    def most_similar_event(
        self,
        query_text:       str,
        event_embeddings: dict[str, np.ndarray],
    ) -> tuple[str, float]:
        """
        Find the most semantically similar event in a pre-encoded catalog.

        Useful for zero-shot inference: given a WC2026 match description,
        retrieve the Copa América event with the closest semantic profile
        to use as the conditioning embedding.

        Returns:
            (event_id, similarity_score) for the closest match.
        """
        query_emb   = self.encode(query_text)
        best_id     = None
        best_sim    = -1.0
        for eid, emb in event_embeddings.items():
            sim = self.similarity(query_emb, emb)
            if sim > best_sim:
                best_sim = sim
                best_id  = eid
        return best_id, best_sim


if __name__ == "__main__":
    # Quick sanity check: encode two semantically similar and one dissimilar text.
    encoder = EventEncoder()

    texts = [
        "Copa América 2024 Semifinal, Argentina vs Canada at MetLife Stadium.",
        "FIFA World Cup 2026 Semifinal at MetLife Stadium, top-ranked teams.",
        "Taylor Swift Eras Tour concert at MetLife Stadium, evening show.",
    ]
    embs = encoder.encode_batch(texts)
    print(f"Embedding shape: {embs.shape}")

    labels = ["CA2024 SF", "WC2026 SF", "Concert"]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = encoder.similarity(embs[i], embs[j])
            print(f"  {labels[i]} <-> {labels[j]}: {sim:.4f}")
