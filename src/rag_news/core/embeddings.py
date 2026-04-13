from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
import hashlib
import math
import re


class HashingEmbeddingFunction:
    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in input]

    def embed_text(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest()
            index = int(digest, 16) % self.dimensions
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class SemanticEmbeddingFunction:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in input]

    @lru_cache(maxsize=8192)
    def _cached_embed(self, text: str) -> tuple[float, ...]:
        vector = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return tuple(float(value) for value in vector.tolist())

    def embed_text(self, text: str) -> list[float]:
        return list(self._cached_embed(text))
