"""Turns text into normalized embedding vectors."""

from typing import Protocol

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim) with L2-normalized rows."""
        ...


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model. Small, fast and runs on a CPU."""

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 64):
        # Imported here so the rest of the package (and the tests) work
        # without the heavy torch dependency.
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,  # so dot product == cosine similarity
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return vectors.astype(np.float32)
