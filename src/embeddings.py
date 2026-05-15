"""
Embedding layer: local sentence-transformers model + Vertex AI mock.

The mock mirrors the real vertexai.language_models.TextEmbeddingModel interface
so the rest of the pipeline is swappable between local and GCP without changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Vertex AI interface contract (matches the real SDK shape)
# ---------------------------------------------------------------------------

@dataclass
class TextEmbedding:
    """Mirrors vertexai.language_models.TextEmbedding."""
    values: list[float]
    statistics: dict | None = None


@runtime_checkable
class EmbeddingModelProtocol(Protocol):
    def get_embeddings(self, texts: list[str]) -> list[TextEmbedding]:
        ...


# ---------------------------------------------------------------------------
# Local model (sentence-transformers — closest public approx to gecko)
# ---------------------------------------------------------------------------

class LocalEmbeddingModel:
    """
    Wraps all-MiniLM-L6-v2 behind the same interface as Vertex AI's
    textembedding-gecko so the pipeline is backend-agnostic.

    Vectors are L2-normalised here so dot-product == cosine similarity.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    def __init__(self, model_name: str = MODEL_NAME):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for LocalEmbeddingModel. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model_name)

    def get_embeddings(self, texts: list[str]) -> list[TextEmbedding]:
        if not texts:
            return []
        raw: np.ndarray = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalise → cosine via dot
            show_progress_bar=False,
        )
        return [TextEmbedding(values=row.tolist()) for row in raw]

    @property
    def embedding_dim(self) -> int:
        return self.EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Mock — injectable for tests and GCP-cost-free local runs
# ---------------------------------------------------------------------------

class MockVertexEmbeddingModel:
    """
    Deterministic mock of vertexai.language_models.TextEmbeddingModel.

    Uses a seeded RNG keyed on the text hash so the same text always
    produces the same vector — critical for reproducible benchmark results.
    """

    EMBEDDING_DIM = 384

    def get_embeddings(self, texts: list[str]) -> list[TextEmbedding]:
        results = []
        for text in texts:
            seed = hash(text) % (2**31)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.EMBEDDING_DIM).astype(np.float32)
            vec /= np.linalg.norm(vec)          # normalise
            results.append(TextEmbedding(values=vec.tolist()))
        return results

    @property
    def embedding_dim(self) -> int:
        return self.EMBEDDING_DIM


def embeddings_to_matrix(embeddings: list[TextEmbedding]) -> np.ndarray:
    """Convert a list of TextEmbedding objects to a float32 numpy matrix."""
    return np.array([e.values for e in embeddings], dtype=np.float32)
