"""Tests for the embedding layer and Vertex AI mock interface."""

import numpy as np
import pytest

from src.embeddings import (
    MockVertexEmbeddingModel,
    TextEmbedding,
    embeddings_to_matrix,
)


@pytest.fixture
def model():
    return MockVertexEmbeddingModel()


class TestMockVertexEmbeddingModel:
    def test_returns_correct_number_of_embeddings(self, model):
        texts = ["alpha", "beta", "gamma"]
        result = model.get_embeddings(texts)
        assert len(result) == 3

    def test_each_result_is_text_embedding(self, model):
        result = model.get_embeddings(["hello world"])
        assert isinstance(result[0], TextEmbedding)
        assert isinstance(result[0].values, list)

    def test_embedding_dimension(self, model):
        result = model.get_embeddings(["test text"])
        assert len(result[0].values) == MockVertexEmbeddingModel.EMBEDDING_DIM

    def test_vectors_are_unit_normalised(self, model):
        result = model.get_embeddings(["some text"])
        vec = np.array(result[0].values, dtype=np.float32)
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    def test_same_text_produces_same_vector(self, model):
        r1 = model.get_embeddings(["deterministic"])
        r2 = model.get_embeddings(["deterministic"])
        assert r1[0].values == r2[0].values

    def test_different_texts_produce_different_vectors(self, model):
        r1 = model.get_embeddings(["apple"])
        r2 = model.get_embeddings(["orange"])
        assert r1[0].values != r2[0].values

    def test_empty_input_returns_empty_list(self, model):
        result = model.get_embeddings([])
        assert result == []


class TestEmbeddingsToMatrix:
    def test_shape(self):
        model = MockVertexEmbeddingModel()
        texts = ["a", "b", "c"]
        embs = model.get_embeddings(texts)
        matrix = embeddings_to_matrix(embs)
        assert matrix.shape == (3, MockVertexEmbeddingModel.EMBEDDING_DIM)

    def test_dtype_is_float32(self):
        model = MockVertexEmbeddingModel()
        matrix = embeddings_to_matrix(model.get_embeddings(["x"]))
        assert matrix.dtype == np.float32
