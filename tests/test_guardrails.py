import os
from unittest.mock import MagicMock, patch

import pytest

from guardrails.grounding_check import compute_grounding_score, _cosine_similarity
from schemas.state import ChunkResult


def _make_chunk(content: str, score: float = 0.8) -> ChunkResult:
    return ChunkResult(content=content, source="test.pdf", page=1, score=score)


def test_cosine_similarity_identical_vectors():
    v = [1.0, 0.0, 0.0]
    assert _cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert _cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector():
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_grounding_score_empty_answer():
    chunks = [_make_chunk("some content")]
    score = compute_grounding_score("", chunks)
    assert score == 0.0


def test_grounding_score_empty_chunks():
    score = compute_grounding_score("some answer", [])
    assert score == 0.0


def test_grounding_score_range():
    mock_emb = MagicMock()
    mock_emb.embed_query.return_value = [1.0, 0.0, 0.0]
    mock_emb.embed_documents.return_value = [[0.9, 0.1, 0.0], [0.5, 0.5, 0.0]]

    with patch("guardrails.grounding_check._get_embeddings", return_value=mock_emb):
        chunks = [_make_chunk("doc chunk one"), _make_chunk("doc chunk two")]
        score = compute_grounding_score("test answer", chunks)

    assert 0.0 <= score <= 1.0


def test_high_similarity_scores_high():
    mock_emb = MagicMock()
    vec = [1.0, 0.0, 0.0]
    mock_emb.embed_query.return_value = vec
    mock_emb.embed_documents.return_value = [vec]

    with patch("guardrails.grounding_check._get_embeddings", return_value=mock_emb):
        score = compute_grounding_score("answer", [_make_chunk("chunk")])

    assert score >= 0.99


def test_unrelated_answer_scores_low():
    mock_emb = MagicMock()
    mock_emb.embed_query.return_value = [1.0, 0.0]
    mock_emb.embed_documents.return_value = [[0.0, 1.0]]

    with patch("guardrails.grounding_check._get_embeddings", return_value=mock_emb):
        score = compute_grounding_score("cars", [_make_chunk("AI research")])

    assert score < 0.1
