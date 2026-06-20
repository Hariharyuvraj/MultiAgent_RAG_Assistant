import logging
from typing import List

import numpy as np
from numpy.linalg import norm

from config import load_config
from providers import get_embeddings
from schemas.state import ChunkResult

logger = logging.getLogger(__name__)

_embeddings = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        cfg = load_config()
        _embeddings = get_embeddings(cfg)
    return _embeddings


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = norm(va) * norm(vb)
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def compute_grounding_score(answer: str, chunks: List[ChunkResult]) -> float:
    if not answer or not chunks:
        logger.debug("Empty answer or chunks — grounding score is 0.0.")
        return 0.0

    emb = _get_embeddings()
    answer_vec = emb.embed_query(answer)
    chunk_texts = [c.content for c in chunks]
    chunk_vecs = emb.embed_documents(chunk_texts)

    similarities = [
        _cosine_similarity(answer_vec, cv) for cv in chunk_vecs
    ]
    score = max(similarities)
    logger.debug(
        "Grounding score: %.4f (max of %d chunk similarities)", score, len(chunks)
    )
    return score
