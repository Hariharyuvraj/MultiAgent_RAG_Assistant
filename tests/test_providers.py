import os
from unittest.mock import patch

import pytest

from providers.llm_factory import get_llm
from providers.embedding_factory import get_embeddings


def _base_config():
    return {
        "llm": {
            "provider": "groq",
            "model_name": "llama-3.1-8b-instant",
            "temperature": 0.2,
            "max_tokens": 512,
            "timeout": 30,
        },
        "embeddings": {
            "provider": "huggingface",
            "model_name": "BAAI/bge-small-en-v1.5",
            "batch_size": 4,
            "device": "cpu",
        },
    }


def test_llm_factory_returns_groq_client():
    from langchain_groq import ChatGroq
    cfg = _base_config()
    with patch.dict(os.environ, {"GROQ_API_KEY": "test-key", "LLM_PROVIDER": "groq"}):
        llm = get_llm(cfg)
    assert isinstance(llm, ChatGroq)


def test_llm_factory_raises_without_key():
    cfg = _base_config()
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("GROQ_API_KEY", None)
        with pytest.raises(EnvironmentError):
            get_llm(cfg)


def test_llm_factory_unsupported_provider():
    cfg = _base_config()
    cfg["llm"]["provider"] = "unknown"
    with patch.dict(os.environ, {"LLM_PROVIDER": "unknown"}):
        with pytest.raises(ValueError):
            get_llm(cfg)


def test_embedding_factory_returns_hf_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    cfg = _base_config()
    with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "huggingface"}):
        emb = get_embeddings(cfg)
    assert isinstance(emb, HuggingFaceEmbeddings)


def test_embedding_dimensions():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    cfg = _base_config()
    with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "huggingface"}):
        emb = get_embeddings(cfg)
    vec = emb.embed_query("test sentence")
    assert isinstance(vec, list)
    assert len(vec) == 384
