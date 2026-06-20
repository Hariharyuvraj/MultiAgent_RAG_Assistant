import logging
import os

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


def get_embeddings(config: dict) -> Embeddings:
    emb_cfg = config["embeddings"]
    provider = os.getenv("EMBEDDING_PROVIDER", emb_cfg["provider"]).lower()
    model_name = emb_cfg["model_name"]
    device = emb_cfg.get("device", "cpu")

    logger.info("Initialising embeddings: provider=%s model=%s", provider, model_name)

    if provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY is not set in the environment.")
        return GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=api_key,
        )

    raise ValueError(
        f"Unsupported embedding provider: '{provider}'. Choose 'huggingface' or 'google'."
    )
