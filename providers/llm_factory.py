import logging
import os
from typing import Any

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


def get_llm(config: dict) -> BaseChatModel:
    llm_cfg = config["llm"]
    provider = os.getenv("LLM_PROVIDER", llm_cfg["provider"]).lower()
    model_name = llm_cfg["model_name"]
    temperature = float(llm_cfg.get("temperature", 0.2))
    max_tokens = int(llm_cfg.get("max_tokens", 2048))
    timeout = int(llm_cfg.get("timeout", 60))

    logger.info("Initialising LLM: provider=%s model=%s", provider, model_name)

    if provider == "groq":
        from langchain_groq import ChatGroq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is not set in the environment.")
        return ChatGroq(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY is not set in the environment.")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    raise ValueError(f"Unsupported LLM provider: '{provider}'. Choose 'groq' or 'gemini'.")
