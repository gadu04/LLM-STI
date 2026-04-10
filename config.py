from __future__ import annotations

from dataclasses import dataclass

LLM_TIMEOUT_SECONDS=90
LLM_TEMPERATURE=0.12
LLM_EXTRACT_MAX_TOKENS=1200
LLM_FALLBACK_ORG_MAX_TOKENS=120
LLM_FALLBACK_PRODUCT_MAX_TOKENS=600
LLM_INPUT_MAX_CHARS=20000
LLM_FALLBACK_CONTEXT_MAX_CHARS=6000

LLM_BASE_URL = "http://localhost:1234/v1"
LLM_API_KEY = "lm-studio"
LLM_MODEL_NAME = "local-model"
LLM_TIMEOUT_SECONDS = 90.0
LLM_TEMPERATURE = 0.1
LLM_INPUT_MAX_CHARS = 20000


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model_name: str
    timeout_seconds: float
    temperature: float
    input_max_chars: int


def get_llm_config() -> LLMConfig:
    return LLMConfig(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model_name=LLM_MODEL_NAME,
        timeout_seconds=LLM_TIMEOUT_SECONDS,
        temperature=LLM_TEMPERATURE,
        input_max_chars=LLM_INPUT_MAX_CHARS,
    )
