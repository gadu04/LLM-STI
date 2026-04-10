from __future__ import annotations

from openai import OpenAI

from config import get_llm_config


def create_llm_client(timeout_seconds: float | None = None) -> OpenAI:
    """Create OpenAI-compatible client for LM Studio local endpoint."""
    llm_config = get_llm_config()
    resolved_timeout = timeout_seconds if timeout_seconds is not None else llm_config.timeout_seconds
    return OpenAI(
        base_url=llm_config.base_url,
        api_key=llm_config.api_key,
        timeout=resolved_timeout,
    )
