"""LLM factory helper.

Provides a simple interface to create LLM clients for use in nodes.
Students should use this helper so the lab works with any supported provider.

Usage in nodes:
    from .llm import get_llm
    llm = get_llm()
    response = llm.invoke("Hello")
"""

from __future__ import annotations

import os
from typing import Any


def get_llm(model: str | None = None, temperature: float = 0.0) -> Any:  # noqa: ANN401
    """Create an LLM client from environment configuration.

    Checks for API keys in this order:
    1. OPENAI_API_BASE → ChatOpenAI with custom base URL (e.g. localhost)
    2. GEMINI_API_KEY → ChatGoogleGenerativeAI
    3. OPENAI_API_KEY → ChatOpenAI
    4. ANTHROPIC_API_KEY → ChatAnthropic

    Override model with the `model` parameter or LLM_MODEL env var.
    """
    openai_api_base = os.getenv("OPENAI_API_BASE")
    if openai_api_base:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-openai") from exc
        model_name = model or os.getenv("LLM_MODEL") or "gemini/gemini-3.1-flash-lite-preview"
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            base_url=openai_api_base,
            api_key=os.getenv("OPENAI_API_KEY", "mock"),  # type: ignore
        )

    if os.getenv("GEMINI_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-google-genai") from exc
        model_name = model or os.getenv("LLM_MODEL") or "gemini-2.5-flash"
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GEMINI_API_KEY"),  # type: ignore
            temperature=temperature,
        )

    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-openai") from exc
        model_name = model or os.getenv("LLM_MODEL") or "gpt-4o-mini"
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
        )

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-anthropic") from exc
        model_name = model or os.getenv("LLM_MODEL") or "claude-sonnet-4-20250514"
        return ChatAnthropic(
            model=model_name,  # type: ignore
            temperature=temperature,
        )

    raise RuntimeError(
        "No LLM API key found. Set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY in .env\n"
        "See .env.example for configuration."
    )
