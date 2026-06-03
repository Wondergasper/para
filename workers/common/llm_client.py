"""
llm_client.py
-------------
Provider-agnostic LLM client using the OpenAI-compatible SDK.
Supports: Ollama (local), Groq (cloud), Gemini (cloud).

Usage:
    from llm_client import generate

    result = generate(
        prompt="What is OpenMP?",
        system="You are an HPC expert.",
        provider="ollama",           # "ollama" | "groq" | "gemini"
        model="deepseek-coder:6.7b", # any model your provider supports
        temp=0.2,
    )
    print(result)
"""

import os


# ── Provider configurations ────────────────────────────────────────────────────

PROVIDERS = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key":  "ollama",           # Ollama doesn't need a real key
        "default_model": "deepseek-r1:8b",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key":  os.getenv("GROQ_API_KEY", ""),
        "default_model": "llama3-8b-8192",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key":  os.getenv("GEMINI_API_KEY", ""),
        "default_model": "gemini-1.5-flash",
    },
}


# ── Client factory ─────────────────────────────────────────────────────────────

def get_client(provider: str = "ollama"):
    """
    Return an OpenAI-compatible client for the given provider.
    Raises ValueError if the provider is unknown.
    """
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Choose from: {list(PROVIDERS.keys())}"
        )

    cfg = PROVIDERS[provider]

    if not cfg["api_key"]:
        raise EnvironmentError(
            f"API key for '{provider}' is missing. "
            f"Set the corresponding environment variable."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise EnvironmentError(
            "The openai package is required for LLM calls. "
            "Install Phase 1 dependencies with: pip install openai"
        ) from exc

    return OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])


# ── Main generate function ─────────────────────────────────────────────────────

def generate(
    prompt:   str,
    system:   str,
    provider: str   = "ollama",
    model:    str   = None,
    temp:     float = 0.2,
    max_tokens: int = 2048,
) -> str:
    """
    Send a prompt to the LLM and return the response text.

    Args:
        prompt:     The user message / task description.
        system:     The system prompt (defines the LLM's role/rules).
        provider:   "ollama" | "groq" | "gemini"
        model:      Model name. If None, uses the provider's default.
        temp:       Temperature. Lower = more deterministic (use 0.1–0.2 for code).
        max_tokens: Maximum tokens in the response.

    Returns:
        The model's response as a plain string.

    Raises:
        EnvironmentError: If the API key is missing.
        ValueError:       If the provider is unknown.
        RuntimeError:     If the API call fails.
    """
    client = get_client(provider)

    # Use provider default model if none given
    if model is None:
        model = PROVIDERS[provider]["default_model"]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",  "content": system},
                {"role": "user",    "content": prompt},
            ],
            temperature=temp,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    except Exception as e:
        raise RuntimeError(f"LLM call failed [{provider}/{model}]: {e}") from e


def generate_batch(
    prompt:   str,
    system:   str,
    provider: str   = "ollama",
    model:    str   = None,
    temp:     float = 0.2,
    max_tokens: int = 2048,
    n:        int = 3,
) -> list[str]:
    """
    Send a prompt to the LLM requesting n choices in parallel.
    Falls back to single-generation loop if n is unsupported.
    """
    client = get_client(provider)
    if model is None:
        model = PROVIDERS[provider]["default_model"]

    choices = []
    remaining = n

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",  "content": system},
                {"role": "user",    "content": prompt},
            ],
            temperature=temp,
            max_tokens=max_tokens,
            n=n,
        )
        choices = [choice.message.content for choice in response.choices if choice.message.content]
        if len(choices) >= n:
            return choices
        remaining = n - len(choices)
    except Exception:
        # Fallback to sequential generation with varying temperatures
        choices = []
        remaining = n

    for i in range(remaining):
        # Slightly alter the temperature to encourage candidate variation
        new_temp = min(1.0, max(0.0, temp + (i + 1) * 0.15))
        try:
            res = generate(prompt, system, provider, model, temp=new_temp, max_tokens=max_tokens)
            if res:
                choices.append(res)
        except Exception:
            pass

    return choices


# ── Quick smoke-test (run this file directly to verify setup) ──────────────────

if __name__ == "__main__":
    print("Testing Ollama connection...")
    try:
        reply = generate(
            prompt="Say 'Ollama is working' and nothing else.",
            system="You are a helpful assistant.",
            provider="ollama",
            temp=0.0,
        )
        print(f"Response: {reply.strip()}")
        print("✓ llm_client.py is working correctly.")
    except Exception as e:
        print(f"✗ Error: {e}")
        print("  Make sure Ollama is running: ollama serve")
