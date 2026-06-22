"""Unified cloud and local LLM gateway.
Includes provider routing and streaming helpers."""

from __future__ import annotations

import json
from typing import Any, Generator

import requests

from src.utils import get_env, load_env

# When provider="auto", we try providers in this order.
PROVIDER_ORDER = [
    "openai",
    "anthropic",
    "gemini",
    "mistral",
    "cohere",
    "deepseek",
    "groq",
    "openrouter",
    "ollama",
]

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def _has_key(key_name: str) -> bool:
    value = get_env(key_name)
    return bool(value and value.strip())


# --- Ollama (local models) ---

def _ollama_base_url() -> str:
    return get_env("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL) or DEFAULT_OLLAMA_BASE_URL


def _ollama_default_model() -> str:
    return get_env("OLLAMA_TEXT_MODEL", "llama3.2") or "llama3.2"


def check_ollama_server() -> dict[str, Any]:
    """Check whether the local Ollama server is reachable."""
    url = f"{_ollama_base_url().rstrip('/')}/api/tags"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        models = [m.get("name", "") for m in data.get("models", [])]
        return {
            "status": "online",
            "base_url": _ollama_base_url(),
            "model_count": len(models),
            "models": models,
            "message": "Ollama server is running.",
        }
    except requests.RequestException as exc:
        return {
            "status": "offline",
            "base_url": _ollama_base_url(),
            "model_count": 0,
            "models": [],
            "message": "Ollama server is not reachable. Start it with: ollama serve",
            "error": str(exc),
        }


def list_ollama_models() -> list[str]:
    """Return installed Ollama model names."""
    status = check_ollama_server()
    return status.get("models", [])


def _model_is_installed(model_name: str, installed: list[str]) -> bool:
    """Match both `llama3.2` and `llama3.2:latest` model name styles."""
    if not model_name:
        return False
    if model_name in installed:
        return True
    return any(name == model_name or name.startswith(f"{model_name}:") for name in installed)


def resolve_ollama_model(preferred: str | None = None) -> str:
    """
    Pick an installed model.

    Priority:
    1. explicit `preferred`
    2. OLLAMA_TEXT_MODEL
    3. first installed model
    """
    installed = list_ollama_models()
    if not installed:
        raise RuntimeError(
            "Ollama is running but no models are installed. "
            "Run: ollama pull llama3.2 (or any model you prefer)."
        )

    candidates = [preferred, _ollama_default_model()]
    for candidate in candidates:
        if candidate and _model_is_installed(candidate, installed):
            for name in installed:
                if name == candidate or name.startswith(f"{candidate}:"):
                    return name
    return installed[0]


def _ollama_error_message(exc: requests.RequestException, model: str) -> str:
    """Turn low-level HTTP errors into actionable guidance."""
    response = getattr(exc, "response", None)
    if response is not None and response.status_code == 404:
        installed = list_ollama_models()
        installed_text = ", ".join(installed) if installed else "(none)"
        return (
            f"Ollama model '{model}' is not installed. "
            f"Installed models: {installed_text}. "
            f"Run: ollama pull {model} or set OLLAMA_TEXT_MODEL in .env."
        )
    return f"Ollama request failed. Is the server running? Details: {exc}"


def pull_ollama_model(model_name: str) -> dict[str, Any]:
    """Download a model from the Ollama library."""
    url = f"{_ollama_base_url().rstrip('/')}/api/pull"
    try:
        response = requests.post(
            url,
            json={"name": model_name, "stream": False},
            timeout=600,
        )
        response.raise_for_status()
        return {
            "status": "success",
            "model": model_name,
            "message": f"Model '{model_name}' is ready.",
            "details": response.json(),
        }
    except requests.RequestException as exc:
        return {
            "status": "error",
            "model": model_name,
            "message": f"Failed to pull model '{model_name}'.",
            "error": str(exc),
        }


def run_ollama_chat(prompt: str, model: str | None = None) -> str:
    """One-turn chat completion with Ollama."""
    resolved_model = resolve_ollama_model(model)
    url = f"{_ollama_base_url().rstrip('/')}/api/chat"
    payload = {
        "model": resolved_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "").strip()
    except requests.RequestException as exc:
        raise RuntimeError(_ollama_error_message(exc, resolved_model)) from exc


def stream_ollama_chat(prompt: str, model: str | None = None) -> Generator[str, None, None]:
    """Stream one-turn chat completion with Ollama."""
    resolved_model = resolve_ollama_model(model)
    url = f"{_ollama_base_url().rstrip('/')}/api/chat"
    payload = {
        "model": resolved_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line.decode("utf-8"))
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
    except requests.RequestException as exc:
        raise RuntimeError(_ollama_error_message(exc, resolved_model)) from exc


# --- Cloud + routing ---

def check_available_providers() -> dict[str, bool]:
    """Return which providers are currently available."""
    load_env()
    ollama_status = check_ollama_server()
    ollama_online = ollama_status.get("status") == "online"
    ollama_ready = ollama_online and bool(ollama_status.get("models"))
    return {
        "openai": _has_key("OPENAI_API_KEY"),
        "anthropic": _has_key("ANTHROPIC_API_KEY"),
        "gemini": _has_key("GEMINI_API_KEY"),
        "mistral": _has_key("MISTRAL_API_KEY"),
        "cohere": _has_key("COHERE_API_KEY"),
        "deepseek": _has_key("DEEPSEEK_API_KEY"),
        "groq": _has_key("GROQ_API_KEY"),
        "openrouter": _has_key("OPENROUTER_API_KEY"),
        "ollama": ollama_ready,
    }


def _resolve_provider(provider: str) -> str:
    selected = (provider or "auto").lower()
    if selected != "auto":
        available = check_available_providers()
        if selected not in available:
            raise ValueError(f"Unknown provider '{selected}'. Use one of: {', '.join(PROVIDER_ORDER)} or 'auto'.")
        if not available[selected]:
            raise RuntimeError(
                f"Provider '{selected}' is not available. Check API key/server and try again."
            )
        return selected

    available = check_available_providers()
    for name in PROVIDER_ORDER:
        if available.get(name):
            return name
    raise RuntimeError(
        "No LLM provider is available. Add at least one API key in .env or run Ollama locally."
    )


def _messages(prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def _run_openai(
    prompt: str,
    system_prompt: str | None,
    temperature: float,
    max_tokens: int,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=get_env("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=get_env("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=_messages(prompt, system_prompt),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


def _run_anthropic(
    prompt: str,
    system_prompt: str | None,
    temperature: float,
    max_tokens: int,
) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=get_env("ANTHROPIC_API_KEY"))
    kwargs: dict[str, Any] = {
        "model": get_env("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        kwargs["system"] = system_prompt
    response = client.messages.create(**kwargs)
    text = "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    )
    return text.strip()


def _run_gemini(
    prompt: str,
    system_prompt: str | None,
    temperature: float,
    max_tokens: int,
) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=get_env("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system_prompt,
    )
    response = client.models.generate_content(
        model=get_env("GEMINI_MODEL", "gemini-1.5-flash"),
        contents=prompt,
        config=config,
    )
    return (response.text or "").strip()


def _run_mistral(
    prompt: str,
    system_prompt: str | None,
    temperature: float,
    max_tokens: int,
) -> str:
    from mistralai import Mistral

    client = Mistral(api_key=get_env("MISTRAL_API_KEY"))
    response = client.chat.complete(
        model=get_env("MISTRAL_MODEL", "mistral-large-latest"),
        messages=_messages(prompt, system_prompt),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


def _run_cohere(
    prompt: str,
    system_prompt: str | None,
    temperature: float,
    max_tokens: int,
) -> str:
    import cohere

    client = cohere.ClientV2(api_key=get_env("COHERE_API_KEY"))
    response = client.chat(
        model=get_env("COHERE_MODEL", "command-r-plus"),
        messages=_messages(prompt, system_prompt),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (response.message.content[0].text or "").strip()


def _run_deepseek(
    prompt: str,
    system_prompt: str | None,
    temperature: float,
    max_tokens: int,
) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=get_env("DEEPSEEK_API_KEY"),
        base_url=get_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    response = client.chat.completions.create(
        model=get_env("DEEPSEEK_MODEL", "deepseek-chat"),
        messages=_messages(prompt, system_prompt),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


def _run_groq(
    prompt: str,
    system_prompt: str | None,
    temperature: float,
    max_tokens: int,
) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=get_env("GROQ_API_KEY"),
        base_url=get_env("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
    )
    response = client.chat.completions.create(
        model=get_env("GROQ_MODEL", "llama-3.1-70b-versatile"),
        messages=_messages(prompt, system_prompt),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


def _run_openrouter(
    prompt: str,
    system_prompt: str | None,
    temperature: float,
    max_tokens: int,
) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=get_env("OPENROUTER_API_KEY"),
        base_url=get_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    model = get_env("OPENROUTER_MODEL", "").strip()
    if not model:
        model = get_env("OPENAI_MODEL", "gpt-4.1-mini")
    response = client.chat.completions.create(
        model=model,
        messages=_messages(prompt, system_prompt),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


def _run_ollama(
    prompt: str,
    system_prompt: str | None,
    temperature: float,
    max_tokens: int,
) -> str:
    del temperature, max_tokens
    full_prompt = prompt
    if system_prompt:
        full_prompt = f"{system_prompt}\n\nUser request:\n{prompt}"
    return run_ollama_chat(full_prompt)


def run_llm(
    prompt: str,
    provider: str = "auto",
    system_prompt: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 1000,
) -> str:
    """
    Run one prompt through the selected provider and return text.
    """
    load_env()
    selected = _resolve_provider(provider)

    runners = {
        "openai": _run_openai,
        "anthropic": _run_anthropic,
        "gemini": _run_gemini,
        "mistral": _run_mistral,
        "cohere": _run_cohere,
        "deepseek": _run_deepseek,
        "groq": _run_groq,
        "openrouter": _run_openrouter,
        "ollama": _run_ollama,
    }

    runner = runners[selected]
    return runner(prompt, system_prompt, temperature, max_tokens)


# --- Streaming ---

def stream_llm(
    prompt: str,
    provider: str = "auto",
    system_prompt: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 1000,
) -> Generator[str, None, None]:
    """Yield streamed chunks where available, otherwise fall back to run_llm."""
    selected = _resolve_provider(provider.lower())

    try:
        if selected == "openai":
            yield from _stream_openai(prompt, system_prompt, temperature, max_tokens)
        elif selected == "anthropic":
            yield from _stream_anthropic(prompt, system_prompt, temperature, max_tokens)
        elif selected == "gemini":
            yield from _stream_gemini(prompt, system_prompt, temperature, max_tokens)
        elif selected == "mistral":
            yield from _stream_mistral(prompt, system_prompt, temperature, max_tokens)
        elif selected == "cohere":
            yield from _stream_cohere(prompt, system_prompt, temperature, max_tokens)
        elif selected == "ollama":
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser request:\n{prompt}"
            yield from stream_ollama_chat(full_prompt)
        else:
            # DeepSeek, Groq, OpenRouter: fall back to non-streaming.
            yield run_llm(
                prompt,
                provider=selected,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
    except Exception:
        yield run_llm(
            prompt,
            provider=selected,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )


def _stream_openai(
    prompt: str,
    system_prompt: str | None,
    temperature: float,
    max_tokens: int,
) -> Generator[str, None, None]:
    from openai import OpenAI

    client = OpenAI(api_key=get_env("OPENAI_API_KEY"))
    stream = client.chat.completions.create(
        model=get_env("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=_messages(prompt, system_prompt),
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _stream_anthropic(
    prompt: str,
    system_prompt: str | None,
    temperature: float,
    max_tokens: int,
) -> Generator[str, None, None]:
    from anthropic import Anthropic

    client = Anthropic(api_key=get_env("ANTHROPIC_API_KEY"))
    kwargs: dict[str, Any] = {
        "model": get_env("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    with client.messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            yield text


def _stream_gemini(
    prompt: str,
    system_prompt: str | None,
    temperature: float,
    max_tokens: int,
) -> Generator[str, None, None]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=get_env("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system_prompt,
    )
    for chunk in client.models.generate_content_stream(
        model=get_env("GEMINI_MODEL", "gemini-1.5-flash"),
        contents=prompt,
        config=config,
    ):
        if chunk.text:
            yield chunk.text


def _stream_mistral(
    prompt: str,
    system_prompt: str | None,
    temperature: float,
    max_tokens: int,
) -> Generator[str, None, None]:
    from mistralai import Mistral

    client = Mistral(api_key=get_env("MISTRAL_API_KEY"))
    response = client.chat.stream(
        model=get_env("MISTRAL_MODEL", "mistral-large-latest"),
        messages=_messages(prompt, system_prompt),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    for chunk in response:
        if chunk.data.choices[0].delta.content:
            yield chunk.data.choices[0].delta.content


def _stream_cohere(
    prompt: str,
    system_prompt: str | None,
    temperature: float,
    max_tokens: int,
) -> Generator[str, None, None]:
    import cohere

    client = cohere.ClientV2(api_key=get_env("COHERE_API_KEY"))
    response = client.chat_stream(
        model=get_env("COHERE_MODEL", "command-r-plus"),
        messages=_messages(prompt, system_prompt),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    for event in response:
        if event.type == "content-delta":
            yield event.delta.message.content.text
