from __future__ import annotations

"""Probe OpenAI-compatible providers for features needed by this experiment.

The real experiment needs two things:
- provider accepts `logit_bias`;
- provider can optionally return `logprobs`.

This script sends tiny requests before spending money on a 1200-run batch.
It intentionally uses only stdlib so it can run before optional tokenizer deps
are installed.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProviderDefaults:
    api_key_env: str
    base_url: str
    chat_path: str
    model_env: str
    default_model: str | None


PROVIDERS: dict[str, ProviderDefaults] = {
    "deepseek": ProviderDefaults(
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        chat_path="/chat/completions",
        model_env="DEEPSEEK_MODEL",
        default_model="deepseek-v4-flash",
    ),
    "fireworks": ProviderDefaults(
        api_key_env="FIREWORKS_API_KEY",
        base_url="https://api.fireworks.ai/inference",
        chat_path="/v1/chat/completions",
        model_env="FIREWORKS_MODEL",
        default_model=None,
    ),
    "together": ProviderDefaults(
        api_key_env="TOGETHER_API_KEY",
        base_url="https://api.together.xyz",
        chat_path="/v1/chat/completions",
        model_env="TOGETHER_MODEL",
        default_model=None,
    ),
    "openrouter": ProviderDefaults(
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api",
        chat_path="/v1/chat/completions",
        model_env="OPENROUTER_MODEL",
        default_model=None,
    ),
    "groq": ProviderDefaults(
        api_key_env="GROQ_API_KEY",
        base_url="https://api.groq.com/openai",
        chat_path="/v1/chat/completions",
        model_env="GROQ_MODEL",
        default_model=None,
    ),
    "mistral": ProviderDefaults(
        api_key_env="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai",
        chat_path="/v1/chat/completions",
        model_env="MISTRAL_MODEL",
        default_model=None,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe provider support for logit_bias/logprobs.")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--model", help="Override provider model.")
    parser.add_argument("--api-key-env", help="Override API key env var.")
    parser.add_argument("--base-url", help="Override base URL.")
    parser.add_argument("--chat-path", help="Override chat completions path.")
    args = parser.parse_args()

    defaults = PROVIDERS[args.provider]
    api_key_env = args.api_key_env or defaults.api_key_env
    api_key = os.getenv(api_key_env)
    if not api_key:
        print(json.dumps({"ok": False, "error": f"missing env var {api_key_env}"}, ensure_ascii=False))
        raise SystemExit(2)

    model = args.model or os.getenv(defaults.model_env) or defaults.default_model
    if not model:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"model is required; pass --model or set {defaults.model_env}",
                },
                ensure_ascii=False,
            )
        )
        raise SystemExit(2)

    base_url = (args.base_url or defaults.base_url).rstrip("/")
    chat_path = args.chat_path or defaults.chat_path
    url = base_url + chat_path

    results = {
        "provider": args.provider,
        "model": model,
        "url": url,
        "base": _probe(url, api_key, _payload(model)),
        "logit_bias": _probe(url, api_key, _payload(model, logit_bias={"0": -1})),
        "logprobs": _probe(url, api_key, _payload(model, **_logprobs_args(args.provider))),
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if not (results["base"]["ok"] and results["logit_bias"]["ok"]):
        raise SystemExit(1)


def _payload(model: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Ответь одним словом: тест"}],
        "temperature": 0,
        "max_tokens": 8,
    }
    payload.update(extra)
    return payload


def _logprobs_args(provider: str) -> dict[str, Any]:
    if provider == "together":
        return {"logprobs": 1}
    return {"logprobs": True, "top_logprobs": 1}


def _probe(url: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "status": exc.code,
            "error": details[:1000],
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    choice = (data.get("choices") or [{}])[0]
    return {
        "ok": True,
        "finish_reason": choice.get("finish_reason"),
        "has_logprobs": choice.get("logprobs") is not None,
    }


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
