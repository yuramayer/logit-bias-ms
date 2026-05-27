from __future__ import annotations

"""Генерация текста и позиционное применение `logit_bias`.

Это один из ключевых модулей эксперимента.

Что мы здесь делаем:
- собираем запросы к OpenAI API;
- разбиваем генерацию на сегменты;
- включаем `logit_bias` только в нужном сегменте;
- склеиваем итоговый текст обратно в один output.

Почему так:
- стандартный API не гарантирует удобное step-wise переключение bias на каждом
  токене;
- поэтому для MVP используется `segment approximation`;
- позже этот модуль будет главным кандидатом на замену, если вы перейдете к
  более точному step-wise декодированию или к локальной inference-схеме.
"""

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .config import ExperimentConfig


@dataclass
class SegmentSpec:
    """Описание одного сегмента генерации.

    Например: "первые 25% с bias", "середина без bias", "конец с bias".
    """

    max_tokens: int
    bias_enabled: bool
    name: str


@dataclass
class GenerationResult:
    """Что вернул один полный запуск генерации."""

    text: str
    completion_tokens: int
    prompt_tokens: int | None
    finish_reason: str | None
    logprobs: list[dict] | None
    notes: list[str]


class OpenAIChatClient:
    """Минимальный REST-клиент для chat completions и embeddings.

    Здесь нет SDK специально: меньше скрытой магии, проще дебажить запросы,
    проще потом адаптировать под другой endpoint.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        _load_env_file()
        api_key = (
            os.getenv(config.openai.api_key_env)
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("OPENAI_TOKEN")
        )
        if not api_key:
            raise RuntimeError(
                f"Environment variable {config.openai.api_key_env}, OPENAI_API_KEY or OPENAI_TOKEN is required for generation."
            )
        self.api_key = api_key

    def generate(
        self,
        prompt: str,
        condition: str,
        repetition_id: int,
        logit_bias: dict[str, int] | None,
    ) -> GenerationResult:
        """Делает одну полную генерацию для выбранного условия.

        Важно понимать механику:
        - `control` обычно состоит из одного сегмента без bias;
        - `early/mid/late` состоят из нескольких сегментов;
        - после каждого сегмента мы просим модель продолжить уже начатый ответ.

        Это и есть техническая аппроксимация позиционного вмешательства.
        """
        segments = build_segments(condition, self.config)
        accumulated = ""
        total_completion_tokens = 0
        prompt_tokens = None
        finish_reason = None
        all_logprobs: list[dict] = []
        notes = [f"generation_approach={self.config.generation_approach}"]
        for segment in segments:
            # На каждом шаге мы пересобираем диалог так, чтобы модель видела
            # уже написанный кусок и продолжала его, а не начинала заново.
            messages = build_messages(
                system_prompt=self.config.system_prompt,
                prompt=prompt,
                accumulated_text=accumulated,
            )
            payload = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                # Для совместимости с не-reasoning chat models используем
                # `max_tokens`. Для текущего MVP это практичнее, чем
                # завязываться на model-specific aliases.
                "max_tokens": segment.max_tokens,
            }
            if self.config.seed is not None:
                payload["seed"] = self.config.seed + repetition_id
            if self.config.enable_logprobs:
                payload["logprobs"] = True
            if segment.bias_enabled and logit_bias:
                # Bias включаем только там, где это предписывает условие.
                payload["logit_bias"] = logit_bias
                notes.append(f"segment={segment.name}:bias=on")
            else:
                notes.append(f"segment={segment.name}:bias=off")
            response = self._post_json(self.config.openai.chat_path, payload)
            choice = response["choices"][0]
            piece = choice["message"]["content"]
            accumulated += piece
            usage = response.get("usage", {})
            total_completion_tokens += int(usage.get("completion_tokens", 0))
            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
            finish_reason = choice.get("finish_reason", finish_reason)
            if self.config.enable_logprobs:
                content = (choice.get("logprobs") or {}).get("content") or []
                all_logprobs.extend(content)
        return GenerationResult(
            text=accumulated,
            completion_tokens=total_completion_tokens,
            prompt_tokens=prompt_tokens,
            finish_reason=finish_reason,
            logprobs=all_logprobs or None,
            notes=notes,
        )

    def embedding(self, text: str) -> list[float]:
        """Получает embedding текста для семантического сравнения."""
        if not self.config.embedding_model:
            raise RuntimeError("embedding_model is not configured.")
        payload = {"model": self.config.embedding_model, "input": text}
        response = self._post_json(self.config.openai.embeddings_path, payload)
        return response["data"][0]["embedding"]

    def _post_json(self, path: str, payload: dict) -> dict:
        """Отправляет JSON-запрос в API и возвращает JSON-ответ."""
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=self.config.openai.base_url.rstrip("/") + path,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.openai.timeout_seconds
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API request failed: {details}") from exc


def build_messages(system_prompt: str | None, prompt: str, accumulated_text: str) -> list[dict]:
    """Собирает сообщения для очередного куска генерации.

    Если текст уже частично сгенерирован, мы подсовываем его как предыдущий
    ответ ассистента и просим продолжить. Иначе модель бы слишком часто
    начинала писать ответ заново.
    """
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    if accumulated_text:
        messages.append({"role": "assistant", "content": accumulated_text})
        messages.append(
            {
                "role": "user",
                "content": "Продолжи тот же ответ с места остановки. Не начинай заново и не повторяй уже написанное.",
            }
        )
    return messages


def build_segments(condition: str, config: ExperimentConfig) -> list[SegmentSpec]:
    """Преобразует условие эксперимента в список сегментов генерации.

    Это место особенно важно для масштабирования:
    если вы захотите другие интервалы, больше сегментов или step-wise режим,
    менять логику нужно в первую очередь здесь.
    """
    max_tokens = config.max_tokens
    bounds = config.segment_boundaries
    if condition == "control":
        return [SegmentSpec(max_tokens=max_tokens, bias_enabled=False, name="full")]

    if condition == "early":
        early_tokens = max(1, round(max_tokens * bounds.early_end))
        return _compress_segments(
            [
                SegmentSpec(max_tokens=early_tokens, bias_enabled=True, name="early"),
                SegmentSpec(max_tokens=max_tokens - early_tokens, bias_enabled=False, name="rest"),
            ]
        )

    if condition == "mid":
        head = max(1, round(max_tokens * bounds.mid_start))
        middle = max(1, round(max_tokens * (bounds.mid_end - bounds.mid_start)))
        tail = max_tokens - head - middle
        return _compress_segments(
            [
                SegmentSpec(max_tokens=head, bias_enabled=False, name="pre_mid"),
                SegmentSpec(max_tokens=middle, bias_enabled=True, name="mid"),
                SegmentSpec(max_tokens=tail, bias_enabled=False, name="post_mid"),
            ]
        )

    if condition == "late":
        prefix = max(1, round(max_tokens * bounds.late_start))
        return _compress_segments(
            [
                SegmentSpec(max_tokens=prefix, bias_enabled=False, name="pre_late"),
                SegmentSpec(max_tokens=max_tokens - prefix, bias_enabled=True, name="late"),
            ]
        )

    raise ValueError(f"Unsupported condition: {condition}")


def build_logit_bias_map(model: str, markers: list[str], bias_value: int) -> dict[str, int]:
    """Преобразует текстовые маркеры в карту token_id -> bias.

    `logit_bias` в OpenAI работает не по словам, а по id токенов. Поэтому
    сначала нужно прогнать маркеры через токенизатор модели.

    Важная практическая оговорка:
    если маркер разбивается на несколько токенов, bias будет применен к каждому
    из них. Это не идеально, но для MVP достаточно.
    """
    try:
        import tiktoken  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Package 'tiktoken' is required to build OpenAI token-id logit bias maps."
        ) from exc

    try:
        encoder = tiktoken.encoding_for_model(model)
    except KeyError:
        encoder = tiktoken.get_encoding("cl100k_base")

    bias_map: dict[str, int] = {}
    for marker in markers:
        for token_id in encoder.encode(marker):
            bias_map[str(token_id)] = bias_value
    return bias_map


def cosine_similarity_embeddings(left: list[float], right: list[float]) -> float | None:
    """Обычный cosine для embedding-векторов."""
    if not left or not right or len(left) != len(right):
        return None
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return None
    return dot / (left_norm * right_norm)


def _compress_segments(segments: list[SegmentSpec]) -> list[SegmentSpec]:
    """Убирает пустые сегменты, если после округления там 0 токенов."""
    return [segment for segment in segments if segment.max_tokens > 0]


def _load_env_file() -> None:
    """Подгружает `.env`, если он есть в корне проекта.

    Нам не нужен полноценный dotenv-парсер: для MVP достаточно уметь читать
    строки вида `KEY=value` или `KEY='value'`.
    """
    env_path = Path(".env")
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
