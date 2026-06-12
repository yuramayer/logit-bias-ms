from __future__ import annotations

"""Загрузка и нормализация конфига эксперимента.

Идея модуля простая:
- весь эксперимент должен управляться не правками кода, а конфигом;
- пути, параметры модели и методики должны собираться в одном месте;
- когда proof-of-concept вырастет в большой прогон, менять нужно в первую
  очередь конфиг, а не логику раннера.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SegmentBoundaries:
    """Границы сегментов генерации в долях от общей длины.

    Пример:
    - `early_end = 0.25` значит, что ранний сегмент занимает первые 25%;
    - `mid_start = 0.40`, `mid_end = 0.60` значит, что mid идет в середине;
    - `late_start = 0.75` значит, что late включается на последних 25%.

    Это нужно, чтобы не зашивать логику `early/mid/late` в код руками.
    """

    early_end: float
    mid_start: float
    mid_end: float
    late_start: float

    def validate(self) -> None:
        """Проверяем, что границы вообще имеют смысл.

        Если тут ошибка, значит эксперимент нельзя считать воспроизводимым:
        сегменты будут пересекаться или выходить за пределы [0, 1].
        """
        values = [self.early_end, self.mid_start, self.mid_end, self.late_start]
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError("Segment boundaries must be inside [0, 1].")
        if not (self.early_end <= self.mid_start <= self.mid_end <= self.late_start):
            raise ValueError("Expected early_end <= mid_start <= mid_end <= late_start.")


@dataclass(frozen=True)
class OpenAISettings:
    """Технические параметры доступа к OpenAI API.

    Вынесены отдельно, чтобы потом было проще:
    - сменить базовый URL;
    - подключить прокси или совместимый endpoint;
    - развести chat и embeddings по разным путям.
    """

    api_key_env: str
    base_url: str
    chat_path: str
    embeddings_path: str
    timeout_seconds: int


@dataclass(frozen=True)
class ExperimentConfig:
    """Нормализованный конфиг всего эксперимента.

    После `load()` остальной код работает уже не с сырым JSON/YAML, а с
    удобным объектом. Это снижает хаос в коде и делает масштабирование проще.
    """

    model: str
    embedding_model: str | None
    provider: str
    tokenizer_backend: str
    tokenizer_model: str | None
    generation_approach: str
    temperature: float
    top_p: float
    max_tokens: int
    repetitions: int
    seed: int | None
    bias_value: int
    bias_categories: list[str]
    delta_p0_mode: str
    similarity_method: str
    perplexity_method: str
    prompts_path: Path
    markers_path: Path
    outputs_dir: Path
    raw_dir: Path
    tables_dir: Path
    logs_dir: Path
    enable_logprobs: bool
    system_prompt: str | None
    openai: OpenAISettings
    segment_boundaries: SegmentBoundaries

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        """Читает конфиг с диска и приводит его к рабочему виду.

        Что тут важно:
        - можно хранить конфиг как JSON или YAML;
        - относительные пути делаются относительными к самому конфигу;
        - сразу вычисляются папки `raw`, `tables`, `logs`.
        """
        config_path = Path(path)
        payload = _load_config_payload(config_path)
        segment_boundaries = SegmentBoundaries(**payload["segment_boundaries"])
        segment_boundaries.validate()
        base_dir = config_path.parent
        outputs_dir = _resolve_path(base_dir, payload["outputs_dir"])
        return cls(
            model=payload["model"],
            embedding_model=payload.get("embedding_model"),
            provider=payload["provider"],
            tokenizer_backend=str(payload.get("tokenizer_backend", "tiktoken")),
            tokenizer_model=payload.get("tokenizer_model"),
            generation_approach=payload["generation_approach"],
            temperature=float(payload["temperature"]),
            top_p=float(payload["top_p"]),
            max_tokens=int(payload["max_tokens"]),
            repetitions=int(payload["repetitions"]),
            seed=payload.get("seed"),
            bias_value=int(payload["bias_value"]),
            bias_categories=list(payload["bias_categories"]),
            delta_p0_mode=payload["delta_p0_mode"],
            similarity_method=payload["similarity_method"],
            perplexity_method=payload["perplexity_method"],
            prompts_path=_resolve_path(base_dir, payload["prompts_path"]),
            markers_path=_resolve_path(base_dir, payload["markers_path"]),
            outputs_dir=outputs_dir,
            raw_dir=outputs_dir / "raw",
            tables_dir=outputs_dir / "tables",
            logs_dir=outputs_dir / "logs",
            enable_logprobs=bool(payload.get("enable_logprobs", False)),
            system_prompt=payload.get("system_prompt"),
            openai=OpenAISettings(**payload["openai"]),
            segment_boundaries=segment_boundaries,
        )


def _load_config_payload(path: Path) -> dict[str, Any]:
    """Пытается прочитать конфиг сначала как JSON, потом как YAML.

    Это сделано не ради красоты, а чтобы не блокироваться на формате файла.
    Сейчас `config.yaml` фактически может быть и JSON-совместимым YAML.
    """
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            raise ValueError(
                "config.yaml must be valid JSON or you must install PyYAML for full YAML support."
            ) from exc
        payload = yaml.safe_load(text)
        if not isinstance(payload, dict):
            raise ValueError("Top-level config structure must be a mapping.")
        return payload


def _resolve_path(base_dir: Path, value: str) -> Path:
    """Превращает путь из конфига в абсолютный путь.

    Это спасает от типичной путаницы, когда эксперимент запускают из разных
    рабочих директорий и относительные пути внезапно ломаются.
    """
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()
