from __future__ import annotations

"""Метрики и агрегаты для анализа результатов.

Здесь специально разведены три слоя:
- похожесть текста по смыслу;
- proxy для деградации текста;
- агрегаты для таблиц и проверки гипотез.

Это важно, потому что в эксперименте нельзя смешивать дискурсивный эффект,
сохранение смысла и общую "поломанность" текста.
"""

import math
from collections import Counter, defaultdict
from typing import Iterable

from .markers import tokenize_words


def cosine_similarity_counts(left: str, right: str) -> float | None:
    """Считает cosine similarity по мешку слов.

    Это простая и дешевая baseline-метрика. Она не идеальна, но для MVP полезна:
    можно быстро увидеть, не развалился ли смысл совсем грубо.
    """
    left_counts = Counter(tokenize_words(left))
    right_counts = Counter(tokenize_words(right))
    if not left_counts or not right_counts:
        return None
    overlap = set(left_counts) | set(right_counts)
    dot = sum(left_counts[token] * right_counts[token] for token in overlap)
    left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
    right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
    if left_norm == 0 or right_norm == 0:
        return None
    return dot / (left_norm * right_norm)


class BigramPerplexityProxy:
    """Очень упрощенный proxy perplexity на базе биграмм.

    Важная честная оговорка:
    это не настоящая perplexity внешней языковой модели. Это локальный proxy,
    который оценивает, насколько текст похож на распределение контрольных
    текстов по соседству слов.

    Зачем это нужно:
    - быстро получить воспроизводимую метрику деградации;
    - не тянуть тяжелую отдельную scoring-модель на первом этапе;
    - иметь место, куда потом можно подставить более серьезный оценщик.
    """

    def __init__(self, training_texts: Iterable[str]) -> None:
        """Строит статистику по control-текстам."""
        self.unigram_counts: Counter[str] = Counter()
        self.bigram_counts: Counter[tuple[str, str]] = Counter()
        self.vocab: set[str] = {"<s>", "</s>"}
        for text in training_texts:
            tokens = ["<s>", *tokenize_words(text), "</s>"]
            if len(tokens) < 2:
                continue
            self.vocab.update(tokens)
            self.unigram_counts.update(tokens[:-1])
            self.bigram_counts.update(zip(tokens[:-1], tokens[1:]))
        self.vocab_size = max(len(self.vocab), 1)

    def score(self, text: str) -> float | None:
        """Оценивает новый текст через сглаженную биграммную модель."""
        tokens = ["<s>", *tokenize_words(text), "</s>"]
        if len(tokens) < 2:
            return None
        nll = 0.0
        steps = 0
        for prev_token, current_token in zip(tokens[:-1], tokens[1:]):
            bigram = self.bigram_counts[(prev_token, current_token)] + 1
            context = self.unigram_counts[prev_token] + self.vocab_size
            probability = bigram / context
            nll += -math.log(probability)
            steps += 1
        return math.exp(nll / max(steps, 1))


def condition_summary(rows: list[dict], numeric_fields: list[str]) -> list[dict]:
    """Собирает агрегированную таблицу по условиям.

    Это будущая таблица вида:
    `control / early / mid / late` + средние значения метрик.
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)

    summary_rows = []
    for condition, items in sorted(grouped.items()):
        summary = {
            "condition": condition,
            "run_count": len(items),
            "error_count": sum(1 for item in items if item["error_flag"]),
        }
        for field in numeric_fields:
            values = [item[field] for item in items if isinstance(item[field], (int, float))]
            summary[f"{field}_mean"] = round(sum(values) / len(values), 6) if values else None
            summary[f"{field}_min"] = round(min(values), 6) if values else None
            summary[f"{field}_max"] = round(max(values), 6) if values else None
        summary_rows.append(summary)
    return summary_rows


def hypothesis_rows(rows: list[dict]) -> list[dict]:
    """Готовит компактную таблицу именно под проверку H1/H2.

    Идея простая:
    - берем абсолютный сдвиг `delta_p0`;
    - смотрим, кто сильнее: `early`, `mid` или `late`;
    - сохраняем это отдельно, чтобы потом не вычислять руками в Excel.
    """
    by_prompt: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        delta = row.get("delta_p0")
        if isinstance(delta, (int, float)):
            by_prompt[row["prompt_id"]][row["condition"]].append(abs(delta))

    output = []
    for prompt_id, prompt_data in sorted(by_prompt.items()):
        means = {
            condition: (sum(values) / len(values) if values else None)
            for condition, values in prompt_data.items()
        }
        ranking = sorted(
            ((condition, value) for condition, value in means.items() if value is not None and condition != "control"),
            key=lambda item: item[1],
            reverse=True,
        )
        output.append(
            {
                "prompt_id": prompt_id,
                "early_abs_delta_mean": _rounded(means.get("early")),
                "mid_abs_delta_mean": _rounded(means.get("mid")),
                "late_abs_delta_mean": _rounded(means.get("late")),
                "delta_strength_ranking": " > ".join(condition for condition, _ in ranking),
                "early_minus_late_abs_delta": _diff(means.get("early"), means.get("late")),
                "early_minus_mid_abs_delta": _diff(means.get("early"), means.get("mid")),
            }
        )
    return output


def marker_category_rows(rows: list[dict], categories: list[str]) -> list[dict]:
    """Готовит таблицу сравнения категорий маркеров между условиями."""
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("error_flag"):
            continue
        per_category = row["marker_counts_by_category"]
        for category in categories:
            value = per_category[category]["normalized"]
            grouped[(row["condition"], category)].append(value)

    output = []
    for condition, category in sorted(grouped):
        values = grouped[(condition, category)]
        output.append(
            {
                "condition": condition,
                "category": category,
                "normalized_marker_mean": round(sum(values) / len(values), 6),
                "normalized_marker_min": round(min(values), 6),
                "normalized_marker_max": round(max(values), 6),
            }
        )
    return output


def _rounded(value: float | None) -> float | None:
    """Аккуратно округляет число, если оно вообще есть."""
    return round(value, 6) if value is not None else None


def _diff(left: float | None, right: float | None) -> float | None:
    """Считает разницу между двумя числами с защитой от `None`."""
    if left is None or right is None:
        return None
    return round(left - right, 6)
