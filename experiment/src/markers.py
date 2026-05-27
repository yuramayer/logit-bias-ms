from __future__ import annotations

"""Работа со словарем дискурсивных маркеров.

Здесь мы фиксируем заранее заданный набор маркеров и считаем их частоты в
тексте. Это и есть operational proxy для дискурсивного эффекта в MVP.

Главная мысль:
- маркеры загружаются из отдельного файла;
- мы не подбираем их по уже полученным output;
- код лишь честно считает то, что было заранее определено.
"""

import re
from dataclasses import dataclass
from typing import Iterable

from .io_utils import load_json


WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+(?:[-'][A-Za-zА-Яа-яЁё0-9]+)?")


@dataclass(frozen=True)
class MarkerCategory:
    """Одна категория маркеров, например hedging или logical_structuring."""

    name: str
    markers: tuple[str, ...]


@dataclass(frozen=True)
class MarkerDictionary:
    """Весь словарь маркеров сразу.

    Нужен как единая точка доступа:
    - получить список категорий;
    - достать маркеры только нужных категорий;
    - не размазывать словарь по разным файлам и функциям.
    """

    categories: tuple[MarkerCategory, ...]

    @classmethod
    def load(cls, path: str) -> "MarkerDictionary":
        """Читает словарь маркеров из JSON."""
        payload = load_json(path)
        categories = []
        for name, markers in payload["categories"].items():
            categories.append(MarkerCategory(name=name, markers=tuple(markers)))
        return cls(categories=tuple(categories))

    def category_names(self) -> list[str]:
        """Возвращает имена категорий в том порядке, как они были в файле."""
        return [category.name for category in self.categories]

    def markers_for_categories(self, names: Iterable[str]) -> list[str]:
        """Возвращает все маркеры только из выбранных категорий.

        Это нужно для `logit_bias`: можно, например, давить только
        `anglocentric_formulas`, не трогая остальные категории.
        """
        requested = set(names)
        markers: list[str] = []
        for category in self.categories:
            if category.name in requested:
                markers.extend(category.markers)
        return markers


def tokenize_words(text: str) -> list[str]:
    """Грубая токенизация на уровне слов.

    Это не токенизация модели. Она нужна только для наших простых локальных
    метрик: нормализации частот, cosine по словам и proxy-perplexity.
    """
    return [match.group(0).lower() for match in WORD_RE.finditer(text.lower())]


def marker_statistics(text: str, marker_dict: MarkerDictionary) -> dict:
    """Считает, сколько раз маркеры встретились в тексте.

    На выходе получаем:
    - общее число слов;
    - общее число маркерных попаданий;
    - нормализованный marker score;
    - разбивку по категориям;
    - разбивку по конкретным маркерам.

    Это одна из ключевых функций MVP: из нее потом строится proxy для `delta_p0`.
    """
    lowered = text.lower()
    total_tokens = max(len(tokenize_words(text)), 1)
    by_category: dict[str, dict] = {}
    total_hits = 0
    marker_matches: dict[str, int] = {}

    for category in marker_dict.categories:
        category_hits = 0
        for marker in category.markers:
            pattern = _compile_marker_pattern(marker)
            hits = len(pattern.findall(lowered))
            if hits:
                marker_matches[marker] = hits
            category_hits += hits
        by_category[category.name] = {
            "count": category_hits,
            "normalized": category_hits / total_tokens,
        }
        total_hits += category_hits

    return {
        "token_count": total_tokens,
        "total_marker_hits": total_hits,
        "total_marker_score": total_hits / total_tokens,
        "marker_counts_by_category": by_category,
        "marker_counts": marker_matches,
    }


def _compile_marker_pattern(marker: str) -> re.Pattern[str]:
    """Собирает regex для точного поиска маркера как отдельной единицы."""
    escaped = re.escape(marker.lower())
    return re.compile(rf"(?<!\w){escaped}(?!\w)")
