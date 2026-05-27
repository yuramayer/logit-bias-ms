from __future__ import annotations

"""Мелкие функции для чтения и записи файлов.

Тут нет исследовательской логики. Это просто утилиты, чтобы основной код не
засорялся однотипной файловой рутиной.
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def load_json(path: str | Path):
    """Читает JSON-файл целиком."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ensure_directories(paths: Iterable[Path]) -> None:
    """Создает нужные директории, если их еще нет."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def utc_timestamp() -> str:
    """Возвращает текущее время в UTC для логирования запусков."""
    return datetime.now(timezone.utc).isoformat()


def write_json(path: str | Path, payload: object) -> None:
    """Сохраняет один объект в красивом JSON-формате."""
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    """Сохраняет список словарей в формате JSONL, строка за строкой."""
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: list[dict]) -> None:
    """Сохраняет список словарей как CSV-таблицу."""
    target = Path(path)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
