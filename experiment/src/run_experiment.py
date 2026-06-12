from __future__ import annotations

"""Основной раннер эксперимента.

Если объяснять совсем просто, этот файл делает весь конвейер:
1. читает конфиг;
2. читает промпты и словарь маркеров;
3. запускает все комбинации prompt x condition x repetition;
4. сохраняет raw results;
5. после прогонов считает метрики;
6. собирает итоговые таблицы.

При расширении до большого эксперимента именно этот файл обычно начинают
делить на более крупные части: scheduler, batch runner, retry logic, queue.
"""

import argparse
import traceback
from collections import defaultdict
from pathlib import Path

from .config import ExperimentConfig
from .generation import (
    OpenAIChatClient,
    build_logit_bias_map,
    cosine_similarity_embeddings,
)
from .io_utils import ensure_directories, load_json, utc_timestamp, write_csv, write_json, write_jsonl
from .markers import MarkerDictionary, marker_statistics
from .metrics import (
    BigramPerplexityProxy,
    condition_summary,
    cosine_similarity_counts,
    hypothesis_rows,
    marker_category_rows,
)


def main() -> None:
    """Точка входа в эксперимент."""
    parser = argparse.ArgumentParser(description="Run positional logit-bias experiment.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    ensure_directories([config.outputs_dir, config.raw_dir, config.tables_dir, config.logs_dir])

    # Загружаем все заранее зафиксированные артефакты эксперимента.
    prompts = load_json(config.prompts_path)["prompts"]
    marker_dict = MarkerDictionary.load(str(config.markers_path))
    client = OpenAIChatClient(config)
    logit_bias = build_logit_bias_map(
        model=config.model,
        markers=marker_dict.markers_for_categories(config.bias_categories),
        bias_value=config.bias_value,
        tokenizer_backend=config.tokenizer_backend,
        tokenizer_model=config.tokenizer_model,
    )

    raw_rows: list[dict] = []
    for prompt in prompts:
        prompt_id = prompt["id"]
        for condition in ("control", "early", "mid", "late"):
            for repetition_id in range(1, config.repetitions + 1):
                # Один row = один воспроизводимый запуск, который потом можно
                # отдельно открыть, проверить и при необходимости перезапустить.
                timestamp = utc_timestamp()
                run_id = f"{prompt_id}__{condition}__rep{repetition_id}"
                row = {
                    "run_id": run_id,
                    "prompt_id": prompt_id,
                    "condition": condition,
                    "repetition_id": repetition_id,
                    "timestamp": timestamp,
                    "model_name": config.model,
                    "prompt_text": prompt["text"],
                    "full_output_text": "",
                    "token_count": None,
                    "marker_counts_by_category": {},
                    "marker_counts": {},
                    "total_marker_score": None,
                    "delta_p0": None,
                    "cosine_similarity": None,
                    "perplexity": None,
                    "error_flag": False,
                    "notes": "",
                    "finish_reason": None,
                }
                try:
                    result = client.generate(
                        prompt=prompt["text"],
                        condition=condition,
                        repetition_id=repetition_id,
                        logit_bias=logit_bias if condition != "control" else None,
                    )
                    # Сразу после генерации считаем маркеры, чтобы raw-файл уже
                    # содержал базовую исследовательскую информацию.
                    stats = marker_statistics(result.text, marker_dict)
                    row["full_output_text"] = result.text
                    row["token_count"] = stats["token_count"]
                    row["marker_counts_by_category"] = stats["marker_counts_by_category"]
                    row["marker_counts"] = stats["marker_counts"]
                    row["total_marker_score"] = stats["total_marker_score"]
                    row["finish_reason"] = result.finish_reason
                    row["notes"] = " | ".join(result.notes)
                    if config.enable_logprobs and result.logprobs is not None:
                        row["logprobs"] = result.logprobs
                except Exception as exc:  # noqa: BLE001
                    # Мы не валим весь эксперимент из-за одного неудачного
                    # запуска. Ошибку сохраняем прямо в row и идем дальше.
                    row["error_flag"] = True
                    row["notes"] = f"{type(exc).__name__}: {exc}"
                    row["traceback"] = traceback.format_exc()
                raw_rows.append(row)
                write_json(config.raw_dir / f"{run_id}.json", row)

    # После того как все сырые тексты готовы, можно считать метрики, которым
    # нужен контекст control-ответов по всему набору.
    apply_posthoc_metrics(config, raw_rows, client, prompts, marker_dict.category_names())
    export_tables(config, raw_rows, marker_dict.category_names())


def apply_posthoc_metrics(
    config: ExperimentConfig,
    raw_rows: list[dict],
    client: OpenAIChatClient,
    prompts: list[dict],
    category_names: list[str],
) -> None:
    """Досчитывает метрики, которым нужен доступ ко всем raw-результатам.

    Почему это отдельный этап:
    - `delta_p0` считается относительно control;
    - proxy perplexity строится по control-текстам;
    - similarity тоже сравнивается с контрольным ответом.

    То есть эти штуки удобнее считать не в момент генерации, а после всего прогона.
    """
    control_rows_by_prompt: dict[str, list[dict]] = defaultdict(list)
    control_rows_by_prompt_rep: dict[tuple[str, int], dict] = {}
    for row in raw_rows:
        if row["condition"] == "control" and not row["error_flag"]:
            control_rows_by_prompt[row["prompt_id"]].append(row)
            control_rows_by_prompt_rep[(row["prompt_id"], row["repetition_id"])] = row

    # На control-ответах строим baseline для части метрик.
    control_texts = [row["full_output_text"] for row in raw_rows if row["condition"] == "control" and not row["error_flag"]]
    ppl_proxy = BigramPerplexityProxy(control_texts)

    embedding_cache: dict[str, list[float]] = {}
    if config.similarity_method == "openai_embeddings":
        # Embeddings кэшируем заранее, чтобы не дублировать API-вызовы.
        for row in raw_rows:
            if row["error_flag"] or not row["full_output_text"]:
                continue
            embedding_cache[row["run_id"]] = client.embedding(row["full_output_text"])

    for row in raw_rows:
        if row["error_flag"]:
            continue
        prompt_controls = control_rows_by_prompt.get(row["prompt_id"], [])
        if prompt_controls:
            # `delta_p0` здесь реализован как сдвиг marker score относительно
            # среднего control по тому же prompt.
            baseline = sum(item["total_marker_score"] for item in prompt_controls) / len(prompt_controls)
            row["delta_p0"] = round(row["total_marker_score"] - baseline, 6)
        reference = control_rows_by_prompt_rep.get((row["prompt_id"], row["repetition_id"]))
        if reference is None and prompt_controls:
            # Если точного control с тем же repetition_id нет, берем первый
            # доступный control для данного prompt как запасной референс.
            reference = prompt_controls[0]
        if reference is not None:
            if config.similarity_method == "openai_embeddings":
                row["cosine_similarity"] = _rounded(
                    cosine_similarity_embeddings(
                        embedding_cache[row["run_id"]],
                        embedding_cache[reference["run_id"]],
                    )
                )
            else:
                row["cosine_similarity"] = _rounded(
                    cosine_similarity_counts(row["full_output_text"], reference["full_output_text"])
                )
        if config.perplexity_method == "control_bigram_proxy":
            row["perplexity"] = _rounded(ppl_proxy.score(row["full_output_text"]))
        else:
            row["perplexity"] = None
        row["notes"] = _append_method_notes(
            existing=row["notes"],
            delta_p0_mode=config.delta_p0_mode,
            similarity_method=config.similarity_method,
            perplexity_method=config.perplexity_method,
        )

    # Этот manifest нужен как короткая техническая памятка: что именно было
    # запущено и какими методами считались основные показатели.
    write_json(
        config.logs_dir / "run_manifest.json",
        {
            "prompt_ids": [prompt["id"] for prompt in prompts],
            "category_names": category_names,
            "similarity_method": config.similarity_method,
            "perplexity_method": config.perplexity_method,
            "delta_p0_mode": config.delta_p0_mode,
            "generation_approach": config.generation_approach,
        },
    )


def export_tables(config: ExperimentConfig, raw_rows: list[dict], categories: list[str]) -> None:
    """Пишет на диск все таблицы, которые потом нужны для анализа и ВКР."""
    sanitized_rows = [_flatten_raw_row(row) for row in raw_rows]
    write_jsonl(config.tables_dir / "raw_runs.jsonl", raw_rows)
    write_csv(config.tables_dir / "raw_runs.csv", sanitized_rows)
    write_csv(
        config.tables_dir / "aggregated_by_condition.csv",
        condition_summary(
            sanitized_rows,
            numeric_fields=["token_count", "total_marker_score", "delta_p0", "cosine_similarity", "perplexity"],
        ),
    )
    write_csv(config.tables_dir / "hypothesis_check.csv", hypothesis_rows(sanitized_rows))
    write_csv(config.tables_dir / "marker_category_comparison.csv", marker_category_rows(raw_rows, categories))


def _flatten_raw_row(row: dict) -> dict:
    """Делает row безопасным для CSV.

    CSV не умеет нормально хранить вложенные словари, поэтому сложные поля
    сначала сериализуем в строку.
    """
    flattened = {key: value for key, value in row.items() if key not in {"marker_counts_by_category", "marker_counts", "logprobs", "traceback"}}
    flattened["marker_counts_by_category"] = _stable_text(row.get("marker_counts_by_category", {}))
    flattened["marker_counts"] = _stable_text(row.get("marker_counts", {}))
    if "logprobs" in row:
        flattened["logprobs"] = _stable_text(row["logprobs"])
    if "traceback" in row:
        flattened["traceback"] = row["traceback"]
    return flattened


def _stable_text(value: object) -> str:
    """Превращает объект в стабильную JSON-строку."""
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _rounded(value: float | None) -> float | None:
    """Округление с защитой от `None`."""
    return round(value, 6) if value is not None else None


def _append_method_notes(
    existing: str,
    delta_p0_mode: str,
    similarity_method: str,
    perplexity_method: str,
) -> str:
    """Добавляет в notes явную запись о том, как считались метрики."""
    additions = [
        f"delta_p0_mode={delta_p0_mode}",
        f"similarity_method={similarity_method}",
        f"perplexity_method={perplexity_method}",
    ]
    if existing:
        return existing + " | " + " | ".join(additions)
    return " | ".join(additions)


if __name__ == "__main__":
    main()
