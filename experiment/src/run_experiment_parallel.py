from __future__ import annotations

"""Parallel runner for the scaled positional logit-bias experiment."""

import argparse
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import ExperimentConfig
from .generation import OpenAIChatClient, build_logit_bias_map
from .io_utils import ensure_directories, load_json, utc_timestamp, write_json
from .markers import MarkerDictionary, marker_statistics
from .run_experiment import apply_posthoc_metrics, export_tables


CONDITIONS = ("control", "early", "mid", "late")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scaled positional logit-bias experiment in parallel.")
    parser.add_argument("--config", required=True, help="Path to experiment config.")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel run workers.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per failed run.")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress after N completed runs.")
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    ensure_directories([config.outputs_dir, config.raw_dir, config.tables_dir, config.logs_dir])

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

    tasks = [
        (prompt, condition, repetition_id)
        for prompt in prompts
        for condition in CONDITIONS
        for repetition_id in range(1, config.repetitions + 1)
    ]

    raw_rows: list[dict] = []
    pending = []
    for prompt, condition, repetition_id in tasks:
        run_id = _run_id(prompt["id"], condition, repetition_id)
        existing = _load_existing_success(config.raw_dir / f"{run_id}.json")
        if existing is not None:
            raw_rows.append(existing)
        else:
            pending.append((prompt, condition, repetition_id))

    total = len(tasks)
    print(
        f"config={args.config} model={config.model} total={total} "
        f"existing_success={len(raw_rows)} pending={len(pending)} workers={args.workers}",
        flush=True,
    )

    completed = len(raw_rows)
    started_at = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                _run_one,
                config,
                client,
                marker_dict,
                logit_bias,
                prompt,
                condition,
                repetition_id,
                args.retries,
            ): (prompt["id"], condition, repetition_id)
            for prompt, condition, repetition_id in pending
        }
        for future in as_completed(futures):
            row = future.result()
            raw_rows.append(row)
            completed += 1
            if completed % args.progress_every == 0 or completed == total:
                elapsed = max(time.time() - started_at, 1.0)
                rate = completed / elapsed
                errors = sum(1 for item in raw_rows if item.get("error_flag"))
                print(
                    f"progress={completed}/{total} errors={errors} rate={rate:.2f}/s last={row['run_id']}",
                    flush=True,
                )

    raw_rows.sort(key=lambda row: (row["prompt_id"], CONDITIONS.index(row["condition"]), row["repetition_id"]))
    apply_posthoc_metrics(config, raw_rows, client, prompts, marker_dict.category_names())
    export_tables(config, raw_rows, marker_dict.category_names())
    print(f"done model={config.model} rows={len(raw_rows)} tables={config.tables_dir}", flush=True)


def _run_one(
    config: ExperimentConfig,
    client: OpenAIChatClient,
    marker_dict: MarkerDictionary,
    logit_bias: dict[str, int],
    prompt: dict,
    condition: str,
    repetition_id: int,
    retries: int,
) -> dict:
    run_id = _run_id(prompt["id"], condition, repetition_id)
    row = _base_row(config, prompt, condition, repetition_id)
    last_error: BaseException | None = None
    for attempt in range(1, retries + 2):
        try:
            result = client.generate(
                prompt=prompt["text"],
                condition=condition,
                repetition_id=repetition_id,
                logit_bias=logit_bias if condition != "control" else None,
            )
            stats = marker_statistics(result.text, marker_dict)
            row["full_output_text"] = result.text
            row["token_count"] = stats["token_count"]
            row["marker_counts_by_category"] = stats["marker_counts_by_category"]
            row["marker_counts"] = stats["marker_counts"]
            row["total_marker_score"] = stats["total_marker_score"]
            row["finish_reason"] = result.finish_reason
            row["notes"] = " | ".join(result.notes)
            row["error_flag"] = False
            if config.enable_logprobs and result.logprobs is not None:
                row["logprobs"] = result.logprobs
            write_json(config.raw_dir / f"{run_id}.json", row)
            return row
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt <= retries:
                time.sleep(min(2**attempt, 20))

    row["error_flag"] = True
    row["notes"] = f"{type(last_error).__name__}: {last_error}"
    row["traceback"] = traceback.format_exc()
    write_json(config.raw_dir / f"{run_id}.json", row)
    return row


def _base_row(config: ExperimentConfig, prompt: dict, condition: str, repetition_id: int) -> dict:
    run_id = _run_id(prompt["id"], condition, repetition_id)
    return {
        "run_id": run_id,
        "prompt_id": prompt["id"],
        "condition": condition,
        "repetition_id": repetition_id,
        "timestamp": utc_timestamp(),
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


def _load_existing_success(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if row.get("error_flag"):
        return None
    if not row.get("full_output_text"):
        return None
    return row


def _run_id(prompt_id: str, condition: str, repetition_id: int) -> str:
    return f"{prompt_id}__{condition}__rep{repetition_id}"


if __name__ == "__main__":
    main()
