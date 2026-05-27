from __future__ import annotations

"""Сравнение уже готовых профилей эксперимента.

Скрипт не запускает API-вызовы. Он только читает агрегированные CSV из разных
output-папок и собирает короткую сводку, чтобы быстро сравнить профили.
"""

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple experiment output directories.")
    parser.add_argument("outputs", nargs="+", help="Paths to output directories, e.g. new_experiment/outputs_vkr_plus")
    parser.add_argument(
        "--out",
        default="comparison_summary.csv",
        help="Path to write combined comparison CSV.",
    )
    args = parser.parse_args()

    rows = []
    for output_dir in args.outputs:
        output_path = Path(output_dir)
        aggregate_path = output_path / "tables" / "aggregated_by_condition.csv"
        if not aggregate_path.exists():
            raise FileNotFoundError(f"Missing aggregated file: {aggregate_path}")
        with aggregate_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row = dict(row)
                row["profile"] = output_path.name
                rows.append(row)

    if not rows:
        raise RuntimeError("No rows collected for comparison.")

    fieldnames = ["profile"] + [key for key in rows[0].keys() if key != "profile"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
