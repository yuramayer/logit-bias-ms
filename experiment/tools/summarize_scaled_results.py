from __future__ import annotations

"""Build defense-ready summaries and SVG charts for scaled experiment outputs."""

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILES = [
    ("scaled_mini", ROOT / "outputs_vkr_scaled_mini"),
    ("scaled_nano", ROOT / "outputs_vkr_scaled_nano"),
]
CONDITIONS = ("control", "early", "mid", "late")
INTERVENTIONS = ("early", "mid", "late")
COLORS = {
    "control": "#4b5563",
    "early": "#0f766e",
    "mid": "#2563eb",
    "late": "#b45309",
    "scaled_mini": "#2563eb",
    "scaled_nano": "#dc2626",
    "combined": "#111827",
}


def main() -> None:
    out_dir = ROOT / "figures"
    out_dir.mkdir(exist_ok=True)

    profile_rows = []
    prompt_rows = []
    combined_raw = []
    for profile_name, profile_dir in PROFILES:
        raw_rows = read_csv(profile_dir / "tables" / "raw_runs.csv")
        agg_rows = read_csv(profile_dir / "tables" / "aggregated_by_condition.csv")
        hyp_rows = read_csv(profile_dir / "tables" / "hypothesis_check.csv")
        combined_raw.extend(add_profile(raw_rows, profile_name))
        profile_rows.extend(profile_summary_rows(profile_name, raw_rows, agg_rows))
        prompt_rows.append(prompt_summary_row(profile_name, hyp_rows))

    combined_agg = aggregate_raw(combined_raw)
    profile_rows.extend(combined_agg)
    prompt_rows.append(prompt_summary_row_from_raw("combined", combined_raw))

    write_csv(ROOT / "scaled_profile_summary.csv", profile_rows)
    write_csv(ROOT / "scaled_prompt_wins.csv", prompt_rows)
    write_delta_chart(ROOT / "figures" / "scaled_delta_p0.svg", profile_rows)
    write_similarity_chart(ROOT / "figures" / "scaled_similarity_tradeoff.svg", profile_rows)
    write_prompt_wins_chart(ROOT / "figures" / "scaled_prompt_wins.svg", prompt_rows)
    write_markdown_summary(ROOT / "scaled_defense_summary.md", profile_rows, prompt_rows)
    print("wrote scaled summaries and figures")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_profile(rows: list[dict[str, str]], profile_name: str) -> list[dict]:
    output = []
    for row in rows:
        copied = dict(row)
        copied["profile"] = profile_name
        output.append(copied)
    return output


def profile_summary_rows(profile_name: str, raw_rows: list[dict[str, str]], agg_rows: list[dict[str, str]]) -> list[dict]:
    finish_counts = Counter((row["condition"], row["finish_reason"]) for row in raw_rows)
    total_by_condition = Counter(row["condition"] for row in raw_rows)
    output = []
    for row in agg_rows:
        condition = row["condition"]
        total = total_by_condition[condition]
        length_count = finish_counts[(condition, "length")]
        output.append(
            {
                "profile": profile_name,
                "condition": condition,
                "run_count": int(row["run_count"]),
                "error_count": int(row["error_count"]),
                "length_finish_rate": round(length_count / total, 6) if total else None,
                "token_count_mean": fnum(row["token_count_mean"]),
                "total_marker_score_mean": fnum(row["total_marker_score_mean"]),
                "delta_p0_mean": fnum(row["delta_p0_mean"]),
                "cosine_similarity_mean": fnum(row["cosine_similarity_mean"]),
                "perplexity_mean": fnum(row["perplexity_mean"]),
            }
        )
    return output


def aggregate_raw(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    output = []
    for condition in CONDITIONS:
        items = grouped[condition]
        length_count = sum(1 for row in items if row["finish_reason"] == "length")
        output.append(
            {
                "profile": "combined",
                "condition": condition,
                "run_count": len(items),
                "error_count": sum(1 for row in items if row["error_flag"] != "False"),
                "length_finish_rate": round(length_count / len(items), 6),
                "token_count_mean": mean_field(items, "token_count"),
                "total_marker_score_mean": mean_field(items, "total_marker_score"),
                "delta_p0_mean": mean_field(items, "delta_p0"),
                "cosine_similarity_mean": mean_field(items, "cosine_similarity"),
                "perplexity_mean": mean_field(items, "perplexity"),
            }
        )
    return output


def prompt_summary_row(profile_name: str, hyp_rows: list[dict[str, str]]) -> dict:
    wins = Counter()
    early_gt_late = 0
    mid_gt_late = 0
    early_gt_mid = 0
    for row in hyp_rows:
        values = {
            "early": float(row["early_abs_delta_mean"]),
            "mid": float(row["mid_abs_delta_mean"]),
            "late": float(row["late_abs_delta_mean"]),
        }
        wins[max(values, key=values.get)] += 1
        early_gt_late += values["early"] > values["late"]
        mid_gt_late += values["mid"] > values["late"]
        early_gt_mid += values["early"] > values["mid"]
    return {
        "profile": profile_name,
        "prompt_count": len(hyp_rows),
        "early_wins": wins["early"],
        "mid_wins": wins["mid"],
        "late_wins": wins["late"],
        "early_gt_late": early_gt_late,
        "mid_gt_late": mid_gt_late,
        "early_gt_mid": early_gt_mid,
    }


def prompt_summary_row_from_raw(profile_name: str, raw_rows: list[dict]) -> dict:
    by_key = defaultdict(list)
    for row in raw_rows:
        if row["condition"] == "control":
            continue
        key = (row["profile"], row["prompt_id"], row["condition"])
        by_key[key].append(abs(float(row["delta_p0"])))
    wins = Counter()
    early_gt_late = 0
    mid_gt_late = 0
    early_gt_mid = 0
    prompt_keys = sorted({(row["profile"], row["prompt_id"]) for row in raw_rows})
    for profile, prompt_id in prompt_keys:
        values = {
            condition: sum(by_key[(profile, prompt_id, condition)]) / len(by_key[(profile, prompt_id, condition)])
            for condition in INTERVENTIONS
        }
        wins[max(values, key=values.get)] += 1
        early_gt_late += values["early"] > values["late"]
        mid_gt_late += values["mid"] > values["late"]
        early_gt_mid += values["early"] > values["mid"]
    return {
        "profile": profile_name,
        "prompt_count": len(prompt_keys),
        "early_wins": wins["early"],
        "mid_wins": wins["mid"],
        "late_wins": wins["late"],
        "early_gt_late": early_gt_late,
        "mid_gt_late": mid_gt_late,
        "early_gt_mid": early_gt_mid,
    }


def mean_field(rows: list[dict], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field) not in {"", None}]
    return round(sum(values) / len(values), 6)


def fnum(value: str) -> float:
    return round(float(value), 6)


def rows_by_profile_condition(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["profile"], row["condition"]): row for row in rows}


def write_delta_chart(path: Path, rows: list[dict]) -> None:
    lookup = rows_by_profile_condition(rows)
    profiles = ["scaled_mini", "scaled_nano", "combined"]
    width, height = 920, 520
    margin = 70
    max_abs = max(abs(float(lookup[(profile, condition)]["delta_p0_mean"])) for profile in profiles for condition in INTERVENTIONS)
    scale = (height - 2 * margin) / max_abs
    baseline = margin
    bars = []
    group_width = 230
    bar_width = 42
    for pi, profile in enumerate(profiles):
        start_x = 120 + pi * group_width
        for ci, condition in enumerate(INTERVENTIONS):
            value = float(lookup[(profile, condition)]["delta_p0_mean"])
            bar_h = abs(value) * scale
            x = start_x + ci * 56
            y = baseline
            bars.append(rect(x, y, bar_width, bar_h, COLORS[condition]))
            bars.append(text(x + bar_width / 2, y + bar_h + 22, f"{value:.4f}", 13, "middle"))
        bars.append(text(start_x + 70, height - 28, profile.replace("_", " "), 16, "middle", "#111827", "600"))
    svg = svg_wrap(
        width,
        height,
        [
            text(width / 2, 36, "Mean delta P0 by condition", 22, "middle", "#111827", "700"),
            line(margin, baseline, width - margin, baseline, "#111827", 1.5),
            *bars,
            legend(690, 86, [("early", COLORS["early"]), ("mid", COLORS["mid"]), ("late", COLORS["late"])]),
            text(72, 72, "0", 13),
            text(72, height - 62, f"-{max_abs:.4f}", 13),
        ],
    )
    path.write_text(svg, encoding="utf-8")


def write_similarity_chart(path: Path, rows: list[dict]) -> None:
    lookup = rows_by_profile_condition(rows)
    profiles = ["scaled_mini", "scaled_nano", "combined"]
    width, height = 920, 520
    left, right, top, bottom = 90, 850, 70, 430
    points = []
    max_x = max(abs(float(lookup[(profile, condition)]["delta_p0_mean"])) for profile in profiles for condition in INTERVENTIONS)
    min_y = min(float(lookup[(profile, condition)]["cosine_similarity_mean"]) for profile in profiles for condition in INTERVENTIONS) - 0.02
    max_y = max(float(lookup[(profile, condition)]["cosine_similarity_mean"]) for profile in profiles for condition in INTERVENTIONS) + 0.02
    for profile in profiles:
        for condition in INTERVENTIONS:
            row = lookup[(profile, condition)]
            x_value = abs(float(row["delta_p0_mean"]))
            y_value = float(row["cosine_similarity_mean"])
            x = left + (x_value / max_x) * (right - left)
            y = bottom - ((y_value - min_y) / (max_y - min_y)) * (bottom - top)
            points.append(circle(x, y, 8, COLORS[condition]))
            points.append(text(x + 10, y - 8, f"{profile.replace('scaled_', '')}:{condition}", 12, "start"))
    svg = svg_wrap(
        width,
        height,
        [
            text(width / 2, 36, "Effect strength vs semantic similarity", 22, "middle", "#111827", "700"),
            line(left, bottom, right, bottom, "#111827", 1.5),
            line(left, top, left, bottom, "#111827", 1.5),
            text((left + right) / 2, 485, "abs(mean delta P0)", 15, "middle"),
            text(24, 250, "cosine similarity", 15, "middle", transform="rotate(-90 24 250)"),
            text(left, bottom + 22, "0", 12, "middle"),
            text(right, bottom + 22, f"{max_x:.4f}", 12, "middle"),
            text(left - 10, bottom, f"{min_y:.2f}", 12, "end"),
            text(left - 10, top, f"{max_y:.2f}", 12, "end"),
            *points,
        ],
    )
    path.write_text(svg, encoding="utf-8")


def write_prompt_wins_chart(path: Path, rows: list[dict]) -> None:
    width, height = 920, 520
    profiles = ["scaled_mini", "scaled_nano", "combined"]
    lookup = {row["profile"]: row for row in rows}
    max_count = max(int(lookup[profile]["prompt_count"]) for profile in profiles)
    left, top = 120, 80
    bar_h = 34
    gap = 58
    elements = [text(width / 2, 36, "Prompt-level abs(delta P0) winners", 22, "middle", "#111827", "700")]
    for idx, profile in enumerate(profiles):
        row = lookup[profile]
        x = left
        y = top + idx * 110
        elements.append(text(30, y + 24, profile.replace("_", " "), 15, "start", "#111827", "600"))
        for condition in INTERVENTIONS:
            count = int(row[f"{condition}_wins"])
            w = count / max_count * 650
            elements.append(rect(x, y, w, bar_h, COLORS[condition]))
            elements.append(text(x + w + 8, y + 22, f"{condition}: {count}", 14, "start"))
            y += gap
    elements.append(legend(650, 405, [("early", COLORS["early"]), ("mid", COLORS["mid"]), ("late", COLORS["late"])]))
    path.write_text(svg_wrap(width, height, elements), encoding="utf-8")


def write_markdown_summary(path: Path, profile_rows: list[dict], prompt_rows: list[dict]) -> None:
    lookup = rows_by_profile_condition(profile_rows)
    prompts = {row["profile"]: row for row in prompt_rows}
    lines = [
        "# Масштабированный эксперимент для защиты",
        "",
        "Этот слой добавлен после сборки текста ВКР как дополнительная масштабированная проверка той же экспериментальной схемы.",
        "",
        "## Масштаб",
        "",
        "- 100 русскоязычных академических промптов.",
        "- 4 условия: `control`, `early`, `mid`, `late`.",
        "- 3 повтора на каждое сочетание prompt x condition.",
        "- 2 модели: `gpt-4.1-mini` и `gpt-4.1-nano`.",
        "- Итого: 2400 новых генераций; `error_count = 0` в обоих scaled-профилях.",
        "- Вместе с двумя исходными max-профилями в репозитории получается 2784 генерации в сопоставимом дизайне.",
        "",
        "## Главные числа scaled-профиля",
        "",
        "| Профиль | Условие | Runs | mean ΔP0 | mean cosine | mean perplexity | length finish rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in ("scaled_mini", "scaled_nano", "combined"):
        for condition in CONDITIONS:
            row = lookup[(profile, condition)]
            lines.append(
                f"| {profile} | {condition} | {row['run_count']} | "
                f"{float(row['delta_p0_mean']):.6f} | {float(row['cosine_similarity_mean']):.6f} | "
                f"{float(row['perplexity_mean']):.3f} | {float(row['length_finish_rate']):.3f} |"
            )
    lines.extend(
        [
            "",
            "## Prompt-level устойчивость",
            "",
            "| Профиль | Prompt cases | early wins | mid wins | late wins | early > late | mid > late |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for profile in ("scaled_mini", "scaled_nano", "combined"):
        row = prompts[profile]
        lines.append(
            f"| {profile} | {row['prompt_count']} | {row['early_wins']} | {row['mid_wins']} | "
            f"{row['late_wins']} | {row['early_gt_late']} | {row['mid_gt_late']} |"
        )
    lines.extend(
        [
            "",
            "## Интерпретация для презентации",
            "",
            "1. Масштабированный профиль подтверждает, что позиция вмешательства меняет наблюдаемый маркерный слой.",
            "2. На `gpt-4.1-mini` раннее и срединное вмешательство дают почти одинаковый средний сдвиг и заметно превосходят позднее.",
            "3. На `gpt-4.1-nano` раннее вмешательство дает самый заметный средний отрицательный сдвиг, а позднее почти не меняет средний маркерный слой.",
            "4. В обеих scaled-моделях late сохраняет более высокую semantic similarity, то есть слабее перестраивает ответ.",
            "5. Prompt-level ранжирование не сводится к простой формуле `early > mid > late`, поэтому H1 лучше формулировать как частичную, а H2 как более устойчивую.",
            "",
            "## Файлы",
            "",
            "- `new_experiment/scaled_profile_summary.csv`",
            "- `new_experiment/scaled_prompt_wins.csv`",
            "- `new_experiment/profile_comparison_scaled.csv`",
            "- `new_experiment/figures/scaled_delta_p0.svg`",
            "- `new_experiment/figures/scaled_similarity_tradeoff.svg`",
            "- `new_experiment/figures/scaled_prompt_wins.svg`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def svg_wrap(width: int, height: int, body: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#ffffff"/>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def rect(x: float, y: float, w: float, h: float, fill: str) -> str:
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" rx="2"/>'


def line(x1: float, y1: float, x2: float, y2: float, stroke: str, width: float) -> str:
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width}"/>'


def circle(x: float, y: float, r: float, fill: str) -> str:
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}"/>'


def text(
    x: float,
    y: float,
    value: str,
    size: int,
    anchor: str = "start",
    fill: str = "#111827",
    weight: str = "400",
    transform: str | None = None,
) -> str:
    transform_attr = f' transform="{transform}"' if transform else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}"{transform_attr}>{escape(value)}</text>'
    )


def legend(x: float, y: float, items: list[tuple[str, str]]) -> str:
    output = []
    for idx, (label, color) in enumerate(items):
        yy = y + idx * 26
        output.append(rect(x, yy - 13, 16, 16, color))
        output.append(text(x + 24, yy, label, 14))
    return "\n".join(output)


def escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    main()
