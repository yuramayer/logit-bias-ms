from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT / "new_experiment" / "outputs_vkr_max_mini" / "tables"
OUT_DIR = ROOT / "advisor_figures"

CONDITION_ORDER = ["control", "early", "mid", "late"]
COLORS = {
    "control": "#6B7280",
    "early": "#C2410C",
    "mid": "#2563EB",
    "late": "#059669",
}

FONT_REGULAR = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
BG = "#F7F7F5"
CARD = "#FFFFFF"
TEXT = "#111827"
SUBTLE = "#6B7280"
GRID = "#D1D5DB"
AXIS = "#9CA3AF"


@dataclass
class AggregateRow:
    condition: str
    run_count: int
    error_count: int
    total_marker_score_mean: float
    delta_p0_mean: float
    cosine_similarity_mean: float
    perplexity_mean: float


@dataclass
class PromptRow:
    prompt_id: str
    early_abs_delta_mean: float
    mid_abs_delta_mean: float
    late_abs_delta_mean: float
    ranking: str


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(path, size=size)


def measure(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[float, float]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def load_aggregate_rows() -> list[AggregateRow]:
    rows: list[AggregateRow] = []
    with (TABLES_DIR / "aggregated_by_condition.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                AggregateRow(
                    condition=row["condition"],
                    run_count=int(row["run_count"]),
                    error_count=int(row["error_count"]),
                    total_marker_score_mean=float(row["total_marker_score_mean"]),
                    delta_p0_mean=float(row["delta_p0_mean"]),
                    cosine_similarity_mean=float(row["cosine_similarity_mean"]),
                    perplexity_mean=float(row["perplexity_mean"]),
                )
            )
    rows.sort(key=lambda item: CONDITION_ORDER.index(item.condition))
    return rows


def load_prompt_rows() -> list[PromptRow]:
    rows: list[PromptRow] = []
    with (TABLES_DIR / "hypothesis_check.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                PromptRow(
                    prompt_id=row["prompt_id"],
                    early_abs_delta_mean=float(row["early_abs_delta_mean"]),
                    mid_abs_delta_mean=float(row["mid_abs_delta_mean"]),
                    late_abs_delta_mean=float(row["late_abs_delta_mean"]),
                    ranking=row["delta_strength_ranking"],
                )
            )
    rows.sort(key=lambda item: int(item.prompt_id[1:]))
    return rows


def rounded_rect(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], radius: int, fill: str, outline: str | None = None) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=1 if outline else 0)


def draw_page_header(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    title: str,
    subtitle: str,
) -> int:
    title_font = load_font(44, bold=True)
    subtitle_font = load_font(24)
    draw.text((x, y), title, font=title_font, fill=TEXT)
    _, title_h = measure(draw, title, title_font)
    draw.text((x, y + title_h + 10), subtitle, font=subtitle_font, fill=SUBTLE)
    _, subtitle_h = measure(draw, subtitle, subtitle_font)
    return y + title_h + subtitle_h + 30


def draw_card_title(draw: ImageDraw.ImageDraw, x: int, y: int, title: str, subtitle: str = "") -> int:
    title_font = load_font(26, bold=True)
    subtitle_font = load_font(18)
    draw.text((x, y), title, font=title_font, fill=TEXT)
    _, title_h = measure(draw, title, title_font)
    if subtitle:
        draw.text((x, y + title_h + 6), subtitle, font=subtitle_font, fill=SUBTLE)
        _, subtitle_h = measure(draw, subtitle, subtitle_font)
        return y + title_h + subtitle_h + 20
    return y + title_h + 12


def draw_horizontal_bar_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    title: str,
    subtitle: str,
    labels: list[str],
    values: list[float],
    colors: list[str],
    x_min: float,
    x_max: float,
    ticks: list[float],
    value_formatter,
    zero_line: bool = False,
) -> None:
    left, top, right, bottom = rect
    rounded_rect(draw, rect, radius=24, fill=CARD, outline="#E5E7EB")
    content_top = draw_card_title(draw, left + 28, top + 22, title, subtitle)

    label_font = load_font(20)
    value_font = load_font(18, bold=True)
    tick_font = load_font(16)

    chart_left = left + 120
    chart_right = right - 28
    chart_top = content_top + 10
    chart_bottom = bottom - 46
    n = len(labels)
    row_gap = (chart_bottom - chart_top) / n
    bar_height = min(30, row_gap * 0.56)

    def x_map(value: float) -> float:
        width = chart_right - chart_left
        return chart_left + (value - x_min) / (x_max - x_min) * width

    # grid + ticks
    for tick in ticks:
        x = x_map(tick)
        draw.line((x, chart_top - 2, x, chart_bottom + 8), fill=GRID, width=1)
        label = value_formatter(tick)
        label_w, label_h = measure(draw, label, tick_font)
        draw.text((x - label_w / 2, chart_bottom + 14), label, font=tick_font, fill=SUBTLE)

    if zero_line:
        zero_x = x_map(0.0)
        draw.line((zero_x, chart_top - 8, zero_x, chart_bottom + 8), fill=AXIS, width=2)

    for idx, (label, value, color) in enumerate(zip(labels, values, colors)):
        y_mid = chart_top + row_gap * idx + row_gap / 2
        y0 = y_mid - bar_height / 2
        y1 = y_mid + bar_height / 2

        label_w, label_h = measure(draw, label, label_font)
        draw.text((left + 28, y_mid - label_h / 2), label, font=label_font, fill=TEXT)

        start_x = x_map(0.0) if zero_line else chart_left
        end_x = x_map(value)
        bar_left = min(start_x, end_x)
        bar_right = max(start_x, end_x)
        rounded_rect(draw, (int(bar_left), int(y0), int(bar_right), int(y1)), radius=10, fill=color)

        value_text = value_formatter(value)
        value_w, value_h = measure(draw, value_text, value_font)
        text_x = bar_right + 10 if value >= 0 else bar_left - value_w - 10
        draw.text((text_x, y_mid - value_h / 2), value_text, font=value_font, fill=TEXT)


def draw_condition_metrics(rows: list[AggregateRow]) -> None:
    width, height = 2200, 1080
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)

    header_bottom = draw_page_header(
        draw,
        70,
        52,
        "Сводные метрики по условиям интервенции",
        "Профиль vkr_max_mini: control, early, mid, late",
    )

    card_gap = 32
    card_top = header_bottom + 10
    card_width = (width - 70 * 2 - card_gap * 2) // 3
    card_height = 760

    labels = [row.condition for row in rows]
    colors = [COLORS[row.condition] for row in rows]

    delta_values = [row.delta_p0_mean for row in rows]
    cosine_values = [row.cosine_similarity_mean for row in rows]
    perplexity_values = [row.perplexity_mean for row in rows]

    draw_horizontal_bar_panel(
        draw,
        (70, card_top, 70 + card_width, card_top + card_height),
        "Mean ΔP0",
        "Сдвиг по нормализованной представленности целевых маркеров",
        labels,
        delta_values,
        colors,
        x_min=min(delta_values) * 1.15,
        x_max=0.0004,
        ticks=[-0.004, -0.003, -0.002, -0.001, 0.0],
        value_formatter=lambda v: f"{v:.4f}",
        zero_line=True,
    )

    draw_horizontal_bar_panel(
        draw,
        (70 + card_width + card_gap, card_top, 70 + card_width * 2 + card_gap, card_top + card_height),
        "Mean cosine similarity",
        "Смысловая близость экспериментального и контрольного ответа",
        labels,
        cosine_values,
        colors,
        x_min=0.0,
        x_max=1.0,
        ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        value_formatter=lambda v: f"{v:.2f}",
    )

    draw_horizontal_bar_panel(
        draw,
        (70 + (card_width + card_gap) * 2, card_top, 70 + card_width * 3 + card_gap * 2, card_top + card_height),
        "Mean perplexity",
        "Языковая цена вмешательства",
        labels,
        perplexity_values,
        colors,
        x_min=350.0,
        x_max=750.0,
        ticks=[400, 500, 600, 700],
        value_formatter=lambda v: f"{v:.0f}",
    )

    image.save(OUT_DIR / "condition_metrics_overview.png")


def draw_legend(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    font = load_font(18)
    cursor_x = x
    for key in ["early", "mid", "late"]:
        draw.rounded_rectangle((cursor_x, y + 3, cursor_x + 18, y + 21), radius=5, fill=COLORS[key])
        draw.text((cursor_x + 28, y), key, font=font, fill=TEXT)
        cursor_x += 110


def draw_prompt_strength(rows: list[PromptRow]) -> None:
    width, height = 2200, 1400
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)

    header_bottom = draw_page_header(
        draw,
        70,
        52,
        "Prompt-level сила эффекта по ΔP0",
        "Среднее абсолютное значение ΔP0 для early, mid и late",
    )

    card = (70, header_bottom + 10, width - 70, height - 60)
    rounded_rect(draw, card, radius=24, fill=CARD, outline="#E5E7EB")
    content_top = draw_card_title(
        draw,
        card[0] + 28,
        card[1] + 22,
        "Сравнение по промптам",
        "Каждая строка показывает три значения: early, mid, late",
    )
    draw_legend(draw, card[2] - 420, card[1] + 28)

    label_font = load_font(20, bold=True)
    small_font = load_font(16)
    value_font = load_font(16, bold=True)

    chart_left = card[0] + 220
    chart_right = card[2] - 240
    chart_top = content_top + 12
    chart_bottom = card[3] - 36
    row_gap = (chart_bottom - chart_top) / len(rows)

    max_value = max(
        max(row.early_abs_delta_mean, row.mid_abs_delta_mean, row.late_abs_delta_mean)
        for row in rows
    )
    x_min = 0.0
    x_max = max_value * 1.15

    def x_map(value: float) -> float:
        width_px = chart_right - chart_left
        return chart_left + (value - x_min) / (x_max - x_min) * width_px

    tick_values = [0.0, 0.005, 0.010, 0.015, 0.020]
    for tick in tick_values:
        x = x_map(tick)
        draw.line((x, chart_top - 4, x, chart_bottom + 8), fill=GRID, width=1)
        text = f"{tick:.3f}"
        tw, th = measure(draw, text, small_font)
        draw.text((x - tw / 2, chart_bottom + 12), text, font=small_font, fill=SUBTLE)

    for idx, row in enumerate(rows):
        row_top = chart_top + idx * row_gap
        row_mid = row_top + row_gap / 2
        draw.line((card[0] + 28, row_top, card[2] - 28, row_top), fill="#F1F5F9", width=1)

        prompt_label = row.prompt_id.upper()
        draw.text((card[0] + 28, row_mid - 22), prompt_label, font=label_font, fill=TEXT)
        draw.text((card[0] + 28, row_mid + 4), row.ranking, font=small_font, fill=SUBTLE)

        series = [
            ("early", row.early_abs_delta_mean, -18),
            ("mid", row.mid_abs_delta_mean, 0),
            ("late", row.late_abs_delta_mean, 18),
        ]
        for name, value, offset in series:
            y = row_mid + offset
            x0 = chart_left
            x1 = x_map(value)
            draw.rounded_rectangle((int(x0), int(y - 8), int(x1), int(y + 8)), radius=6, fill=COLORS[name])
            value_text = f"{value:.4f}"
            tw, th = measure(draw, value_text, value_font)
            draw.text((x1 + 10, y - th / 2), value_text, font=value_font, fill=TEXT)

    draw.line((card[0] + 28, chart_bottom, card[2] - 28, chart_bottom), fill="#F1F5F9", width=1)
    image.save(OUT_DIR / "prompt_delta_strength.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    aggregate_rows = load_aggregate_rows()
    prompt_rows = load_prompt_rows()
    draw_condition_metrics(aggregate_rows)
    draw_prompt_strength(prompt_rows)
    print("Generated files:")
    for path in sorted(OUT_DIR.glob("*.png")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
