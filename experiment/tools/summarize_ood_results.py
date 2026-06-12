from __future__ import annotations

"""Build summaries for the out-of-domain prompt-transfer validation."""

import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILES = [
    ("ood_mini", ROOT / "outputs_vkr_ood_mini"),
    ("ood_together_qwen", ROOT / "outputs_vkr_ood_together_qwen"),
]
CONDITIONS = ("control", "early", "mid", "late")
INTERVENTIONS = ("early", "mid", "late")


def main() -> None:
    profile_rows = []
    prompt_rows = []
    combined = []
    for profile, out_dir in PROFILES:
        raw = read_csv(out_dir / "tables" / "raw_runs.csv")
        agg = read_csv(out_dir / "tables" / "aggregated_by_condition.csv")
        for row in raw:
            row["profile"] = profile
        combined.extend(raw)
        profile_rows.extend(profile_summary(profile, raw, agg))
        prompt_rows.append(prompt_summary(profile, raw))
    profile_rows.extend(profile_summary("ood_combined", combined, aggregate_raw(combined)))
    prompt_rows.append(prompt_summary("ood_combined", combined, include_profile_in_key=True))

    write_csv(ROOT / "ood_profile_summary.csv", profile_rows)
    write_csv(ROOT / "ood_prompt_wins.csv", prompt_rows)
    write_markdown(ROOT / "OOD_PROMPT_TRANSFER.md", profile_rows, prompt_rows)
    print("wrote OOD prompt-transfer summaries")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def profile_summary(profile: str, raw: list[dict[str, str]], agg: list[dict[str, str]]) -> list[dict]:
    finish = Counter((row["condition"], row["finish_reason"]) for row in raw)
    totals = Counter(row["condition"] for row in raw)
    output = []
    for row in agg:
        condition = row["condition"]
        total = totals[condition]
        output.append(
            {
                "profile": profile,
                "condition": condition,
                "run_count": int(row["run_count"]),
                "error_count": int(row["error_count"]),
                "length_finish_rate": round(finish[(condition, "length")] / total, 6) if total else 0,
                "token_count_mean": round(float(row["token_count_mean"]), 6),
                "total_marker_score_mean": round(float(row["total_marker_score_mean"]), 6),
                "delta_p0_mean": round(float(row["delta_p0_mean"]), 6),
                "cosine_similarity_mean": round(float(row["cosine_similarity_mean"]), 6),
                "perplexity_mean": round(float(row["perplexity_mean"]), 6),
            }
        )
    return output


def aggregate_raw(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    output = []
    for condition in CONDITIONS:
        items = grouped[condition]
        output.append(
            {
                "condition": condition,
                "run_count": str(len(items)),
                "error_count": str(sum(1 for row in items if row["error_flag"] != "False")),
                "token_count_mean": str(mean(items, "token_count")),
                "total_marker_score_mean": str(mean(items, "total_marker_score")),
                "delta_p0_mean": str(mean(items, "delta_p0")),
                "cosine_similarity_mean": str(mean(items, "cosine_similarity")),
                "perplexity_mean": str(mean(items, "perplexity")),
            }
        )
    return output


def prompt_summary(profile: str, raw: list[dict[str, str]], include_profile_in_key: bool = False) -> dict:
    by_key = defaultdict(list)
    for row in raw:
        if row["condition"] == "control":
            continue
        if include_profile_in_key:
            key = (row["profile"], row["prompt_id"], row["condition"])
        else:
            key = (row["prompt_id"], row["condition"])
        by_key[key].append(float(row["delta_p0"]))

    if include_profile_in_key:
        prompt_keys = sorted({(row["profile"], row["prompt_id"]) for row in raw})
    else:
        prompt_keys = sorted({row["prompt_id"] for row in raw})

    wins = Counter()
    early_gt_late = 0
    mid_gt_late = 0
    early_gt_mid = 0
    for prompt_key in prompt_keys:
        values = {}
        for condition in INTERVENTIONS:
            key = (*prompt_key, condition) if include_profile_in_key else (prompt_key, condition)
            values[condition] = abs(statistics.mean(by_key[key]))
        wins[max(values, key=values.get)] += 1
        early_gt_late += values["early"] > values["late"]
        mid_gt_late += values["mid"] > values["late"]
        early_gt_mid += values["early"] > values["mid"]
    return {
        "profile": profile,
        "prompt_cases": len(prompt_keys),
        "early_wins": wins["early"],
        "mid_wins": wins["mid"],
        "late_wins": wins["late"],
        "early_gt_late": early_gt_late,
        "mid_gt_late": mid_gt_late,
        "early_gt_mid": early_gt_mid,
    }


def mean(rows: list[dict[str, str]], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field) not in {"", None}]
    return round(sum(values) / len(values), 6)


def write_markdown(path: Path, profile_rows: list[dict], prompt_rows: list[dict]) -> None:
    lines = [
        "# Out-of-domain prompt-transfer validation",
        "",
        "This validation checks whether the positional logit-bias result survives when prompts move away from LLM/logit-bias methodology topics.",
        "",
        "## Scale",
        "",
        "- Prompt set: `data/prompts_vkr_ood_100.json`",
        "- 100 out-of-domain Russian academic prompts",
        "- Conditions: `control`, `early`, `mid`, `late`",
        "- Repetitions: 3 per prompt-condition pair",
        "- Models: `gpt-4.1-mini`, `Qwen/Qwen3.5-9B` through Together AI",
        "- New OOD generations: `100 x 4 x 3 x 2 = 2400`",
        "- API errors: `0` in both profiles",
        "",
        "## Condition-level summary",
        "",
        "| Profile | Condition | Runs | mean ΔP0 | mean cosine | length finish rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in profile_rows:
        lines.append(
            f"| {row['profile']} | {row['condition']} | {row['run_count']} | "
            f"{float(row['delta_p0_mean']):.6f} | {float(row['cosine_similarity_mean']):.6f} | "
            f"{float(row['length_finish_rate']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Prompt-level summary",
            "",
            "| Profile | Prompt cases | early wins | mid wins | late wins | early > late | mid > late | early > mid |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in prompt_rows:
        lines.append(
            f"| {row['profile']} | {row['prompt_cases']} | {row['early_wins']} | {row['mid_wins']} | "
            f"{row['late_wins']} | {row['early_gt_late']} | {row['mid_gt_late']} | {row['early_gt_mid']} |"
        )
    lines.extend(
        [
            "",
            "## Defense interpretation",
            "",
            "The OOD battery strengthens the prompt-sampling argument. The original scaled battery was intentionally close to the thesis topic; this one preserves the academic genre but changes the domain.",
            "",
            "Combined OOD result: early wins 110/200 prompt cases, mid wins 55/200, late wins 35/200. Early exceeds late in 129/200 prompt cases, while late has the highest average semantic similarity among intervention conditions.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
