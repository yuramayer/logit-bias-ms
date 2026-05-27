<div align="right">

| 🇬🇧 English | [🇷🇺 Русский](README.md) |
|---|---|

</div>

# Positional Sensitivity of Logit Bias in LLMs

This is a public repository for my master's thesis. The thesis studies how the position of `logit_bias` changes the observable discourse tone of an LLM answer.

## TL;DR

Short thesis: the same `logit_bias` works differently depending on where it is applied during generation. Early and middle intervention changes the marker layer more strongly. Late intervention is usually softer and more local.

Practical takeaway for data scientists: controlled generation should not be evaluated only as "we added bias and got a different text". You also need to check the position of the intervention, the cost for text quality, and meaning preservation. In my experiments, a stronger shift in discourse markers often came with a higher text quality cost. So `late` was safer but weaker, while `early` and `mid` were more sensitive but more risky.

## What Was Studied

The work checks whether the moment of applying `logit_bias` changes the surface marker layer of a Russian academic answer.

I used fixed categories of discourse markers as a proxy for discourse tone:

- `logical_structuring`: Russian markers like "first", "however", "therefore", "thus";
- `hedging_epistemic`: Russian markers like "probably", "possibly", "one may assume";
- extra categories for side effects: `argument_development`, `intensification_assertiveness`, `anglocentric_formulas`.

The core idea is simple: if text is generated step by step, then an intervention at the start, in the middle, and near the end should not always have the same effect. The start often sets the frame. The middle develops the argument. The end usually closes an already formed trajectory.

## Experiment Design

The main experiment compares four conditions:

| Condition | What happens |
|---|---|
| `control` | normal generation without `logit_bias` |
| `early` | negative `logit_bias` is applied at the start of the answer |
| `mid` | negative `logit_bias` is applied in the middle of the answer |
| `late` | negative `logit_bias` is applied near the end of the answer |

The positional intervention is implemented with `segment_approximation`: the answer is split into segments, and `logit_bias` is active only in the target segment. This is not true token-by-token steering, but it is reproducible and practical with the standard Chat Completions API.

Main configuration:

| Parameter | Value |
|---|---|
| Main model | `gpt-4.1-mini` |
| Extra check | `gpt-4.1-nano` |
| Temperature | `0.3` |
| `top_p` | `1.0` |
| Max length | `180` tokens |
| Repetitions per condition | `4` |
| Prompts | `12` |
| `bias_value` | `-8` |
| Main bias categories | `logical_structuring`, `hedging_epistemic` |
| Segments | `early_end = 0.25`, `mid = 0.40-0.60`, `late_start = 0.75` |

## Metrics

The experiment separates three different things:

| Metric | Meaning |
|---|---|
| `total_marker_score` | normalized frequency of discourse markers |
| `delta_p0` | marker shift compared to `control` for the same prompt |
| `cosine_similarity` | closeness to the control answer by token counts |
| `perplexity` | proxy for text quality cost, based on a bigram model trained on `control` |

The metrics should be read together. A strong marker shift is not enough if the text loses meaning or becomes unstable.

## Main Results

### Main Profile: `gpt-4.1-mini`

| Condition | Runs | Marker score | `delta_p0` | Similarity | Proxy perplexity |
|---|---:|---:|---:|---:|---:|
| `control` | 48 | 0.022198 | 0.000000 | 1.000000 | 430.657701 |
| `early` | 48 | 0.018377 | -0.003821 | 0.635795 | 708.251565 |
| `mid` | 48 | 0.018029 | -0.004170 | 0.683362 | 642.660427 |
| `late` | 48 | 0.020458 | -0.001741 | 0.707965 | 615.383812 |

What this shows:

- all intervention modes reduce the average marker layer compared to `control`;
- `early` and `mid` create a stronger shift than `late`;
- `late` keeps better similarity and has a lower text quality cost;
- `early` has a strong effect, but it costs more in text stability;
- most of the effect comes from `logical_structuring`.

### Check on `gpt-4.1-nano`

| Condition | Runs | Marker score | `delta_p0` | Similarity | Proxy perplexity |
|---|---:|---:|---:|---:|---:|
| `control` | 48 | 0.017196 | 0.000000 | 1.000000 | 427.378569 |
| `early` | 48 | 0.016681 | -0.000515 | 0.614393 | 705.416285 |
| `mid` | 48 | 0.015500 | -0.001696 | 0.674498 | 639.396919 |
| `late` | 48 | 0.016551 | -0.000645 | 0.701033 | 608.995428 |

The second model keeps the same general pattern: the position of intervention matters, `late` is more conservative, and `early`/`mid` tend to affect the answer structure more strongly.

## Main Thesis Result

The position of `logit_bias` is a real experimental variable. In this setup, late intervention is usually weaker and more local. Early and middle intervention change the discourse marker layer more strongly. But stronger control has a cost: a larger marker shift also increases the risk of lower text stability.

This is useful for production LLM work and DS experiments with controlled generation:

- `logit_bias` should not be treated only as a global style switch;
- the position of the intervention may matter as much as the bias value;
- early steering can change the rhetorical trajectory, but it is often more expensive;
- late steering may be safer when you only need to adjust the ending;
- evaluation should include shift strength, meaning preservation, and text quality cost.

## Repository Structure

```text
.
├── README.md
├── README.en.md
├── experiment/
│   ├── src/                         # experiment code
│   ├── data/                        # prompts and marker dictionaries
│   ├── config_vkr_max_mini.yaml     # main gpt-4.1-mini profile
│   ├── config_vkr_max_nano.yaml     # extra gpt-4.1-nano profile
│   ├── outputs_vkr_max_mini/        # raw, tables, logs for the main profile
│   ├── outputs_vkr_max_nano/        # raw, tables, logs for the nano profile
│   ├── outputs_vkr_plus/            # intermediate extended profile
│   ├── outputs_vkr_fast/            # fast pilot
│   └── profile_comparison.csv       # profile comparison
├── figures/                         # quick visual result overview
└── thesis-md/                       # thesis text in Markdown sections
```

## Where to Find Results

Main tables:

- [`experiment/outputs_vkr_max_mini/tables/aggregated_by_condition.csv`](experiment/outputs_vkr_max_mini/tables/aggregated_by_condition.csv) - main condition-level table for `gpt-4.1-mini`;
- [`experiment/outputs_vkr_max_mini/tables/marker_category_comparison.csv`](experiment/outputs_vkr_max_mini/tables/marker_category_comparison.csv) - marker category contribution;
- [`experiment/outputs_vkr_max_mini/tables/hypothesis_check.csv`](experiment/outputs_vkr_max_mini/tables/hypothesis_check.csv) - prompt-level hypothesis check;
- [`experiment/outputs_vkr_max_nano/tables/aggregated_by_condition.csv`](experiment/outputs_vkr_max_nano/tables/aggregated_by_condition.csv) - check on `gpt-4.1-nano`;
- [`experiment/profile_comparison.csv`](experiment/profile_comparison.csv) - comparison of all profiles.

Raw answers are stored as `raw/*.json` inside each output profile. Each file is one run:

```text
prompt_id__condition__repN.json
```

Example: `p03__early__rep2.json`.

## Quick Visual Overview

![Condition metrics overview](figures/condition_metrics_overview.png)

![Prompt delta strength](figures/prompt_delta_strength.png)

## How to Run

Requirements:

- Python 3.11+
- `OPENAI_API_KEY`
- dependencies from `experiment/requirements.txt`

Install dependencies:

```bash
python3 -m pip install -r experiment/requirements.txt
```

Run the main profile:

```bash
export OPENAI_API_KEY=...
python3 experiment/run_experiment.py --config experiment/config_vkr_max_mini.yaml
```

Run the second model check:

```bash
export OPENAI_API_KEY=...
python3 experiment/run_experiment.py --config experiment/config_vkr_max_nano.yaml
```

Compare profiles:

```bash
python3 experiment/compare_profiles.py \
  experiment/outputs_vkr_fast \
  experiment/outputs_vkr_plus \
  experiment/outputs_vkr_max_mini \
  experiment/outputs_vkr_max_nano \
  --out experiment/profile_comparison.csv
```

## Limitations

- Positional control uses `segment_approximation`, not true token-by-token decoding.
- `delta_p0` is a marker frequency proxy.
- `perplexity` is a local bigram proxy trained on `control`.
- The main data is Russian academic short answers, so other genres need separate testing.
