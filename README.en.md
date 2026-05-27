<div align="right">

| 🇬🇧 English | [🇷🇺 Русский](README.md) |
|---|---|

</div>

# Positional Sensitivity of Logit Bias in LLMs

This repo supports my master's thesis about `logit_bias`.

The question is simple: **does it matter where we turn on `logit_bias` during generation?** In my experiment, the answer was yes.

## TL;DR

The same `logit_bias` can have different effects in different parts of the answer.

In my runs:

- `early` and `mid` changed the answer structure more;
- `late` changed the text less;
- `late` kept meaning and text quality better;
- stronger steering had a higher cost.

Main takeaway: `logit_bias` has two important settings. The first one is the bias value. The second one is the position.

## What I Tested

LLMs generate text step by step. First tokens come first. Later tokens use the previous text as context.

So an early intervention can affect the rest of the answer. A late intervention usually affects only the ending.

I tested this on short Russian academic answers. I used a simple proxy: discourse markers.

Examples:

- `first`;
- `however`;
- `therefore`;
- `thus`;
- `probably`;
- `one may assume`.

This is not a full discourse parser. It is a simple measurable signal. It shows how the model uses transitions, conclusions, and hedging.

## Experiment Design

Each run used the same prompt set. There were four modes:

| Mode | What happens |
|---|---|
| `control` | normal answer, no `logit_bias` |
| `early` | `logit_bias` is active at the start |
| `mid` | `logit_bias` is active in the middle |
| `late` | `logit_bias` is active near the end |

The bias was negative. It suppressed selected discourse markers.

I used `segment_approximation`. The answer is split into parts. Bias is active only in one part. This is an approximation. It works with the normal Chat Completions API.

Main settings:

| Parameter | Value |
|---|---|
| Main model | `gpt-4.1-mini` |
| Second model | `gpt-4.1-nano` |
| Temperature | `0.3` |
| `top_p` | `1.0` |
| Max length | `180` tokens |
| Prompts | `12` |
| Repeats per mode | `4` |
| `bias_value` | `-8` |
| Main bias categories | `logical_structuring`, `hedging_epistemic` |

## Metrics

I tracked three things.

| Metric | Why it is used |
|---|---|
| `total_marker_score` | how many markers remain in the text |
| `delta_p0` | marker shift compared to `control` |
| `cosine_similarity` | how close the answer is to `control` |
| `perplexity` | proxy for the text quality cost |

These metrics should be read together. A strong marker shift is not enough. If the text becomes bad, the steering is not useful.

## Results

### `gpt-4.1-mini`

| Mode | Runs | Marker score | `delta_p0` | Similarity | Proxy perplexity |
|---|---:|---:|---:|---:|---:|
| `control` | 48 | 0.022198 | 0.000000 | 1.000000 | 430.657701 |
| `early` | 48 | 0.018377 | -0.003821 | 0.635795 | 708.251565 |
| `mid` | 48 | 0.018029 | -0.004170 | 0.683362 | 642.660427 |
| `late` | 48 | 0.020458 | -0.001741 | 0.707965 | 615.383812 |

Short version:

- `early` and `mid` reduced markers more;
- `late` reduced markers less;
- `late` kept higher similarity;
- `early` had a stronger effect, but also a higher cost.

### `gpt-4.1-nano`

| Mode | Runs | Marker score | `delta_p0` | Similarity | Proxy perplexity |
|---|---:|---:|---:|---:|---:|
| `control` | 48 | 0.017196 | 0.000000 | 1.000000 | 427.378569 |
| `early` | 48 | 0.016681 | -0.000515 | 0.614393 | 705.416285 |
| `mid` | 48 | 0.015500 | -0.001696 | 0.674498 | 639.396919 |
| `late` | 48 | 0.016551 | -0.000645 | 0.701033 | 608.995428 |

The second model shows a similar pattern. The effect is weaker, but position still matters. `late` is again the safer mode.

## Why This Matters for DS

If you use `logit_bias`, check two things: the bias value and the position.

- early bias can change the whole answer;
- mid bias can hit the main turn in the argument;
- late bias is usually local;
- stronger effect can make the text worse.

For production systems this matters. Sometimes you need a strong shift. Sometimes you only want to adjust the ending safely. These are different setups.

## Repository Structure

```text
.
├── README.md
├── README.en.md
├── experiment/
│   ├── src/                         # experiment code
│   ├── data/                        # prompts and marker dictionaries
│   ├── config_vkr_max_mini.yaml     # main profile
│   ├── config_vkr_max_nano.yaml     # second model profile
│   ├── outputs_vkr_max_mini/        # gpt-4.1-mini results
│   ├── outputs_vkr_max_nano/        # gpt-4.1-nano results
│   ├── outputs_vkr_plus/            # intermediate profile
│   ├── outputs_vkr_fast/            # fast pilot
│   └── profile_comparison.csv       # profile comparison
├── figures/                         # plots
└── thesis-md/                       # thesis text in Markdown
```

## Data

Main tables:

- [`experiment/outputs_vkr_max_mini/tables/aggregated_by_condition.csv`](experiment/outputs_vkr_max_mini/tables/aggregated_by_condition.csv) - main `gpt-4.1-mini` result;
- [`experiment/outputs_vkr_max_mini/tables/marker_category_comparison.csv`](experiment/outputs_vkr_max_mini/tables/marker_category_comparison.csv) - marker categories;
- [`experiment/outputs_vkr_max_mini/tables/hypothesis_check.csv`](experiment/outputs_vkr_max_mini/tables/hypothesis_check.csv) - prompt-level check;
- [`experiment/outputs_vkr_max_nano/tables/aggregated_by_condition.csv`](experiment/outputs_vkr_max_nano/tables/aggregated_by_condition.csv) - `gpt-4.1-nano` result;
- [`experiment/profile_comparison.csv`](experiment/profile_comparison.csv) - profile comparison.

Raw answers are in `raw/*.json` inside each `outputs_*` directory.

File name format:

```text
prompt_id__condition__repN.json
```

Example:

```text
p03__early__rep2.json
```

## Plots

![Condition metrics overview](figures/condition_metrics_overview.png)

![Prompt delta strength](figures/prompt_delta_strength.png)

## How to Run

You need:

- Python 3.11+
- `OPENAI_API_KEY`
- packages from `experiment/requirements.txt`

Install:

```bash
python3 -m pip install -r experiment/requirements.txt
```

Main profile:

```bash
export OPENAI_API_KEY=...
python3 experiment/run_experiment.py --config experiment/config_vkr_max_mini.yaml
```

Second model:

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

## Limits

- This is segment-based steering, not token-by-token steering.
- `delta_p0` is a marker frequency proxy.
- `perplexity` is a local bigram proxy.
- The data is short Russian academic answers.
- Other genres and models need separate tests.
