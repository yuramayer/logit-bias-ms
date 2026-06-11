# Scaled positional logit-bias experiment

This document describes the scaled experiment added after the thesis manuscript
was assembled. It keeps the same experimental logic as the manuscript:
`control / early / mid / late`, fixed generation parameters, the same marker
dictionary, and post-hoc comparison against control answers.

## Scale

- Prompt set: `data/prompts_vkr_scaled_100.json`
- Prompts: 100 Russian academic analytical prompts
- Conditions: `control`, `early`, `mid`, `late`
- Repetitions: 3 per prompt-condition pair
- Models: `gpt-4.1-mini`, `gpt-4.1-nano`
- New scaled generations: `100 x 4 x 3 x 2 = 2400`
- Scaled API errors: `0`
- Comparable max-profile generations already in the repository: `384`
- Total comparable max/scaled generations: `2784`

## Run commands

```bash
python3 experiment/tools/generate_scaled_prompts.py

python3 -m experiment.src.run_experiment_parallel \
  --config experiment/config_vkr_scaled_mini.yaml \
  --workers 10 \
  --retries 4 \
  --progress-every 25

python3 -m experiment.src.run_experiment_parallel \
  --config experiment/config_vkr_scaled_nano.yaml \
  --workers 10 \
  --retries 4 \
  --progress-every 25

python3 experiment/compare_profiles.py \
  experiment/outputs_vkr_max_mini \
  experiment/outputs_vkr_max_nano \
  experiment/outputs_vkr_scaled_mini \
  experiment/outputs_vkr_scaled_nano \
  --out experiment/profile_comparison_scaled.csv

python3 experiment/tools/summarize_scaled_results.py
```

The parallel runner is resumable: successful raw files are reused, so an
interrupted large run can be continued without restarting the whole profile.

## Main scaled results

Combined over the two scaled profiles:

| Condition | Runs | mean delta P0 | mean cosine similarity | mean proxy-perplexity |
|---|---:|---:|---:|---:|
| control | 600 | -0.000000 | 1.000000 | 768.116 |
| early | 600 | -0.002745 | 0.634811 | 1158.480 |
| mid | 600 | -0.001759 | 0.678999 | 1148.299 |
| late | 600 | -0.000479 | 0.703939 | 1105.217 |

The scaled result supports the main interpretation:

1. The position of logit-bias intervention changes the observed marker layer.
2. Early and mid intervention are stronger than late intervention on average.
3. Late intervention preserves higher similarity to control, so it is a softer
   and more local regime.
4. Prompt-level rankings remain heterogeneous, so H1 should be phrased as
   partially supported, while H2 is the more stable conclusion.

## Prompt-level summary

| Profile | Prompt cases | early wins | mid wins | late wins | early > late | mid > late |
|---|---:|---:|---:|---:|---:|---:|
| scaled mini | 100 | 42 | 36 | 22 | 58 | 60 |
| scaled nano | 100 | 35 | 30 | 35 | 46 | 47 |
| combined | 200 | 77 | 66 | 57 | 104 | 107 |

## Generated artifacts

- `scaled_defense_summary.md`
- `scaled_profile_summary.csv`
- `scaled_prompt_wins.csv`
- `profile_comparison_scaled.csv`
- `figures/scaled_delta_p0.svg`
- `figures/scaled_similarity_tradeoff.svg`
- `figures/scaled_prompt_wins.svg`
- `outputs_vkr_scaled_mini/raw/*.json`
- `outputs_vkr_scaled_nano/raw/*.json`
- `outputs_vkr_scaled_mini/tables/*.csv`
- `outputs_vkr_scaled_nano/tables/*.csv`
