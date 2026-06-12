# Post-submission extended experiment results

This document summarizes the additional empirical layer added after the thesis
manuscript was assembled.

## Total added scale

| Layer | Models | Generations | API errors |
|---|---:|---:|---:|
| In-domain scaled battery | 3 | 3600 | 0 |
| Out-of-domain prompt-transfer battery | 2 | 2400 | 0 |
| Total post-submission extension | 5 model profiles | 6000 | 0 |

The repository also contains the earlier max-profile runs, so the full
comparable experimental corpus is larger than the manuscript corpus.

## In-domain scaled battery

Prompt set: `data/prompts_vkr_scaled_100.json`

Design:

- 100 Russian academic prompts close to the thesis topic.
- 4 conditions: `control`, `early`, `mid`, `late`.
- 3 repetitions per prompt-condition pair.
- Models: `gpt-4.1-mini`, `gpt-4.1-nano`, `Qwen/Qwen3.5-9B` through Together AI.
- Total: `100 x 4 x 3 x 3 = 3600` generations.

Prompt-level result:

| Profile | Prompt cases | early wins | mid wins | late wins | early > late | mid > late |
|---|---:|---:|---:|---:|---:|---:|
| scaled_mini | 100 | 42 | 36 | 22 | 58 | 60 |
| scaled_nano | 100 | 35 | 30 | 35 | 46 | 47 |
| scaled_together_qwen | 100 | 44 | 40 | 16 | 61 | 68 |
| combined | 300 | 121 | 106 | 73 | 165 | 175 |

Interpretation: position matters across model profiles. The external Qwen run
shows that the effect is not only an OpenAI-specific artifact.

## Out-of-domain prompt-transfer battery

Prompt set: `data/prompts_vkr_ood_100.json`

Design:

- 100 Russian academic prompts from non-LLM domains.
- Same 4 conditions and 3 repetitions.
- Models: `gpt-4.1-mini`, `Qwen/Qwen3.5-9B` through Together AI.
- Total: `100 x 4 x 3 x 2 = 2400` generations.

Prompt-level result:

| Profile | Prompt cases | early wins | mid wins | late wins | early > late | mid > late |
|---|---:|---:|---:|---:|---:|---:|
| ood_mini | 100 | 63 | 21 | 16 | 71 | 60 |
| ood_together_qwen | 100 | 47 | 34 | 19 | 58 | 57 |
| ood_combined | 200 | 110 | 55 | 35 | 129 | 117 |

Interpretation: the prompt-transfer battery weakens the objection that the
result is caused only by prompts about LLMs, logit bias, or the thesis
methodology itself.

## Defense-ready conclusion

The manuscript experiment was expanded into a scaled validation corpus. Across
6000 additional generations, the main result is stable enough for defense:
the position of logit-bias intervention changes the observed marker layer.
Early and mid interventions more often produce the strongest marker shift,
while late intervention generally remains the softer regime by semantic
similarity.

The result should still be phrased as a controlled empirical validation, not as
a universal law for every task, model, or metric.

## Main files

- `SCALED_EXPERIMENT.md`
- `SCALED_PRESENTATION_BLOCK.md`
- `scaled_defense_summary.md`
- `OOD_PROMPT_TRANSFER.md`
- `scaled_profile_summary.csv`
- `scaled_prompt_wins.csv`
- `ood_profile_summary.csv`
- `ood_prompt_wins.csv`
- `outputs_vkr_scaled_together_qwen/`
- `outputs_vkr_ood_mini/`
- `outputs_vkr_ood_together_qwen/`
