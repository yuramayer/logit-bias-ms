# Out-of-domain prompt-transfer validation

This validation checks whether the positional logit-bias result survives when prompts move away from LLM/logit-bias methodology topics.

## Scale

- Prompt set: `data/prompts_vkr_ood_100.json`
- 100 out-of-domain Russian academic prompts
- Conditions: `control`, `early`, `mid`, `late`
- Repetitions: 3 per prompt-condition pair
- Models: `gpt-4.1-mini`, `Qwen/Qwen3.5-9B` through Together AI
- New OOD generations: `100 x 4 x 3 x 2 = 2400`
- API errors: `0` in both profiles

## Condition-level summary

| Profile | Condition | Runs | mean ΔP0 | mean cosine | length finish rate |
|---|---:|---:|---:|---:|---:|
| ood_mini | control | 300 | -0.000000 | 1.000000 | 0.007 |
| ood_mini | early | 300 | -0.009495 | 0.621769 | 0.103 |
| ood_mini | late | 300 | -0.004885 | 0.737432 | 0.963 |
| ood_mini | mid | 300 | -0.005657 | 0.702068 | 0.673 |
| ood_together_qwen | control | 300 | 0.000000 | 1.000000 | 0.640 |
| ood_together_qwen | early | 300 | -0.001368 | 0.405560 | 0.413 |
| ood_together_qwen | late | 300 | -0.002113 | 0.449490 | 0.980 |
| ood_together_qwen | mid | 300 | -0.001645 | 0.433512 | 0.983 |
| ood_combined | control | 600 | 0.000000 | 1.000000 | 0.323 |
| ood_combined | early | 600 | -0.005431 | 0.513664 | 0.258 |
| ood_combined | mid | 600 | -0.003651 | 0.567790 | 0.828 |
| ood_combined | late | 600 | -0.003499 | 0.593461 | 0.972 |

## Prompt-level summary

| Profile | Prompt cases | early wins | mid wins | late wins | early > late | mid > late | early > mid |
|---|---:|---:|---:|---:|---:|---:|---:|
| ood_mini | 100 | 63 | 21 | 16 | 71 | 60 | 73 |
| ood_together_qwen | 100 | 47 | 34 | 19 | 58 | 57 | 57 |
| ood_combined | 200 | 110 | 55 | 35 | 129 | 117 | 130 |

## Defense interpretation

The OOD battery strengthens the prompt-sampling argument. The original scaled battery was intentionally close to the thesis topic; this one preserves the academic genre but changes the domain.

Combined OOD result: early wins 110/200 prompt cases, mid wins 55/200, late wins 35/200. Early exceeds late in 129/200 prompt cases, while late has the highest average semantic similarity among intervention conditions.
