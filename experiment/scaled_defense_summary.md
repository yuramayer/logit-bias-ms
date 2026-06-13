# Масштабированный эксперимент для защиты

Этот слой добавлен после сборки текста ВКР как дополнительная масштабированная проверка той же экспериментальной схемы.

## Масштаб

- 100 русскоязычных академических промптов.
- 4 условия: `control`, `early`, `mid`, `late`.
- 3 повтора на каждое сочетание prompt x condition.
- 3 модели: `gpt-4.1-mini`, `gpt-4.1-nano`, `Qwen/Qwen3.5-9B` через Together AI.
- Итого: 3600 новых генераций; `error_count = 0` во всех трех scaled-профилях.
- Вместе с двумя исходными max-профилями в репозитории получается 3984 генерации в сопоставимом дизайне.

## Главные числа scaled-профиля

| Профиль | Условие | Runs | mean ΔP0 | mean cosine | mean perplexity | length finish rate |
|---|---:|---:|---:|---:|---:|---:|
| scaled_mini | control | 300 | -0.000000 | 1.000000 | 793.055 | 0.093 |
| scaled_mini | early | 300 | -0.003198 | 0.633223 | 1222.430 | 0.207 |
| scaled_mini | mid | 300 | -0.003127 | 0.682370 | 1194.594 | 0.720 |
| scaled_mini | late | 300 | -0.000954 | 0.711684 | 1149.292 | 0.983 |
| scaled_nano | control | 300 | 0.000000 | 1.000000 | 743.177 | 0.143 |
| scaled_nano | early | 300 | -0.002293 | 0.636399 | 1094.530 | 0.313 |
| scaled_nano | mid | 300 | -0.000391 | 0.675628 | 1102.004 | 0.810 |
| scaled_nano | late | 300 | -0.000004 | 0.696194 | 1061.142 | 0.977 |
| scaled_together_qwen | control | 300 | 0.000000 | 1.000000 | 1716.487 | 0.600 |
| scaled_together_qwen | early | 300 | -0.002442 | 0.383407 | 2948.881 | 0.423 |
| scaled_together_qwen | mid | 300 | -0.003360 | 0.410614 | 2841.825 | 1.000 |
| scaled_together_qwen | late | 300 | -0.003179 | 0.421751 | 2819.539 | 0.963 |
| combined | control | 900 | 0.000000 | 1.000000 | 1084.240 | 0.279 |
| combined | early | 900 | -0.002644 | 0.551010 | 1755.281 | 0.314 |
| combined | mid | 900 | -0.002293 | 0.589537 | 1712.808 | 0.843 |
| combined | late | 900 | -0.001379 | 0.609876 | 1676.658 | 0.974 |

## Prompt-level устойчивость

| Профиль | Prompt cases | early wins | mid wins | late wins | early > late | mid > late |
|---|---:|---:|---:|---:|---:|---:|
| scaled_mini | 100 | 42 | 36 | 22 | 58 | 60 |
| scaled_nano | 100 | 35 | 30 | 35 | 46 | 47 |
| scaled_together_qwen | 100 | 44 | 40 | 16 | 61 | 68 |
| combined | 300 | 121 | 106 | 73 | 165 | 175 |

## Интерпретация для презентации

1. Масштабированный профиль подтверждает, что позиция вмешательства меняет наблюдаемый маркерный слой.
2. Внешний профиль `Qwen/Qwen3.5-9B` подтверждает, что эффект не сводится к OpenAI-only артефакту.
3. По prompt-level ранжированию `early` чаще всего дает максимальный абсолютный сдвиг: особенно на `gpt-4.1-mini` и `Qwen/Qwen3.5-9B`.
4. Late обычно сохраняет более высокую semantic similarity, то есть слабее перестраивает ответ.
5. Prompt-level ранжирование не сводится к простой формуле `early > mid > late`, поэтому H1 лучше формулировать как частичную, а H2 как более устойчивую.

## Файлы

- `experiment/scaled_profile_summary.csv`
- `experiment/scaled_prompt_wins.csv`
- `experiment/profile_comparison_scaled.csv`
- `experiment/figures/scaled_delta_p0.svg`
- `experiment/figures/scaled_similarity_tradeoff.svg`
- `experiment/figures/scaled_prompt_wins.svg`
