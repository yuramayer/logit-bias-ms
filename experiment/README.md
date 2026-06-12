# Эксперимент

Здесь лежит код эксперимента, конфиги, промпты, словари маркеров и результаты.

## Что делает код

Код делает простой pipeline:

1. читает конфиг;
2. читает промпты и словарь маркеров;
3. запускает `control`, `early`, `mid`, `late`;
4. сохраняет каждый ответ в `raw/*.json`;
5. считает метрики;
6. пишет таблицы в `outputs_*/tables/`.

## Профили

| Профиль | Для чего |
|---|---|
| `config_vkr_fast.yaml` | быстрый пилот |
| `config_vkr_plus.yaml` | расширенный пилот |
| `config_vkr_max_mini.yaml` | основной запуск на `gpt-4.1-mini` |
| `config_vkr_max_nano.yaml` | проверка на `gpt-4.1-nano` |
| `config_vkr_scaled_mini.yaml` | масштабированный запуск на `gpt-4.1-mini`: 100 промптов, 1200 генераций |
| `config_vkr_scaled_nano.yaml` | масштабированный запуск на `gpt-4.1-nano`: 100 промптов, 1200 генераций |

Основные результаты для ВКР лежат в `outputs_vkr_max_mini`. Проверка на второй модели лежит в `outputs_vkr_max_nano`.
Масштабированная проверка перед защитой лежит в `outputs_vkr_scaled_mini` и `outputs_vkr_scaled_nano`.

## Запуск

```bash
python3 -m pip install -r experiment/requirements.txt

export OPENAI_API_KEY=...
python3 experiment/run_experiment.py --config experiment/config_vkr_max_mini.yaml
```

Вторая модель:

```bash
export OPENAI_API_KEY=...
python3 experiment/run_experiment.py --config experiment/config_vkr_max_nano.yaml
```

Масштабированный запуск через resumable parallel runner:

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

python3 experiment/tools/summarize_scaled_results.py
```

## Масштабированный результат

- 100 русскоязычных академических промптов.
- 4 условия: `control`, `early`, `mid`, `late`.
- 3 повтора на каждое сочетание.
- 2 модели: `gpt-4.1-mini`, `gpt-4.1-nano`.
- 2400 новых генераций, `error_count = 0`.

Основные материалы:

- `SCALED_EXPERIMENT.md` - описание scaled-эксперимента;
- `SCALED_PRESENTATION_BLOCK.md` - готовый блок для слайдов;
- `scaled_defense_summary.md` - краткая сводка;
- `scaled_profile_summary.csv`, `scaled_prompt_wins.csv`, `profile_comparison_scaled.csv`;
- `figures/scaled_delta_p0.svg`, `figures/scaled_similarity_tradeoff.svg`,
  `figures/scaled_prompt_wins.svg`.

## Горизонтальное масштабирование

Feature audit по внешним провайдерам лежит в `HORIZONTAL_PROVIDER_AUDIT.md`.

Для запуска через OpenAI-compatible провайдеры добавлены template-конфиги:

- `config_template_fireworks_deepseek.yaml`;
- `config_template_together_qwen.yaml`;
- `config_template_openrouter_qwen.yaml`.

Перед полным запуском проверьте конкретную модель smoke-probe:

```bash
python3 experiment/tools/probe_provider_features.py \
  --provider fireworks \
  --model accounts/fireworks/models/REPLACE_WITH_MODEL_ID
```

Для Qwen/Llama/DeepSeek-family моделей используйте в конфиге
`tokenizer_backend: "huggingface"` и задайте совместимый `tokenizer_model`.

## Что лежит в outputs

В каждом `outputs_*`:

- `raw/*.json` - один файл на один ответ;
- `tables/raw_runs.csv` - все запуски в одной таблице;
- `tables/raw_runs.jsonl` - полные записи запусков;
- `tables/aggregated_by_condition.csv` - средние значения по режимам;
- `tables/hypothesis_check.csv` - проверка по промптам;
- `tables/marker_category_comparison.csv` - вклад категорий маркеров;
- `logs/run_manifest.json` - короткое описание запуска.

## Ограничения

- Bias включается по сегментам, не на каждом токене.
- `delta_p0` - частотный proxy по маркерам.
- Similarity - `token_count_cosine`.
- `perplexity` - proxy на биграммах из `control`-ответов.
