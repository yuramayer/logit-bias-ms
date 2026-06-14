# Материалы к защите ВКР

Эта страница — короткий навигатор по репозиторию к магистерской диссертации о позиционной чувствительности `logit bias` в LLM.

QR-код на слайдах ведет сюда, чтобы быстро открыть код, конфиги, данные и таблицы результатов. Текст ВКР и литературный обзор будут вынесены в отдельную ссылку.

## Оглавление

- [Что здесь лежит](#что-здесь-лежит)
- [Код и методика](#код-и-методика)
- [Данные и конфиги](#данные-и-конфиги)
- [Результаты](#результаты)
- [Связь со слайдами](#связь-со-слайдами)
- [Как воспроизвести](#как-воспроизвести)

## Что здесь лежит

Репозиторий содержит экспериментальный контур для проверки вопроса: зависит ли эффект `logit bias` от участка генерации, на котором он включается.

Быстрый маршрут:

| Что нужно проверить | Где открыть |
|---|---|
| Код генерации и включения `logit bias` | [`experiment/src/generation.py`](experiment/src/generation.py) |
| Метрики `ΔP0`, `total_marker_score`, `cosine similarity`, `proxy-perplexity` | [`experiment/src/metrics.py`](experiment/src/metrics.py) |
| Словари дискурсивных маркеров | [`experiment/data/markers_vkr_max.json`](experiment/data/markers_vkr_max.json), [`experiment/data/markers.json`](experiment/data/markers.json) |
| Промпты исходной серии ВКР | [`experiment/data/prompts_vkr_max.json`](experiment/data/prompts_vkr_max.json) |
| Промпты масштабированной профильной серии | [`experiment/data/prompts_vkr_scaled_100.json`](experiment/data/prompts_vkr_scaled_100.json) |
| Промпты нового домена | [`experiment/data/prompts_vkr_ood_100.json`](experiment/data/prompts_vkr_ood_100.json) |
| Сводка дополнительных экспериментов | [`experiment/POST_SUBMISSION_RESULTS.md`](experiment/POST_SUBMISSION_RESULTS.md) |

## Код и методика

Главная идея эксперимента: модель, промпты, параметры и словарь маркеров фиксируются; меняется только участок ответа, где применяется `logit bias`.

- [`experiment/src/generation.py`](experiment/src/generation.py) — генерация ответов и включение `logit bias` по сегментам ответа.
- [`experiment/src/config.py`](experiment/src/config.py) — чтение экспериментальных конфигов.
- [`experiment/src/markers.py`](experiment/src/markers.py) — загрузка и обработка словаря дискурсивных маркеров.
- [`experiment/src/metrics.py`](experiment/src/metrics.py) — расчет `total_marker_score`, `ΔP0`, `cosine similarity`, `proxy-perplexity`.
- [`experiment/src/run_experiment_parallel.py`](experiment/src/run_experiment_parallel.py) — параллельный запуск масштабированных серий.
- [`experiment/tools/summarize_scaled_results.py`](experiment/tools/summarize_scaled_results.py) — сбор агрегированных таблиц для расширенной проверки.

Режимы в коде называются `control`, `early`, `mid`, `late`. В презентации им соответствуют: контрольный, ранний, средний и поздний режим.

## Данные и конфиги

### Исходная серия ВКР

Первая проверка методики на `gpt-4.1-mini`:

- конфиг: [`experiment/config_vkr_max_mini.yaml`](experiment/config_vkr_max_mini.yaml);
- промпты: [`experiment/data/prompts_vkr_max.json`](experiment/data/prompts_vkr_max.json);
- словарь маркеров: [`experiment/data/markers_vkr_max.json`](experiment/data/markers_vkr_max.json);
- результаты: [`experiment/outputs_vkr_max_mini/`](experiment/outputs_vkr_max_mini/).

### OpenAI-масштабирование

Профильная проверка на 100 академических промптах:

- `gpt-4.1-mini`: [`experiment/config_vkr_scaled_mini.yaml`](experiment/config_vkr_scaled_mini.yaml);
- `gpt-4.1-nano`: [`experiment/config_vkr_scaled_nano.yaml`](experiment/config_vkr_scaled_nano.yaml);
- промпты: [`experiment/data/prompts_vkr_scaled_100.json`](experiment/data/prompts_vkr_scaled_100.json).

### Qwen

Проверка той же процедуры на другой модельной семье:

- конфиг: [`experiment/config_vkr_scaled_together_qwen.yaml`](experiment/config_vkr_scaled_together_qwen.yaml);
- модель: `Qwen/Qwen3.5-9B` через Together AI;
- сводка: [`experiment/scaled_defense_summary.md`](experiment/scaled_defense_summary.md).

### Новый домен

Дополнительная проверка на академических промптах о транспорте, образовании, здравоохранении, госуслугах, рынке труда и цифровых платформах:

- `gpt-4.1-mini`: [`experiment/config_vkr_ood_mini.yaml`](experiment/config_vkr_ood_mini.yaml);
- `Qwen/Qwen3.5-9B`: [`experiment/config_vkr_ood_together_qwen.yaml`](experiment/config_vkr_ood_together_qwen.yaml);
- промпты: [`experiment/data/prompts_vkr_ood_100.json`](experiment/data/prompts_vkr_ood_100.json);
- описание: [`experiment/OOD_PROMPT_TRANSFER.md`](experiment/OOD_PROMPT_TRANSFER.md).

## Результаты

### Исходная серия ВКР

- средние значения по режимам: [`experiment/outputs_vkr_max_mini/tables/aggregated_by_condition.csv`](experiment/outputs_vkr_max_mini/tables/aggregated_by_condition.csv);
- проверка по отдельным промптам: [`experiment/outputs_vkr_max_mini/tables/hypothesis_check.csv`](experiment/outputs_vkr_max_mini/tables/hypothesis_check.csv);
- вклад категорий маркеров: [`experiment/outputs_vkr_max_mini/tables/marker_category_comparison.csv`](experiment/outputs_vkr_max_mini/tables/marker_category_comparison.csv);
- сырые ответы: [`experiment/outputs_vkr_max_mini/raw/`](experiment/outputs_vkr_max_mini/raw/).

### Профильные 3600 генераций

- combined-агрегация по моделям и режимам: [`experiment/scaled_profile_summary.csv`](experiment/scaled_profile_summary.csv);
- режим с наибольшим снижением `ΔP0` по prompt-model случаям: [`experiment/scaled_prompt_wins.csv`](experiment/scaled_prompt_wins.csv);
- сравнение профилей: [`experiment/profile_comparison_scaled.csv`](experiment/profile_comparison_scaled.csv);
- короткая сводка для защиты: [`experiment/scaled_defense_summary.md`](experiment/scaled_defense_summary.md).

### Новый домен

- combined-агрегация: [`experiment/ood_profile_summary.csv`](experiment/ood_profile_summary.csv);
- режим с наибольшим снижением `ΔP0` по prompt-model случаям: [`experiment/ood_prompt_wins.csv`](experiment/ood_prompt_wins.csv);
- методическое описание: [`experiment/OOD_PROMPT_TRANSFER.md`](experiment/OOD_PROMPT_TRANSFER.md).

### Графики

- [`experiment/figures/scaled_delta_p0.svg`](experiment/figures/scaled_delta_p0.svg) — `ΔP0` по режимам;
- [`experiment/figures/scaled_similarity_tradeoff.svg`](experiment/figures/scaled_similarity_tradeoff.svg) — `cosine similarity` и цена вмешательства;
- [`experiment/figures/scaled_prompt_wins.svg`](experiment/figures/scaled_prompt_wins.svg) — распределение режимов по максимальному снижению `ΔP0`.

## Связь со слайдами

| Блок презентации | Где смотреть в репозитории |
|---|---|
| Механика `logit bias` | [`experiment/src/generation.py`](experiment/src/generation.py), [`experiment/config_vkr_max_mini.yaml`](experiment/config_vkr_max_mini.yaml) |
| Метрики | [`experiment/src/metrics.py`](experiment/src/metrics.py) |
| Дизайн эксперимента | [`experiment/README.md`](experiment/README.md), [`experiment/data/prompts_vkr_max.json`](experiment/data/prompts_vkr_max.json), [`experiment/data/markers_vkr_max.json`](experiment/data/markers_vkr_max.json) |
| Серия 1: ВКР | [`experiment/outputs_vkr_max_mini/tables/aggregated_by_condition.csv`](experiment/outputs_vkr_max_mini/tables/aggregated_by_condition.csv), [`experiment/outputs_vkr_max_mini/tables/hypothesis_check.csv`](experiment/outputs_vkr_max_mini/tables/hypothesis_check.csv) |
| OpenAI-масштабирование | [`experiment/config_vkr_scaled_mini.yaml`](experiment/config_vkr_scaled_mini.yaml), [`experiment/config_vkr_scaled_nano.yaml`](experiment/config_vkr_scaled_nano.yaml), [`experiment/scaled_prompt_wins.csv`](experiment/scaled_prompt_wins.csv) |
| Qwen | [`experiment/config_vkr_scaled_together_qwen.yaml`](experiment/config_vkr_scaled_together_qwen.yaml), [`experiment/scaled_defense_summary.md`](experiment/scaled_defense_summary.md) |
| H2 и цена вмешательства | [`experiment/scaled_profile_summary.csv`](experiment/scaled_profile_summary.csv), [`experiment/ood_profile_summary.csv`](experiment/ood_profile_summary.csv) |
| Новый домен | [`experiment/data/prompts_vkr_ood_100.json`](experiment/data/prompts_vkr_ood_100.json), [`experiment/ood_profile_summary.csv`](experiment/ood_profile_summary.csv), [`experiment/ood_prompt_wins.csv`](experiment/ood_prompt_wins.csv) |

## Как воспроизвести

Минимальная установка:

```bash
python3 -m pip install -r experiment/requirements.txt
```

Исходная серия ВКР:

```bash
export OPENAI_API_KEY=...
python3 experiment/run_experiment.py --config experiment/config_vkr_max_mini.yaml
```

OpenAI-масштабирование:

```bash
export OPENAI_API_KEY=...
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
```

Qwen через Together AI:

```bash
export TOGETHER_TOKEN=...
python3 -m experiment.src.run_experiment_parallel \
  --config experiment/config_vkr_scaled_together_qwen.yaml \
  --workers 8 \
  --retries 4 \
  --progress-every 50
```

Сбор сводных таблиц:

```bash
python3 experiment/tools/summarize_scaled_results.py
python3 experiment/tools/summarize_ood_results.py
```

Полное воспроизведение требует API-ключей `OPENAI_API_KEY` и `TOGETHER_TOKEN`. Для проверки структуры эксперимента можно читать конфиги, исходные промпты, словари маркеров, сырые ответы и готовые таблицы без запуска API.
