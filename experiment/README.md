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

Основные результаты для ВКР лежат в `outputs_vkr_max_mini`. Проверка на второй модели лежит в `outputs_vkr_max_nano`.

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
