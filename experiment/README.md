# Эксперимент

Эта директория содержит код, конфиги, промпты, словари маркеров и уже полученные результаты эксперимента по позиционному применению `logit_bias`.

## Что делает код

Конвейер:

1. читает конфиг, промпты и словарь маркеров;
2. запускает условия `control`, `early`, `mid`, `late`;
3. применяет `segment_approximation` для позиционного включения `logit_bias`;
4. сохраняет `raw`-результаты по каждому запуску;
5. считает `delta_p0`, similarity и proxy-perplexity;
6. экспортирует итоговые таблицы в `outputs_*/tables/`.

## Основные профили

| Профиль | Назначение |
|---|---|
| `config_vkr_fast.yaml` | быстрый пилот |
| `config_vkr_plus.yaml` | расширенный промежуточный профиль |
| `config_vkr_max_mini.yaml` | основной профиль на `gpt-4.1-mini` |
| `config_vkr_max_nano.yaml` | проверка переносимости на `gpt-4.1-nano` |

В финальном анализе ВКР основной упор сделан на `outputs_vkr_max_mini`, а `outputs_vkr_max_nano` используется как дополнительная проверка устойчивости схемы.

## Запуск

```bash
python3 -m pip install -r experiment/requirements.txt

export OPENAI_API_KEY=...
python3 experiment/run_experiment.py --config experiment/config_vkr_max_mini.yaml
```

Для второй модели:

```bash
export OPENAI_API_KEY=...
python3 experiment/run_experiment.py --config experiment/config_vkr_max_nano.yaml
```

## Артефакты

В каждом `outputs_*` профиле:

- `raw/*.json` - один файл на запуск;
- `tables/raw_runs.csv` - плоская таблица всех запусков;
- `tables/raw_runs.jsonl` - полные raw-строки;
- `tables/aggregated_by_condition.csv` - агрегаты по условиям;
- `tables/hypothesis_check.csv` - prompt-level таблица для проверки гипотез;
- `tables/marker_category_comparison.csv` - вклад категорий маркеров;
- `logs/run_manifest.json` - короткий manifest профиля.

## Ограничения реализации

- `logit_bias` включается по сегментам, а не на каждом токене.
- `delta_p0` считается как `text_frequency_proxy`.
- Similarity считается как `token_count_cosine`.
- `perplexity` является proxy на основе биграммной модели по `control`-ответам.
