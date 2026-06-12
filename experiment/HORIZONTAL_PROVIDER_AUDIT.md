# Horizontal provider audit for scaled logit-bias experiment

Checked date: 2026-06-12.

Goal: find providers/models where the existing positional experiment can be
scaled horizontally without changing the core question. A usable provider must
accept `logit_bias`; `logprobs` is useful but not mandatory for the current
proxy metrics.

## Provider matrix

| Provider | Directly usable for this experiment? | Why |
| --- | --- | --- |
| OpenAI | yes | Already used; `logit_bias` works with `tiktoken` token ids. |
| Fireworks AI | yes, best next target | Chat completions schema includes `logit_bias`, `logprobs`, `top_logprobs`, and OpenAI-compatible endpoint. Use `huggingface` tokenizer backend for non-OpenAI models. |
| Together AI | yes, best next target | Chat completions schema includes `logit_bias` and `logprobs`. Use `huggingface` tokenizer backend. |
| OpenRouter | possible, second target | Request schema includes `logit_bias`, `logprobs`, and `top_logprobs`, but support can depend on routed model/provider. Must smoke-test a concrete model. |
| DeepSeek native API | no for direct replication | Official chat completion docs expose `logprobs`, but `logit_bias` is not present in the documented request schema. Use a DeepSeek-family model through Fireworks/OpenRouter instead. |
| Groq | no for this methodology | The API reference lists `logit_bias` and `logprobs`, but says they are not yet supported by any model. |
| Mistral / Anthropic / Gemini | not primary targets | They are useful comparison providers generally, but not clean direct targets for this exact `logit_bias` intervention unless their current endpoint exposes token-level bias. |

## Why tokenizer support matters

`logit_bias` maps token ids to bias values. Token ids are tokenizer-specific.

OpenAI models use `tiktoken`. Qwen, Llama, DeepSeek-family, Kimi, and similar
models should use their HuggingFace tokenizer ids. The runner now supports:

- `tokenizer_backend: "tiktoken"` for OpenAI configs.
- `tokenizer_backend: "huggingface"` plus `tokenizer_model` for external models.

Without this, a request can technically contain `logit_bias` but bias the wrong
tokens.

## Smoke-test before a full run

Examples:

```bash
export FIREWORKS_API_KEY=...
python experiment/tools/probe_provider_features.py \
  --provider fireworks \
  --model accounts/fireworks/models/REPLACE_WITH_MODEL_ID
```

```bash
export TOGETHER_API_KEY=...
python experiment/tools/probe_provider_features.py \
  --provider together \
  --model REPLACE_WITH_TOGETHER_MODEL_ID
```

```bash
export OPENROUTER_API_KEY=...
python experiment/tools/probe_provider_features.py \
  --provider openrouter \
  --model REPLACE_WITH_OPENROUTER_MODEL_ID
```

DeepSeek native API can also be probed, but the expectation is that `logit_bias`
will fail or be ignored because it is not documented:

```bash
export DEEPSEEK_API_KEY=...
python experiment/tools/probe_provider_features.py \
  --provider deepseek \
  --model deepseek-v4-flash
```

## Full-run pattern

After a probe succeeds:

1. Copy the closest template config.
2. Replace `model`.
3. Replace `tokenizer_model` with the matching HuggingFace tokenizer id.
4. Set a unique `outputs_dir`.
5. Run:

```bash
python -m experiment.src.run_experiment_parallel \
  --config experiment/config_vkr_scaled_PROVIDER_MODEL.yaml \
  --workers 6 \
  --retries 4 \
  --progress-every 25
```

Then regenerate summary tables:

```bash
python experiment/tools/summarize_scaled_results.py
```

## Defense framing

If one or two external providers are added, the presentation line becomes:

> After the submitted thesis text, the experiment was horizontally extended
> from OpenAI-only profiles to additional OpenAI-compatible providers. Provider
> selection was not arbitrary: candidates were first filtered by API-level
> support for token-level `logit_bias`, then run through the same 100-prompt,
> four-condition, three-repetition protocol.

