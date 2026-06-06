# Evaluation for Geoscience LLM

This folder contains the evaluation scripts for language model taking geoscience exams.

```
.
├── multiple_choice_samples_wa.txt # prompt (with 'the answer is') for 5-shot eval
├── multiple_choice_samples.txt # naive prompt for 5-shot eval
└── post_process.py # benchmark preprocessing scripts
└── memtra # Memorizing transformers
└── memtra # Memorizing transformers
```

**We will release end to end version at the end of October, along with Geo-Eval**

## Usage

-> Here is an example:

- For k2 series model with lora
```bash
python run_eval.py --model_name k2_ni --base_model /home/daven/llm/qokori/llama-2023-05-07-15-10/checkpoint/ --lora_weights /home/daven/llm/qokori/qokori-sft/outputs/geo_llama/
```

- For k2 series model without lora
```bash
python run_eval.py --model_name geollama --base_model /home/daven/llm/qokori/llama-2023-05-07-15-10/checkpoint/
```

- For non-k2 series model
```bash 
python run_eval.py --model_name gpt2_xl --base_model gpt2-xl
```

## vLLM / OpenAI-compatible endpoints

This is the recommended lightweight workflow for this checkout. After cloning on
a server, run `uv sync` once, then run evaluator commands with `uv run`.

There are two independent vLLM evaluation paths:

1. `run_eval_vllm.py` keeps the original GeoBench objective-task style by asking
   the completions endpoint for one next-token answer among the valid labels
   (`A/B/C/...` or `True/False`) with logprobs and `allowed_token_ids`.
2. `run_eval_vllm_chat.py` uses the OpenAI Python SDK `chat.completions` API,
   generates a full response, strips any text before the final `</think>` tag,
   parses the answer label, and compares it with the GeoBench ground truth.

### Chat-completions evaluator

Use this path for modern chat or thinking models served by vLLM:

```bash
uv run python evaluation/run_eval_vllm_chat.py \
  --model qwen3-4b-instruct-2507 \
  --base-url http://SERVER:8000/v1 \
  --benchmark all \
  --prompt-variant both
```

Useful smoke test before calling the endpoint:

```bash
uv run python evaluation/run_eval_vllm_chat.py \
  --model qwen3-4b-instruct-2507 \
  --benchmark npee \
  --tasks choice \
  --limit 2 \
  --dry-run \
  --print-sample-prompts 1
```

`--prompt-variant woa` keeps the original no-answer-prefix prompt ending in
`### Response:`. `--prompt-variant wa` keeps the original answer-prefix prompt
ending with:

```text
### Response:
The answer is:
```

`both` runs both variants.

### Next-token logprob evaluator

Use this path when you want the old constrained next-token scoring behavior:

```bash
uv run python evaluation/run_eval_vllm.py \
  --model qwen3-4b-instruct-2507 \
  --base-url http://SERVER:8000/v1 \
  --tokenizer Qwen/Qwen3-4B-Instruct-2507 \
  --benchmark all \
  --prompt-variant both
```

Useful smoke test before calling the endpoint:

```bash
uv run python evaluation/run_eval_vllm.py \
  --model qwen3-4b-instruct-2507 \
  --tokenizer Qwen/Qwen3-4B-Instruct-2507 \
  --benchmark npee \
  --tasks choice \
  --limit 2 \
  --dry-run \
  --print-sample-prompts 1
```

### Progress and score reporting

Use the progress reporter while either vLLM path is running, or after it finishes:

```bash
uv run python scripts/report_eval_progress.py
```

Useful filters:

```bash
uv run python scripts/report_eval_progress.py --evaluator vllm_chat
uv run python scripts/report_eval_progress.py --model qwen3-4b-instruct-2507 --no-splits
uv run python scripts/report_eval_progress.py --json
```

The reporter scans `evaluation/results_vllm/` and `evaluation/results_vllm_chat/`,
counts JSONL rows with a boolean `correct` field as completed, reports
`correct/completed` accuracy, and for chat-completions runs also reports
`parse_failed` and `invalid` parser counts.  Before a `summary_*.json` exists, it
estimates expected totals from the full GeoBench scope encoded in the JSONL
filename, so runs launched with `--tasks`, `--limit`, or `--offset` may need that
scope caveat when interpreting progress.
