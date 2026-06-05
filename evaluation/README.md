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

## vLLM / OpenAI-compatible endpoint

If the model is already served by vLLM, use `run_eval_vllm.py` instead of
`run_eval.py`. The script keeps the GeoBench objective-task evaluation style by
asking the completions endpoint for one next-token answer among the valid labels
(`A/B/C/...` or `True/False`).

Example for a Qwen3 model served by vLLM:

```bash
python evaluation/run_eval_vllm.py \
  --model qwen3-4b-instruct-2507 \
  --base-url http://SERVER:8000/v1 \
  --tokenizer Qwen/Qwen3-4B-Instruct-2507 \
  --benchmark all \
  --prompt-variant both
```

Useful smoke test before calling the endpoint:

```bash
python evaluation/run_eval_vllm.py \
  --model qwen3-4b-instruct-2507 \
  --tokenizer Qwen/Qwen3-4B-Instruct-2507 \
  --benchmark npee \
  --tasks choice \
  --limit 2 \
  --dry-run \
  --print-sample-prompts 1
```
