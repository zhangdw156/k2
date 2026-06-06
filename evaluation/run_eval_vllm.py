#!/usr/bin/env python3
"""Evaluate GeoBench against a vLLM/OpenAI-compatible completions server.

This script is intentionally separate from ``run_eval.py``. The original K2
GeoBench evaluator loads a local HuggingFace model and scores the logits of the
next answer token. For a model already served by vLLM, this script uses the
OpenAI-compatible ``/v1/completions`` endpoint with ``logprobs`` and, by default,
``allowed_token_ids`` so that each item is still judged by the model's next-token
preference among the valid labels (A/B/C/... or True/False).

Prerequisites:
    1. Start vLLM with a stable served model name, for example:

       vllm serve Qwen/Qwen3-4B-Instruct-2507 \
         --host 0.0.0.0 \
         --port 8000 \
         --served-model-name qwen3-4b-instruct-2507 \
         --trust-remote-code

    2. Run this script from the repository root or any working directory. It
       resolves GeoBench data relative to this file.

    3. Install ``transformers`` in the evaluation environment because the script
       uses the target tokenizer to map labels such as ``A`` and ``True`` to
       token ids.

Quick endpoint smoke test:
    curl http://SERVER:8000/v1/models

Dry-run without calling the endpoint:
    uv run python evaluation/run_eval_vllm.py \
      --model qwen3-4b-instruct-2507 \
      --tokenizer Qwen/Qwen3-4B-Instruct-2507 \
      --benchmark npee \
      --tasks choice \
      --limit 2 \
      --dry-run \
      --print-sample-prompts 1

Evaluate all objective GeoBench subsets:
    uv run python evaluation/run_eval_vllm.py \
      --model qwen3-4b-instruct-2507 \
      --base-url http://SERVER:8000/v1 \
      --tokenizer Qwen/Qwen3-4B-Instruct-2507 \
      --benchmark all \
      --prompt-variant both

Evaluate NPEE only:
    uv run python evaluation/run_eval_vllm.py \
      --model qwen3-4b-instruct-2507 \
      --base-url http://SERVER:8000/v1 \
      --tokenizer Qwen/Qwen3-4B-Instruct-2507 \
      --benchmark npee \
      --tasks choice,tf \
      --prompt-variant both

Evaluate APStudy only:
    uv run python evaluation/run_eval_vllm.py \
      --model qwen3-4b-instruct-2507 \
      --base-url http://SERVER:8000/v1 \
      --tokenizer Qwen/Qwen3-4B-Instruct-2507 \
      --benchmark apstudy \
      --tasks choice \
      --prompt-variant both

Outputs:
    By default, records are written under::

        evaluation/results_vllm/<served-model-name>/

    The per-item JSONL contains the benchmark/task/index, gold answer,
    prediction, correctness flag, generated text, and candidate label logprobs.
    The summary JSON contains per-split accuracy.

Useful options:
    ``--prompt-variant woa|wa|both``
        ``woa`` asks for the answer directly. ``wa`` appends ``The answer is:``
        before the one-token answer. ``both`` mirrors the two-prompt style used
        by the original K2 evaluation code.

    ``--label-prefix``
        Use this if the tokenizer does not encode labels as a single token.
        For example, try ``--label-prefix ' '`` if ``A`` is split but `` A`` is
        one token.

    ``--use-chat-template``
        Wrap the Alpaca-style prompt in the tokenizer's chat template before
        sending it to the completions endpoint. This can help chat-tuned models,
        but changes the exact prompt format relative to the original K2 script.

    ``--disable-allowed-token-ids``
        Do not constrain decoding to answer labels. This is mainly for debugging
        non-vLLM OpenAI-compatible servers; accuracy may be unreliable if the
        returned ``top_logprobs`` do not include every candidate label.

Notes:
    * This script covers objective GeoBench tasks: NPEE multiple choice,
      NPEE true/false, and APStudy multiple choice.
    * Subjective NPEE questions are not scored here because the original
      repository does not provide an automatic judge for them.
    * If a label is not a single tokenizer token, the script fails fast so that
      the scoring protocol does not silently diverge from next-token evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

CHOICE_INSTRUCTION = (
    "Identify the choice that best completes the statement or answers the question."
)
TF_INSTRUCTION = "Determine whether the following statement is True or False."


@dataclass(frozen=True)
class Example:
    benchmark: str
    task: str
    index: int
    question: str
    answer: str
    labels: Tuple[str, ...]
    item_id: Optional[str] = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_few_shot(eval_dir: Path, task: str, variant: str) -> str:
    """Return the few-shot prefix used before the current question.

    The K2 repo only ships few-shot examples for multiple-choice questions.  For
    true/false, keep zero-shot prompts rather than inventing demonstrations.
    """

    if task != "choice":
        return ""
    filename = (
        "multiple_choice_samples_wa.txt"
        if variant == "wa"
        else "multiple_choice_samples.txt"
    )
    return read_text(eval_dir / filename)


def extract_choice_labels(question: str) -> Tuple[str, ...]:
    labels = re.findall(r"(?m)^\s*([A-Z])\.", question)
    # Preserve order while removing duplicates.
    return tuple(dict.fromkeys(labels))


def format_apstudy_question(item: Dict[str, Any]) -> str:
    q = item["question"]
    lines = [q["stem"], "Choose from:", ""]
    for choice in q["choices"]:
        lines.append(f"{choice['label']}. {choice['text']}")
    return "\n".join(lines)


def load_examples(data_dir: Path, benchmark: str) -> List[Example]:
    examples: List[Example] = []

    if benchmark in ("npee", "all"):
        path = data_dir / "geobench_npee.json"
        data = json.loads(path.read_text(encoding="utf-8"))

        for idx, (question, answer) in enumerate(
            zip(data["choice"]["question"], data["choice"]["answer"])
        ):
            labels = extract_choice_labels(question)
            if not labels:
                raise ValueError(f"No choice labels found for npee choice item {idx}")
            examples.append(
                Example(
                    benchmark="npee",
                    task="choice",
                    index=idx,
                    question=question,
                    answer=answer,
                    labels=labels,
                )
            )

        for idx, (question, answer) in enumerate(
            zip(data["tf"]["question"], data["tf"]["answer"])
        ):
            examples.append(
                Example(
                    benchmark="npee",
                    task="tf",
                    index=idx,
                    question=question,
                    answer=answer,
                    labels=("True", "False"),
                )
            )

    if benchmark in ("apstudy", "all"):
        path = data_dir / "geobench_apstudy.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for idx, item in enumerate(data):
            labels = tuple(choice["label"] for choice in item["question"]["choices"])
            examples.append(
                Example(
                    benchmark="apstudy",
                    task="choice",
                    index=idx,
                    question=format_apstudy_question(item),
                    answer=item["answerKey"],
                    labels=labels,
                    item_id=item.get("id"),
                )
            )

    return examples


def build_prompt(example: Example, eval_dir: Path, variant: str, use_chat_template: bool, tokenizer: Any) -> str:
    prefix = load_few_shot(eval_dir, example.task, variant)
    instruction = CHOICE_INSTRUCTION if example.task == "choice" else TF_INSTRUCTION
    response_prefix = "The answer is:\n" if variant == "wa" else ""

    current = (
        f"### Instruction:\n{instruction}\n{example.question}\n\n"
        f"### Response:\n{response_prefix}"
    )
    prompt = f"{prefix}\n\n{current}" if prefix else current

    if use_chat_template:
        if tokenizer is None:
            raise ValueError("--use-chat-template requires a tokenizer")
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def import_tokenizer(tokenizer_name_or_path: str) -> Any:
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "transformers is required for token-id constrained vLLM evaluation. "
            "Install it or run in the K2 environment."
        ) from exc
    return AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)


def label_token_ids(tokenizer: Any, labels: Sequence[str], label_prefix: str) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for label in labels:
        text = f"{label_prefix}{label}"
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"Label {label!r} with prefix {label_prefix!r} is not one token: {ids}. "
                "Use --label-prefix, or implement multi-token scoring for this tokenizer."
            )
        mapping[label] = ids[0]
    return mapping


def normalize_generated_label(text: str, labels: Sequence[str], label_prefix: str) -> Optional[str]:
    cleaned = text.strip()
    if label_prefix and cleaned.startswith(label_prefix.strip()):
        cleaned = cleaned[len(label_prefix.strip()) :].strip()
    for label in labels:
        if cleaned == label:
            return label
    # Be forgiving for outputs such as "A." or "A\n".
    match = re.match(r"^([A-Z])\b", cleaned)
    if match and match.group(1) in labels:
        return match.group(1)
    for label in ("True", "False"):
        if label in labels and cleaned.lower().startswith(label.lower()):
            return label
    return None


def completion_request(
    base_url: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout: float,
    retries: int,
    retry_sleep: float,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/completions"
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error: Optional[BaseException] = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
        except Exception as exc:  # noqa: BLE001 - CLI should report request failures cleanly.
            last_error = exc
        if attempt < retries:
            time.sleep(retry_sleep)
    raise RuntimeError(f"Completion request failed after {retries + 1} attempts: {last_error}")


def prediction_from_response(
    response: Dict[str, Any],
    labels: Sequence[str],
    label_prefix: str,
    token_id_to_label: Dict[int, str],
    tokenizer: Any,
) -> Tuple[Optional[str], Dict[str, float], str]:
    choice = response["choices"][0]
    text = choice.get("text", "")

    scores: Dict[str, float] = {}
    logprobs = choice.get("logprobs") or {}
    top_logprobs = logprobs.get("top_logprobs") or []
    if top_logprobs:
        first_step = top_logprobs[0] or {}
        decoded_to_label: Dict[str, str] = {}
        for token_id, label in token_id_to_label.items():
            decoded = tokenizer.decode([token_id])
            decoded_to_label[decoded] = label
            decoded_to_label[decoded.strip()] = label
        for token_text, logprob in first_step.items():
            label = decoded_to_label.get(token_text)
            if label is None:
                label = decoded_to_label.get(token_text.strip())
            if label is not None:
                scores[label] = max(scores.get(label, float("-inf")), float(logprob))

    if scores:
        return max(scores, key=scores.get), scores, text

    # If the server does not return all top_logprobs, the generated token is
    # still the argmax when temperature=0 and allowed_token_ids is active.
    return normalize_generated_label(text, labels, label_prefix), scores, text


def iter_selected_examples(
    examples: Sequence[Example],
    tasks: Sequence[str],
    offset: int,
    limit: Optional[int],
) -> List[Example]:
    selected = [ex for ex in examples if ex.task in tasks]
    if offset:
        selected = selected[offset:]
    if limit is not None:
        selected = selected[:limit]
    return selected


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GeoBench using a vLLM/OpenAI-compatible completions endpoint."
    )
    parser.add_argument(
        "--model", required=True, help="Served model name exposed by vLLM"
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        help="OpenAI-compatible base URL, e.g. http://host:8000/v1",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="API key for the endpoint; vLLM usually accepts any non-empty value",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="HF tokenizer name/path. Defaults to --model, but served names often need an explicit HF id.",
    )
    parser.add_argument(
        "--benchmark",
        choices=("npee", "apstudy", "all"),
        default="all",
        help="GeoBench subset to evaluate",
    )
    parser.add_argument(
        "--tasks",
        default="choice,tf",
        help="Comma-separated tasks to include: choice,tf. APStudy only has choice.",
    )
    parser.add_argument(
        "--prompt-variant",
        choices=("woa", "wa", "both"),
        default="both",
        help="woa: direct response; wa: response starts with 'The answer is:'; both: run both variants",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of selected examples"
    )
    parser.add_argument(
        "--offset", type=int, default=0, help="Skip selected examples before evaluating"
    )
    parser.add_argument(
        "--label-prefix",
        default="",
        help="Prefix used when encoding labels as one token, e.g. a single space if needed.",
    )
    parser.add_argument(
        "--disable-allowed-token-ids",
        action="store_true",
        help="Do not send vLLM allowed_token_ids. Accuracy may be unreliable if labels are not in top_logprobs.",
    )
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Wrap the prompt with tokenizer.apply_chat_template before sending to /completions.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument(
        "--request-sleep", type=float, default=0.0, help="Sleep between API requests"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for jsonl results and summary. Defaults to evaluation/results_vllm/<model>.",
    )
    parser.add_argument(
        "--save-prompts", action="store_true", help="Include full prompts in JSONL"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts and validate labels, but do not call the endpoint",
    )
    parser.add_argument(
        "--print-sample-prompts",
        type=int,
        default=0,
        help="Print the first N prompts during dry-run/debugging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    eval_dir = root / "evaluation"
    data_dir = root / "data" / "geobench"

    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    invalid_tasks = set(tasks) - {"choice", "tf"}
    if invalid_tasks:
        raise SystemExit(f"Unsupported task(s): {sorted(invalid_tasks)}")

    variants = ("woa", "wa") if args.prompt_variant == "both" else (args.prompt_variant,)
    examples = iter_selected_examples(
        load_examples(data_dir, args.benchmark), tasks, args.offset, args.limit
    )
    if not examples:
        raise SystemExit("No examples selected")

    tokenizer_name = args.tokenizer or args.model
    tokenizer = import_tokenizer(tokenizer_name)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else eval_dir / "results_vllm" / safe_name(args.model)
    )
    ensure_output_dir(output_dir)

    summary: Dict[str, Dict[str, int]] = {}
    jsonl_path = output_dir / f"geobench_{args.benchmark}_{'-'.join(variants)}.jsonl"

    dry_count = 0
    with jsonl_path.open("w", encoding="utf-8") as out:
        for variant in variants:
            for ex in examples:
                prompt = build_prompt(ex, eval_dir, variant, args.use_chat_template, tokenizer)
                token_map = label_token_ids(tokenizer, ex.labels, args.label_prefix)
                token_id_to_label = {token_id: label for label, token_id in token_map.items()}

                if args.print_sample_prompts and dry_count < args.print_sample_prompts:
                    print("=" * 80)
                    print(
                        f"{ex.benchmark}/{ex.task}/{ex.index} "
                        f"variant={variant} labels={ex.labels}"
                    )
                    print(prompt)
                    dry_count += 1

                record: Dict[str, Any] = {
                    "benchmark": ex.benchmark,
                    "task": ex.task,
                    "index": ex.index,
                    "item_id": ex.item_id,
                    "prompt_variant": variant,
                    "answer": ex.answer,
                    "labels": list(ex.labels),
                    "label_token_ids": token_map,
                }
                if args.save_prompts:
                    record["prompt"] = prompt

                if args.dry_run:
                    record.update({"prediction": None, "correct": None, "dry_run": True})
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue

                payload: Dict[str, Any] = {
                    "model": args.model,
                    "prompt": prompt,
                    "max_tokens": 1,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "logprobs": max(len(ex.labels), 5),
                }
                if not args.disable_allowed_token_ids:
                    payload["allowed_token_ids"] = list(token_id_to_label.keys())

                response = completion_request(
                    args.base_url,
                    args.api_key,
                    payload,
                    timeout=args.timeout,
                    retries=args.retries,
                    retry_sleep=args.retry_sleep,
                )
                prediction, scores, generated_text = prediction_from_response(
                    response, ex.labels, args.label_prefix, token_id_to_label, tokenizer
                )
                correct = prediction == ex.answer

                key = f"{ex.benchmark}/{ex.task}/{variant}"
                bucket = summary.setdefault(key, {"correct": 0, "total": 0})
                bucket["total"] += 1
                if correct:
                    bucket["correct"] += 1

                record.update(
                    {
                        "prediction": prediction,
                        "correct": correct,
                        "generated_text": generated_text,
                        "label_logprobs": scores,
                    }
                )
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()

                print(
                    f"[{key} #{ex.index}] pred={prediction!r} gold={ex.answer!r} "
                    f"correct={correct}"
                )
                if args.request_sleep:
                    time.sleep(args.request_sleep)

    summary_with_accuracy: Dict[str, Dict[str, float]] = {}
    if args.dry_run:
        print(f"Dry-run wrote {jsonl_path}")
        print(f"Selected examples: {len(examples)}; variants: {', '.join(variants)}")
    else:
        for key, counts in summary.items():
            total = counts["total"]
            correct = counts["correct"]
            summary_with_accuracy[key] = {
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total else 0.0,
            }
        summary_path = output_dir / f"summary_{args.benchmark}_{'-'.join(variants)}.json"
        write_json(summary_path, summary_with_accuracy)
        print(f"Wrote results: {jsonl_path}")
        print(f"Wrote summary: {summary_path}")
        print(json.dumps(summary_with_accuracy, ensure_ascii=False, indent=2))

    return 0


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "model"


if __name__ == "__main__":
    raise SystemExit(main())
