#!/usr/bin/env python3
"""Evaluate GeoBench with an OpenAI-compatible chat completions server.

This evaluator is a parallel path to ``run_eval_vllm.py``.  It intentionally
leaves the existing logprob/``allowed_token_ids`` evaluator unchanged and adds a
modern generation-style path for chat or reasoning models served by vLLM.

The evaluation flow is:

1. Build the same GeoBench objective examples used by the existing evaluator.
2. Build Alpaca-style few-shot prompts with the original K2 ``woa``/``wa``
   variants:

   * ``woa``: the current prompt ends with ``### Response:``.
   * ``wa``: the current prompt ends with ``### Response:\nThe answer is:``.

3. Send the prompt as a chat ``user`` message to ``client.chat.completions``.
4. Parse the final answer from the generated message content.
5. If the content contains ``</think>``, only parse the text after the last
   ``</think>`` tag.  This keeps reasoning and non-reasoning models on the same
   scoring path while avoiding accidental matches inside chain-of-thought text.
6. Compare the parsed label with the GeoBench ground truth and write JSONL plus
   summary accuracy outputs.

Example for a Qwen3 model served by vLLM:

    uv run python evaluation/run_eval_vllm_chat.py \
      --model qwen3-4b-instruct-2507 \
      --base-url http://localhost:8000/v1 \
      --benchmark all \
      --prompt-variant both

Dry-run without calling the endpoint:

    uv run python evaluation/run_eval_vllm_chat.py \
      --model qwen3-4b-instruct-2507 \
      --benchmark npee \
      --tasks choice \
      --limit 2 \
      --dry-run \
      --print-sample-prompts 1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from openai import OpenAI

CHOICE_INSTRUCTION = (
    "Identify the choice that best completes the statement or answers the question."
)
TF_INSTRUCTION = "Determine whether the following statement is True or False."
DEFAULT_THINK_END_TAG = "</think>"


@dataclass(frozen=True)
class Example:
    benchmark: str
    task: str
    index: int
    question: str
    answer: str
    labels: Tuple[str, ...]
    item_id: Optional[str] = None


@dataclass(frozen=True)
class ParseResult:
    prediction: Optional[str]
    parser_status: str
    answer_region: str
    parsed_text: Optional[str]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_few_shot(eval_dir: Path, task: str, variant: str) -> str:
    """Return the few-shot prefix used before the current question.

    The K2 repo only ships few-shot examples for multiple-choice questions. For
    true/false questions, keep zero-shot prompts rather than inventing examples.
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


def build_prompt(example: Example, eval_dir: Path, variant: str) -> str:
    """Build the original K2-style prompt for the selected ``woa``/``wa`` variant."""

    prefix = load_few_shot(eval_dir, example.task, variant)
    instruction = CHOICE_INSTRUCTION if example.task == "choice" else TF_INSTRUCTION
    response_prefix = "The answer is:\n" if variant == "wa" else ""
    current = (
        f"### Instruction:\n{instruction}\n{example.question}\n\n"
        f"### Response:\n{response_prefix}"
    )
    return f"{prefix}\n\n{current}" if prefix else current


def build_messages(prompt: str, system_prompt: Optional[str]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


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


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "model"


def normalize_content(content: Any) -> str:
    """Normalize OpenAI/vLLM message content into a plain string."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None) or getattr(item, "content", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(content)


def extract_extra_message_field(message: Any, names: Iterable[str]) -> Optional[str]:
    for name in names:
        value = getattr(message, name, None)
        if value:
            return normalize_content(value)
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict):
        for name in names:
            value = extra.get(name)
            if value:
                return normalize_content(value)
    if isinstance(message, dict):
        for name in names:
            value = message.get(name)
            if value:
                return normalize_content(value)
    return None


def usage_to_dict(usage: Any) -> Optional[Dict[str, Any]]:
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return {
        key: getattr(usage, key)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if hasattr(usage, key)
    }


def chat_complete(
    client: OpenAI,
    args: argparse.Namespace,
    messages: List[Dict[str, str]],
    extra_body: Optional[Dict[str, Any]],
) -> Any:
    kwargs: Dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
    }
    if extra_body:
        kwargs["extra_body"] = extra_body
    return client.chat.completions.create(**kwargs)


def chat_complete_with_retries(
    client: OpenAI,
    args: argparse.Namespace,
    messages: List[Dict[str, str]],
    extra_body: Optional[Dict[str, Any]],
) -> Any:
    last_error: Optional[BaseException] = None
    for attempt in range(args.retries + 1):
        try:
            return chat_complete(client, args, messages, extra_body)
        except Exception as exc:  # noqa: BLE001 - CLI should report request failures cleanly.
            last_error = exc
            if attempt < args.retries:
                time.sleep(args.retry_sleep)
    raise RuntimeError(f"Chat completion failed after {args.retries + 1} attempts: {last_error}")


def answer_region(raw_output: str, think_end_tag: str = DEFAULT_THINK_END_TAG) -> str:
    if think_end_tag and think_end_tag in raw_output:
        return raw_output.rsplit(think_end_tag, 1)[1].strip()
    return raw_output.strip()


def label_alternation(labels: Sequence[str]) -> str:
    return "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))


def normalize_label(candidate: str, labels: Sequence[str]) -> Optional[str]:
    cleaned = candidate.strip().strip("()[]{}<>.,;:：。")
    for label in labels:
        if cleaned == label:
            return label
    for label in labels:
        if cleaned.lower() == label.lower():
            return label
    return None


def find_invalid_explicit_label(text: str, labels: Sequence[str]) -> Optional[str]:
    """Find explicit answer prefixes followed by an out-of-set label."""

    if all(re.fullmatch(r"[A-Z]", label) for label in labels):
        candidate_pattern = r"[A-Z]"
    elif set(labels) == {"True", "False"}:
        candidate_pattern = r"[A-Za-z]+"
    else:
        candidate_pattern = r"[A-Za-z]+"
    prefix_pattern = (
        r"(?:final\s+answer|the\s+answer\s+is|answer)"
        r"\s*(?:is\s*)?(?:[:：\-]\s*)?[\(\[]?"
        rf"(?P<label>{candidate_pattern})"
    )
    for match in re.finditer(prefix_pattern, text, flags=re.IGNORECASE):
        token = match.group("label")
        if normalize_label(token, labels) is None:
            return match.group(0).strip()
    return None


def parse_answer(raw_output: str, labels: Sequence[str], think_end_tag: str) -> ParseResult:
    region = answer_region(raw_output, think_end_tag)
    if not region:
        return ParseResult(None, "no_answer_found", region, None)

    alts = label_alternation(labels)
    boundary = r"(?![A-Za-z0-9_-])"
    explicit_prefixes = [
        r"final\s+answer",
        r"the\s+answer\s+is",
        r"answer",
    ]

    for prefix in explicit_prefixes:
        pattern = (
            rf"(?:^|\b){prefix}\b\s*"
            rf"(?:is\s*)?(?:[:：\-]\s*)?[\(\[]?\s*(?P<label>{alts}){boundary}"
        )
        matches = list(re.finditer(pattern, region, flags=re.IGNORECASE | re.MULTILINE))
        if matches:
            match = matches[-1]
            prediction = normalize_label(match.group("label"), labels)
            if prediction is not None:
                return ParseResult(prediction, "ok", region, match.group(0).strip())

    invalid = find_invalid_explicit_label(region, labels)
    if invalid is not None:
        return ParseResult(None, "invalid_label", region, invalid)

    standalone_pattern = (
        rf"^\s*(?:[-*]\s*)?[\(\[]?\s*(?P<label>{alts})\s*"
        rf"(?:[\)\]\.。:：])?\s*$"
    )
    standalone_matches = [
        match
        for line in region.splitlines()
        for match in [re.match(standalone_pattern, line, flags=re.IGNORECASE)]
        if match
    ]
    if standalone_matches:
        match = standalone_matches[-1]
        prediction = normalize_label(match.group("label"), labels)
        if prediction is not None:
            return ParseResult(prediction, "ok", region, match.group(0).strip())

    leading_pattern = rf"^\s*[\(\[]?\s*(?P<label>{alts})(?:[\)\]\.。:：])?(?:\s|$)"
    match = re.match(leading_pattern, region, flags=re.IGNORECASE)
    if match:
        prediction = normalize_label(match.group("label"), labels)
        if prediction is not None:
            return ParseResult(prediction, "ok", region, match.group(0).strip())

    return ParseResult(None, "no_answer_found", region, None)


def response_to_fields(response: Any) -> Dict[str, Any]:
    choice = response.choices[0]
    message = choice.message
    content = normalize_content(getattr(message, "content", ""))
    reasoning = extract_extra_message_field(message, ("reasoning", "reasoning_content"))
    return {
        "raw_output": content,
        "raw_reasoning": reasoning,
        "finish_reason": getattr(choice, "finish_reason", None),
        "usage": usage_to_dict(getattr(response, "usage", None)),
    }


def parse_extra_body(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--extra-body-json is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("--extra-body-json must decode to a JSON object")
    return parsed


def run_parser_self_test() -> int:
    cases = [
        ("A", ("A", "B", "C", "D"), "A", "ok"),
        ("The answer is: B", ("A", "B", "C", "D"), "B", "ok"),
        ("Final answer: C", ("A", "B", "C", "D"), "C", "ok"),
        ("<think>Maybe A, then B.</think>\nFinal answer: D", ("A", "B", "C", "D"), "D", "ok"),
        ("<think>Final answer: A</think>\nNo final response", ("A", "B", "C", "D"), None, "no_answer_found"),
        ("Answer: E", ("A", "B", "C", "D"), None, "invalid_label"),
        ("true", ("True", "False"), "True", "ok"),
        ("</think> The answer is False", ("True", "False"), "False", "ok"),
    ]
    failures: List[str] = []
    for raw, labels, expected_prediction, expected_status in cases:
        result = parse_answer(raw, labels, DEFAULT_THINK_END_TAG)
        if result.prediction != expected_prediction or result.parser_status != expected_status:
            failures.append(
                f"raw={raw!r}: got ({result.prediction!r}, {result.parser_status}), "
                f"expected ({expected_prediction!r}, {expected_status})"
            )
    if failures:
        print("Parser self-test failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"Parser self-test passed ({len(cases)} cases)")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GeoBench using OpenAI-compatible chat completions."
    )
    parser.add_argument("--model", required=False, help="Served model name exposed by vLLM")
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
        help="woa: no answer prefix; wa: current prompt ends with 'The answer is:'; both: run both variants",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of selected examples")
    parser.add_argument("--offset", type=int, default=0, help="Skip selected examples before evaluating")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum generated tokens per item")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--request-sleep", type=float, default=0.0, help="Sleep between API requests")
    parser.add_argument(
        "--think-end-tag",
        default=DEFAULT_THINK_END_TAG,
        help="If present in model output, parse only the text after its last occurrence.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system message. Defaults to no system message to preserve the K2 prompt style.",
    )
    parser.add_argument(
        "--extra-body-json",
        default=None,
        help="Optional JSON object passed as OpenAI SDK extra_body for vLLM-specific options.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for jsonl results and summary. Defaults to evaluation/results_vllm_chat/<model>.",
    )
    parser.add_argument("--save-prompts", action="store_true", help="Include full prompts/messages in JSONL")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts and write JSONL, but do not call the endpoint",
    )
    parser.add_argument(
        "--print-sample-prompts",
        type=int,
        default=0,
        help="Print the first N prompts during dry-run/debugging",
    )
    parser.add_argument(
        "--self-test-parser",
        action="store_true",
        help="Run built-in parser tests and exit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test_parser:
        return run_parser_self_test()
    if not args.model:
        raise SystemExit("--model is required unless --self-test-parser is used")

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

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else eval_dir / "results_vllm_chat" / safe_name(args.model)
    )
    ensure_output_dir(output_dir)
    jsonl_path = output_dir / f"geobench_{args.benchmark}_{'-'.join(variants)}.jsonl"

    client = None if args.dry_run else OpenAI(api_key=args.api_key, base_url=args.base_url)
    extra_body = parse_extra_body(args.extra_body_json)
    summary: Dict[str, Dict[str, int]] = {}

    dry_count = 0
    with jsonl_path.open("w", encoding="utf-8") as out:
        for variant in variants:
            for ex in examples:
                prompt = build_prompt(ex, eval_dir, variant)
                messages = build_messages(prompt, args.system_prompt)

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
                }
                if args.save_prompts:
                    record["messages"] = messages

                key = f"{ex.benchmark}/{ex.task}/{variant}"
                bucket = summary.setdefault(
                    key, {"correct": 0, "total": 0, "parse_failed": 0, "invalid": 0}
                )
                bucket["total"] += 1

                if args.dry_run:
                    record.update(
                        {
                            "prediction": None,
                            "correct": None,
                            "parser_status": None,
                            "answer_region": None,
                            "parsed_text": None,
                            "raw_output": None,
                            "dry_run": True,
                        }
                    )
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue

                assert client is not None
                response = chat_complete_with_retries(client, args, messages, extra_body)
                fields = response_to_fields(response)
                parse = parse_answer(fields["raw_output"], ex.labels, args.think_end_tag)
                correct = parse.prediction == ex.answer
                if correct:
                    bucket["correct"] += 1
                if parse.parser_status == "no_answer_found":
                    bucket["parse_failed"] += 1
                if parse.parser_status == "invalid_label":
                    bucket["invalid"] += 1

                record.update(
                    {
                        "prediction": parse.prediction,
                        "correct": correct,
                        "parser_status": parse.parser_status,
                        "answer_region": parse.answer_region,
                        "parsed_text": parse.parsed_text,
                        **fields,
                    }
                )
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()

                print(
                    f"[{key} #{ex.index}] pred={parse.prediction!r} "
                    f"gold={ex.answer!r} status={parse.parser_status} correct={correct}"
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
                "parse_failed": counts["parse_failed"],
                "invalid": counts["invalid"],
            }
        summary_path = output_dir / f"summary_{args.benchmark}_{'-'.join(variants)}.json"
        write_json(summary_path, summary_with_accuracy)
        print(f"Wrote results: {jsonl_path}")
        print(f"Wrote summary: {summary_path}")
        print(json.dumps(summary_with_accuracy, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
