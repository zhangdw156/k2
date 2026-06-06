#!/usr/bin/env python3
"""Report progress and scores for K2 GeoBench vLLM evaluations.

The two K2 server-side evaluators write per-item JSONL files under:

    evaluation/results_vllm/<model>/geobench_*.jsonl
    evaluation/results_vllm_chat/<model>/geobench_*.jsonl

This reporter scans those files while an experiment is running or after it has
finished.  A row is counted as completed only when its ``correct`` field is a
boolean and the row is not marked as ``dry_run``.  That matches both K2 vLLM
paths: failed chat parsing is still a completed evaluated sample because the
chat evaluator records ``correct: false`` plus ``parser_status``.

Examples:
    uv run python scripts/report_eval_progress.py
    uv run python scripts/report_eval_progress.py --evaluator vllm_chat
    uv run python scripts/report_eval_progress.py --model qwen3-4b-instruct-2507 --json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVALUATION_DIR = REPO_ROOT / "evaluation"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "geobench"
EVALUATOR_DIRS = {
    "vllm": "results_vllm",
    "vllm_chat": "results_vllm_chat",
}
JSONL_RE = re.compile(r"^geobench_(?P<benchmark>npee|apstudy|all)_(?P<variants>[A-Za-z0-9_-]+)\.jsonl$")
DEFAULT_TASKS_BY_BENCHMARK = {
    "npee": (("npee", "choice"), ("npee", "tf")),
    "apstudy": (("apstudy", "choice"),),
    "all": (("npee", "choice"), ("npee", "tf"), ("apstudy", "choice")),
}


@dataclass(frozen=True)
class SplitProgress:
    split: str
    completed: int
    correct: int
    expected: int | None
    progress: float | None
    accuracy_done: float | None
    accuracy_lower_bound: float | None
    parse_failed: int
    invalid: int
    dry_run: int
    records: int


@dataclass(frozen=True)
class RunProgress:
    evaluator: str
    model: str
    result_file: str
    benchmark: str
    variants: list[str]
    status: str
    completed: int
    correct: int
    expected: int | None
    progress: float | None
    accuracy_done: float | None
    accuracy_lower_bound: float | None
    parse_failed: int
    invalid: int
    dry_run: int
    records: int
    duplicate_records: int
    malformed_lines: int
    expected_source: str
    splits: list[SplitProgress]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report K2 GeoBench progress/scores for vLLM and vLLM-chat JSONL outputs."
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=DEFAULT_EVALUATION_DIR,
        help="Evaluation directory containing results_vllm*/. Default: evaluation/.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="GeoBench data directory used to estimate expected totals. Default: data/geobench/.",
    )
    parser.add_argument(
        "--evaluator",
        action="append",
        choices=("vllm", "vllm_chat", "all"),
        help="Evaluator to include. Can be repeated. Default: all.",
    )
    parser.add_argument(
        "--model",
        action="append",
        help="Model directory/name to include. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of terminal tables.",
    )
    parser.add_argument(
        "--no-splits",
        action="store_true",
        help="Only print the per-run summary table.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any result file has malformed JSON lines.",
    )
    return parser.parse_args()


def split_values(values: Sequence[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for value in values:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return out


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_geobench_counts(data_dir: Path) -> dict[tuple[str, str], int]:
    """Return full-scope item counts used by the K2 vLLM evaluators."""

    counts: dict[tuple[str, str], int] = {}

    npee_path = data_dir / "geobench_npee.json"
    if npee_path.exists():
        npee = load_json(npee_path)
        choice = npee.get("choice", {})
        tf = npee.get("tf", {})
        counts[("npee", "choice")] = min(
            len(choice.get("question", [])), len(choice.get("answer", []))
        )
        counts[("npee", "tf")] = min(len(tf.get("question", [])), len(tf.get("answer", [])))

    apstudy_path = data_dir / "geobench_apstudy.json"
    if apstudy_path.exists():
        apstudy = load_json(apstudy_path)
        counts[("apstudy", "choice")] = len(apstudy) if isinstance(apstudy, list) else 0

    return counts


def parse_result_filename(path: Path) -> tuple[str, list[str]] | None:
    match = JSONL_RE.match(path.name)
    if not match:
        return None
    benchmark = match.group("benchmark")
    variants = [part for part in match.group("variants").split("-") if part]
    return benchmark, variants


def summary_path_for(jsonl_path: Path) -> Path:
    return jsonl_path.with_name(jsonl_path.name.replace("geobench_", "summary_", 1).replace(".jsonl", ".json"))


def load_summary_totals(summary_path: Path) -> dict[str, int]:
    if not summary_path.exists():
        return {}
    try:
        data = load_json(summary_path)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    totals: dict[str, int] = {}
    for split, stats in data.items():
        if isinstance(stats, dict) and isinstance(stats.get("total"), int):
            totals[str(split)] = int(stats["total"])
    return totals


def default_expected_totals(
    benchmark: str,
    variants: Sequence[str],
    counts: dict[tuple[str, str], int],
) -> dict[str, int]:
    totals: dict[str, int] = {}
    for variant in variants:
        for bench, task in DEFAULT_TASKS_BY_BENCHMARK[benchmark]:
            totals[f"{bench}/{task}/{variant}"] = counts.get((bench, task), 0)
    return totals


def is_completed_record(record: dict[str, Any]) -> bool:
    return record.get("dry_run") is not True and isinstance(record.get("correct"), bool)


def read_latest_records(path: Path) -> tuple[list[dict[str, Any]], int, int]:
    """Read JSONL rows, keeping the last row for duplicate item keys."""

    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    malformed = 0
    duplicates = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(record, dict):
                malformed += 1
                continue
            key = (
                record.get("benchmark"),
                record.get("task"),
                record.get("prompt_variant"),
                record.get("index"),
            )
            if None in key:
                key = ("__line__", line_no)
            if key in latest:
                duplicates += 1
            latest[key] = record
    return list(latest.values()), duplicates, malformed


def summarize_split(split: str, records: list[dict[str, Any]], expected: int | None) -> SplitProgress:
    completed = 0
    correct = 0
    parse_failed = 0
    invalid = 0
    dry_run = 0

    for record in records:
        if record.get("dry_run") is True:
            dry_run += 1
        if is_completed_record(record):
            completed += 1
            if record.get("correct") is True:
                correct += 1
            if record.get("parser_status") == "no_answer_found":
                parse_failed += 1
            if record.get("parser_status") == "invalid_label":
                invalid += 1

    progress = completed / expected if expected else None
    accuracy_done = correct / completed if completed else None
    accuracy_lower_bound = correct / expected if expected else None
    return SplitProgress(
        split=split,
        completed=completed,
        correct=correct,
        expected=expected,
        progress=progress,
        accuracy_done=accuracy_done,
        accuracy_lower_bound=accuracy_lower_bound,
        parse_failed=parse_failed,
        invalid=invalid,
        dry_run=dry_run,
        records=len(records),
    )


def status_for(completed: int, expected: int | None, dry_run: int, malformed: int) -> str:
    if malformed:
        return "CHECK"
    if completed == 0 and dry_run > 0:
        return "DRY-RUN"
    if expected is None:
        return "UNKNOWN"
    if completed > expected:
        return "OVER"
    if expected > 0 and completed == expected:
        return "DONE"
    if completed > 0:
        return "PARTIAL"
    return "PENDING"


def summarize_file(
    evaluator: str,
    model: str,
    path: Path,
    counts: dict[tuple[str, str], int],
) -> RunProgress | None:
    parsed = parse_result_filename(path)
    if parsed is None:
        return None
    benchmark, variants = parsed

    records, duplicates, malformed = read_latest_records(path)
    by_split: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        bench = record.get("benchmark")
        task = record.get("task")
        variant = record.get("prompt_variant")
        if not bench or not task or not variant:
            continue
        by_split.setdefault(f"{bench}/{task}/{variant}", []).append(record)

    summary_totals = load_summary_totals(summary_path_for(path))
    default_totals = default_expected_totals(benchmark, variants, counts)
    expected_source = "summary" if summary_totals else "default-geobench"

    split_names = sorted(
        set(default_totals) | set(summary_totals) | set(by_split),
        key=lambda value: (
            {"npee": 0, "apstudy": 1}.get(value.split("/")[0], 9),
            {"choice": 0, "tf": 1}.get(value.split("/")[1] if "/" in value else "", 9),
            {"woa": 0, "wa": 1}.get(value.split("/")[-1], 9),
            value,
        ),
    )
    splits: list[SplitProgress] = []
    for split in split_names:
        expected = summary_totals.get(split, default_totals.get(split))
        splits.append(summarize_split(split, by_split.get(split, []), expected))

    completed = sum(split.completed for split in splits)
    correct = sum(split.correct for split in splits)
    expected = sum(split.expected for split in splits if split.expected is not None)
    expected_value: int | None = expected if all(split.expected is not None for split in splits) else None
    parse_failed = sum(split.parse_failed for split in splits)
    invalid = sum(split.invalid for split in splits)
    dry_run = sum(split.dry_run for split in splits)
    progress = completed / expected_value if expected_value else None
    accuracy_done = correct / completed if completed else None
    accuracy_lower_bound = correct / expected_value if expected_value else None

    return RunProgress(
        evaluator=evaluator,
        model=model,
        result_file=relpath(path),
        benchmark=benchmark,
        variants=variants,
        status=status_for(completed, expected_value, dry_run, malformed),
        completed=completed,
        correct=correct,
        expected=expected_value,
        progress=progress,
        accuracy_done=accuracy_done,
        accuracy_lower_bound=accuracy_lower_bound,
        parse_failed=parse_failed,
        invalid=invalid,
        dry_run=dry_run,
        records=len(records),
        duplicate_records=duplicates,
        malformed_lines=malformed,
        expected_source=expected_source,
        splits=splits,
    )


def discover_files(evaluation_dir: Path, evaluators: Iterable[str], models: set[str]) -> list[tuple[str, str, Path]]:
    found: list[tuple[str, str, Path]] = []
    for evaluator in evaluators:
        root = evaluation_dir / EVALUATOR_DIRS[evaluator]
        if not root.exists():
            continue
        for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            model = model_dir.name
            if models and model not in models:
                continue
            for path in sorted(model_dir.glob("geobench_*.jsonl")):
                if parse_result_filename(path) is not None:
                    found.append((evaluator, model, path))
    return found


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def ratio(done: int, total: int | None) -> str:
    return f"{done:,}/?" if total is None else f"{done:,}/{total:,}"


def int_or_na(value: int | None) -> str:
    return "n/a" if value is None else f"{value:,}"


def print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    if not rows:
        return
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * width for width in widths)))
    for row in rows:
        print(fmt.format(*row))


def print_reports(reports: list[RunProgress], *, no_splits: bool) -> None:
    if not reports:
        print("No K2 vLLM result JSONL files found.")
        return

    print("K2 GeoBench vLLM evaluation progress")
    print(
        "Completion rule: completed = JSONL rows with boolean correct and not dry_run; "
        "accuracy(done) = correct/completed; lower-bound = correct/expected."
    )
    print(
        "Note: when no summary_*.json exists yet, expected totals assume the full benchmark scope "
        "encoded by the filename; --tasks/--limit/--offset are not encoded in JSONL filenames."
    )
    print()

    run_rows: list[list[str]] = []
    for report in reports:
        run_rows.append(
            [
                report.evaluator,
                report.model,
                report.status,
                ratio(report.completed, report.expected),
                pct(report.progress),
                f"{report.correct:,}",
                pct(report.accuracy_done),
                pct(report.accuracy_lower_bound),
                f"{report.parse_failed:,}",
                f"{report.invalid:,}",
                f"{report.dry_run:,}",
                report.expected_source,
                report.result_file,
            ]
        )
    print_table(
        [
            "evaluator",
            "model",
            "status",
            "done/expected",
            "progress",
            "correct",
            "acc(done)",
            "acc(lower)",
            "parse_fail",
            "invalid",
            "dry_run",
            "expected",
            "file",
        ],
        run_rows,
    )

    if no_splits:
        return

    print("\nPer-split details")
    split_rows: list[list[str]] = []
    for report in reports:
        for split in report.splits:
            split_rows.append(
                [
                    report.evaluator,
                    report.model,
                    split.split,
                    ratio(split.completed, split.expected),
                    pct(split.progress),
                    f"{split.correct:,}",
                    pct(split.accuracy_done),
                    pct(split.accuracy_lower_bound),
                    f"{split.parse_failed:,}",
                    f"{split.invalid:,}",
                    f"{split.dry_run:,}",
                ]
            )
    print_table(
        [
            "evaluator",
            "model",
            "split",
            "done/expected",
            "progress",
            "correct",
            "acc(done)",
            "acc(lower)",
            "parse_fail",
            "invalid",
            "dry_run",
        ],
        split_rows,
    )


def main() -> int:
    args = parse_args()
    requested = split_values(args.evaluator)
    evaluators = list(EVALUATOR_DIRS) if not requested or "all" in requested else requested
    models = set(split_values(args.model))

    counts = load_geobench_counts(args.data_dir)
    files = discover_files(args.evaluation_dir, evaluators, models)
    reports = [report for evaluator, model, path in files if (report := summarize_file(evaluator, model, path, counts))]

    if args.json:
        print(json.dumps([asdict(report) for report in reports], ensure_ascii=False, indent=2))
    else:
        print_reports(reports, no_splits=args.no_splits)

    if args.strict and any(report.malformed_lines for report in reports):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
