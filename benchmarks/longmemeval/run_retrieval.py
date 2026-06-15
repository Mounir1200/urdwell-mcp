"""Evaluate ContextMemory retrieval on the cleaned LongMemEval benchmark.

This runner follows the official flat retrieval setup:
  - user turns or user-only sessions are indexed;
  - the benchmark question is used as the retrieval query;
  - abstention and instances without user-side targets are excluded;
  - recall-any, recall-all, and NDCG are reported at several cutoffs.

The run measures retrieval only. It does not claim end-to-end QA accuracy,
because ContextMemory does not yet include an extraction LLM or reader LLM.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import embeddings
from pipeline import SIMILARITY_THRESHOLD


DEFAULT_DATASET = (
    PROJECT_ROOT
    / "benchmarks"
    / "longmemeval"
    / "longmemeval_oracle.json"
)
DEFAULT_REPORT_DIR = PROJECT_ROOT / "benchmarks" / "longmemeval" / "reports"
DEFAULT_CUTOFFS = (1, 5, 10)
DEFAULT_THRESHOLD_SWEEP = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)


@dataclass
class RetrievalCase:
    question_id: str
    question_type: str
    query: str
    corpus: list[str]
    corpus_ids: list[str]
    correct_ids: set[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--backend",
        choices=["transformer", "hashing"],
        default="transformer",
    )
    parser.add_argument(
        "--granularity",
        choices=["turn", "session"],
        default="turn",
    )
    parser.add_argument(
        "--include-date",
        action="store_true",
        help="Prefix indexed text with the session date (ContextMemory variant).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run a reproducible stratified subset instead of all 500 cases.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--threshold",
        type=float,
        default=SIMILARITY_THRESHOLD,
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--save-details",
        action="store_true",
        help="Include per-question rankings and metrics in the report.",
    )
    return parser.parse_args()


def question_type(entry: dict[str, Any]) -> str:
    if entry["question_id"].endswith("_abs"):
        return "abstention"
    return entry["question_type"]


def select_entries(
    entries: list[dict[str, Any]],
    limit: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if limit is None or limit >= len(entries):
        return entries
    if limit <= 0:
        raise ValueError("limit must be positive")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        groups[question_type(entry)].append(entry)

    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)

    selected = []
    group_names = sorted(groups)
    while len(selected) < limit:
        made_progress = False
        for name in group_names:
            if groups[name] and len(selected) < limit:
                selected.append(groups[name].pop())
                made_progress = True
        if not made_progress:
            break
    rng.shuffle(selected)
    return selected


def build_case(
    entry: dict[str, Any],
    granularity: str,
    include_date: bool = False,
) -> RetrievalCase:
    corpus: list[str] = []
    corpus_ids: list[str] = []
    correct_ids: set[str] = set()

    for session_id, session, timestamp in zip(
        entry["haystack_session_ids"],
        entry["haystack_sessions"],
        entry["haystack_dates"],
    ):
        user_turns = [
            (turn_index, turn)
            for turn_index, turn in enumerate(session)
            if turn["role"] == "user"
        ]

        if granularity == "session":
            content = "\n".join(turn["content"] for _, turn in user_turns)
            if include_date:
                content = f"Session date: {timestamp}\n{content}"
            corpus.append(content)
            corpus_ids.append(session_id)
            if any(turn.get("has_answer", False) for _, turn in user_turns):
                correct_ids.add(session_id)
            continue

        for turn_index, turn in user_turns:
            corpus_id = f"{session_id}_{turn_index + 1}"
            content = turn["content"]
            if include_date:
                content = f"Session date: {timestamp}\n{content}"
            corpus.append(content)
            corpus_ids.append(corpus_id)
            if turn.get("has_answer", False):
                correct_ids.add(corpus_id)

    return RetrievalCase(
        question_id=entry["question_id"],
        question_type=question_type(entry),
        query=entry["question"],
        corpus=corpus,
        corpus_ids=corpus_ids,
        correct_ids=correct_ids,
    )


def dcg(relevances: list[int], k: int) -> float:
    values = relevances[:k]
    if not values:
        return 0.0
    score = float(values[0])
    for rank, relevance in enumerate(values[1:], start=2):
        score += relevance / math.log2(rank)
    return score


def retrieval_metrics(
    ranked_ids: list[str],
    correct_ids: set[str],
    k: int,
) -> dict[str, float]:
    retrieved = set(ranked_ids[:k])
    relevances = [1 if item_id in correct_ids else 0 for item_id in ranked_ids]
    ideal = sorted(relevances, reverse=True)
    ideal_dcg = dcg(ideal, k)
    return {
        f"recall_any@{k}": float(bool(retrieved & correct_ids)),
        f"recall_all@{k}": float(correct_ids <= retrieved),
        f"ndcg_any@{k}": dcg(relevances, k) / ideal_dcg if ideal_dcg else 0.0,
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {
        key: round(sum(row[key] for row in rows) / len(rows), 4)
        for key in keys
    }


def calibrate_thresholds(
    answerable_rankings: list[tuple[list[tuple[str, float]], set[str]]],
    abstention_top_scores: list[float],
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLD_SWEEP,
) -> list[dict[str, float]]:
    """Measure retrieval recall and abstention specificity for each threshold."""
    rows = []
    for threshold in thresholds:
        retrieval_hits = sum(
            any(
                item_id in correct_ids and score >= threshold
                for item_id, score in ranking[:5]
            )
            for ranking, correct_ids in answerable_rankings
        )
        correct_abstentions = sum(
            top_score < threshold for top_score in abstention_top_scores
        )
        recall = (
            retrieval_hits / len(answerable_rankings)
            if answerable_rankings
            else 0.0
        )
        specificity = (
            correct_abstentions / len(abstention_top_scores)
            if abstention_top_scores
            else 0.0
        )
        rows.append(
            {
                "threshold": threshold,
                "recall_any@5": round(recall, 4),
                "abstention_specificity": round(specificity, 4),
                "balanced_score": round((recall + specificity) / 2, 4),
            }
        )
    return rows


def evaluate(
    cases: list[RetrievalCase],
    vectors: list[list[float]],
    slices: list[tuple[int, int, int]],
    threshold: float,
    save_details: bool,
) -> dict[str, Any]:
    standard_rows: list[dict[str, float]] = []
    threshold_rows: list[dict[str, float]] = []
    standard_by_type: dict[str, list[dict[str, float]]] = defaultdict(list)
    threshold_by_type: dict[str, list[dict[str, float]]] = defaultdict(list)
    answerable_rankings: list[tuple[list[tuple[str, float]], set[str]]] = []
    abstention_top_scores: list[float] = []
    details = []
    skipped = {"abstention": 0, "no_user_target": 0, "empty_corpus": 0}

    for case, (query_index, corpus_start, corpus_end) in zip(cases, slices):
        if not case.corpus:
            skipped["empty_corpus"] += 1
            continue

        query_vector = vectors[query_index]
        scored = [
            (
                corpus_index - corpus_start,
                embeddings.cosine_similarity(
                    query_vector,
                    vectors[corpus_index],
                ),
            )
            for corpus_index in range(corpus_start, corpus_end)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        ranked_ids = [case.corpus_ids[index] for index, _ in scored]
        ranking = [
            (case.corpus_ids[index], score)
            for index, score in scored
        ]

        if case.question_type == "abstention":
            skipped["abstention"] += 1
            abstention_top_scores.append(scored[0][1])
            continue
        if not case.correct_ids:
            skipped["no_user_target"] += 1
            continue

        answerable_rankings.append((ranking, case.correct_ids))
        thresholded_ids = [
            case.corpus_ids[index]
            for index, score in scored
            if score >= threshold
        ]

        standard = {}
        thresholded = {}
        for k in DEFAULT_CUTOFFS:
            standard.update(retrieval_metrics(ranked_ids, case.correct_ids, k))
            thresholded.update(
                retrieval_metrics(thresholded_ids, case.correct_ids, k)
            )

        standard_rows.append(standard)
        threshold_rows.append(thresholded)
        standard_by_type[case.question_type].append(standard)
        threshold_by_type[case.question_type].append(thresholded)

        if save_details:
            details.append(
                {
                    "question_id": case.question_id,
                    "question_type": case.question_type,
                    "query": case.query,
                    "correct_ids": sorted(case.correct_ids),
                    "ranked_items": [
                        {
                            "corpus_id": case.corpus_ids[index],
                            "score": round(score, 4),
                        }
                        for index, score in scored[:10]
                    ],
                    "metrics": {
                        "standard": standard,
                        "thresholded": thresholded,
                    },
                }
            )

    report = {
        "evaluated": len(standard_rows),
        "skipped": skipped,
        "standard": {
            "overall": mean_metrics(standard_rows),
            "by_type": {
                name: mean_metrics(rows)
                for name, rows in sorted(standard_by_type.items())
            },
        },
        "thresholded": {
            "overall": mean_metrics(threshold_rows),
            "by_type": {
                name: mean_metrics(rows)
                for name, rows in sorted(threshold_by_type.items())
            },
        },
        "threshold_calibration": calibrate_thresholds(
            answerable_rankings,
            abstention_top_scores,
        ),
    }
    if save_details:
        report["details"] = details
    return report


def format_metrics(metrics: dict[str, float]) -> str:
    return ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())


def print_report(report: dict[str, Any]) -> None:
    print(f"Evaluated: {report['evaluated']}")
    print(f"Skipped: {json.dumps(report['skipped'], sort_keys=True)}")
    print("\nStandard ranking:")
    print("  overall:", format_metrics(report["standard"]["overall"]))
    for name, metrics in report["standard"]["by_type"].items():
        print(f"  {name}: {format_metrics(metrics)}")
    print("\nContextMemory thresholded ranking:")
    print("  overall:", format_metrics(report["thresholded"]["overall"]))
    for name, metrics in report["thresholded"]["by_type"].items():
        print(f"  {name}: {format_metrics(metrics)}")
    calibration = report["threshold_calibration"]
    if calibration:
        best = max(calibration, key=lambda row: row["balanced_score"])
        print("\nThreshold calibration:")
        print(
            "  best balanced:",
            format_metrics(best),
        )


def main() -> None:
    args = parse_args()
    os.environ["CONTEXT_MEMORY_EMBEDDING_BACKEND"] = args.backend

    entries = json.loads(args.dataset.read_text(encoding="utf-8"))
    entries = select_entries(entries, args.limit, args.seed)
    cases = [
        build_case(entry, args.granularity, include_date=args.include_date)
        for entry in entries
    ]

    texts: list[str] = []
    slices: list[tuple[int, int, int]] = []
    for case in cases:
        query_index = len(texts)
        texts.append(case.query)
        corpus_start = len(texts)
        texts.extend(case.corpus)
        slices.append((query_index, corpus_start, len(texts)))

    started = time.perf_counter()
    vectors = embeddings.embed_many(texts, batch_size=args.batch_size)
    embedding_seconds = time.perf_counter() - started

    result = evaluate(
        cases,
        vectors,
        slices,
        threshold=args.threshold,
        save_details=args.save_details,
    )
    result["benchmark"] = {
        "name": "LongMemEval cleaned Oracle retrieval",
        "dataset": str(args.dataset),
        "selected_instances": len(entries),
        "backend": args.backend,
        "granularity": args.granularity,
        "include_date": args.include_date,
        "threshold": args.threshold,
        "seed": args.seed,
        "embedding_seconds": round(embedding_seconds, 3),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    report_path = args.report
    if report_path is None:
        DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        suffix = args.limit if args.limit is not None else "all"
        date_suffix = "_dated" if args.include_date else ""
        report_path = DEFAULT_REPORT_DIR / (
            f"retrieval_{args.backend}_{args.granularity}_{suffix}"
            f"{date_suffix}.json"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print_report(result)
    print(f"\nEmbedding time: {embedding_seconds:.3f}s")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
