"""Evaluate UrdWell retrieval with hybrid BM25 + cosine ranking (RRF).

Same harness as ``run_retrieval.py`` (identical dataset, cases, and metrics);
the only difference is the ranking strategy. Dense cosine scores are fused with
a lexical BM25 ranking through Reciprocal Rank Fusion, so exact-term matches the
embedding under-ranks get rescued. Abstention stays a pure cosine decision, so
the abstention/specificity numbers match the baseline run by construction.

Run the baseline (no RRF) with ``run_retrieval.py`` and this hybrid variant on
the same ``--limit``/``--seed`` to compare recall and NDCG on an identical stack:

    uv run python benchmarks/longmemeval/run_retrieval.py        --limit 100
    uv run python benchmarks/longmemeval/run_retrieval_hybrid.py --limit 100
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, BENCHMARK_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from urdwell import ranking
from urdwell.pipeline import SIMILARITY_THRESHOLD
from run_retrieval import (
    DEFAULT_DATASET,
    DEFAULT_REPORT_DIR,
    RankingStrategy,
    evaluate,
    prepare_cases,
    print_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--backend", choices=["fastembed", "hashing"], default="fastembed")
    parser.add_argument("--granularity", choices=["turn", "session"], default="turn")
    parser.add_argument(
        "--include-date",
        action="store_true",
        help="Prefix indexed text with the session date (UrdWell variant).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run a reproducible stratified subset instead of all cases.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD)
    parser.add_argument(
        "--pool-size",
        type=int,
        default=ranking.DEFAULT_POOL_SIZE,
        help="How many top-cosine candidates the lexical leg may reorder.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=ranking.RRF_K,
        help="Reciprocal Rank Fusion constant.",
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--save-details",
        action="store_true",
        help="Include per-question rankings and metrics in the report.",
    )
    return parser.parse_args()


def make_hybrid_ranking(pool_size: int, rrf_k: int) -> RankingStrategy:
    """Fuse the cosine ranking with a BM25 ranking over the top-cosine pool."""

    def hybrid_ranking(
        query: str,
        cosines: list[float],
        corpus_texts: list[str],
    ) -> list[tuple[int, float]]:
        cosine_order = sorted(range(len(cosines)), key=lambda i: cosines[i], reverse=True)
        pool = cosine_order[:pool_size]

        bm25 = ranking.bm25_scores(query, [corpus_texts[index] for index in pool])
        bm25_order = [
            pool[position]
            for position in sorted(range(len(pool)), key=lambda j: bm25[j], reverse=True)
        ]

        fused = ranking.reciprocal_rank_fusion(pool, bm25_order, k=rrf_k)
        fused_pool = sorted(pool, key=lambda index: fused[index], reverse=True)

        # The reordered pool first, then the cosine tail unchanged. The reported
        # score stays the cosine (interpretable magnitude); only order is fused.
        ordered = fused_pool + cosine_order[pool_size:]
        return [(index, cosines[index]) for index in ordered]

    return hybrid_ranking


def main() -> None:
    args = parse_args()
    cases, vectors, slices, embedding_seconds, selected = prepare_cases(args)

    result = evaluate(
        cases,
        vectors,
        slices,
        threshold=args.threshold,
        save_details=args.save_details,
        rank_fn=make_hybrid_ranking(args.pool_size, args.rrf_k),
    )
    result["benchmark"] = {
        "name": "LongMemEval cleaned Oracle retrieval (hybrid BM25 + cosine, RRF)",
        "dataset": str(args.dataset),
        "selected_instances": selected,
        "backend": args.backend,
        "granularity": args.granularity,
        "include_date": args.include_date,
        "threshold": args.threshold,
        "pool_size": args.pool_size,
        "rrf_k": args.rrf_k,
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
            f"retrieval_hybrid_{args.backend}_{args.granularity}_{suffix}{date_suffix}.json"
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
