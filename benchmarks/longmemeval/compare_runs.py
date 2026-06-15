"""Compare accuracy, reader context, and total usage across E2E runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    headers = (
        "run",
        "system",
        "accuracy",
        "completed",
        "reader_input",
        "extraction_input",
        "total_requests",
        "elapsed_s",
    )
    rows = []
    for path in args.summaries:
        summary = json.loads(path.read_text(encoding="utf-8"))
        usage = summary.get("generation_usage", {})
        answer = usage.get("answer", {})
        extraction = usage.get("extraction", {})
        total = usage.get("total", {})
        evaluation = summary.get("evaluation") or {}
        judge_requests = evaluation.get("usage", {}).get("requests", 0)
        rows.append(
            (
                summary["run_name"],
                summary["configuration"]["system"],
                evaluation.get("accuracy", 0.0),
                summary["completed"],
                answer.get("prompt_tokens", 0),
                extraction.get("prompt_tokens", 0),
                total.get("requests", 0) + judge_requests,
                summary["elapsed_seconds"],
            )
        )

    widths = [
        max(len(str(value)) for value in [header, *[row[index] for row in rows]])
        for index, header in enumerate(headers)
    ]
    print(
        "  ".join(
            str(header).ljust(width)
            for header, width in zip(headers, widths)
        )
    )
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(
            "  ".join(
                str(value).ljust(width)
                for value, width in zip(row, widths)
            )
        )


if __name__ == "__main__":
    main()
