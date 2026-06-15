"""Download an official cleaned LongMemEval dataset with progress reporting."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen


BASE_URL = (
    "https://huggingface.co/datasets/xiaowu0162/"
    "longmemeval-cleaned/resolve/main"
)
FILES = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}
OUTPUT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=FILES, default="s")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def download(url: str, destination: Path, force: bool = False) -> None:
    if destination.exists() and not force:
        print(f"Already downloaded: {destination}")
        return

    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    request = Request(url, headers=headers)

    with urlopen(request) as response:
        resumed = existing > 0 and response.status == 206
        if existing and not resumed:
            existing = 0
        mode = "ab" if resumed else "wb"
        remaining = int(response.headers.get("Content-Length", "0"))
        total = existing + remaining if remaining else 0
        downloaded = existing
        started = time.perf_counter()
        last_update = 0.0

        destination.parent.mkdir(parents=True, exist_ok=True)
        with partial.open(mode) as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                now = time.perf_counter()
                if now - last_update >= 0.5:
                    elapsed = max(now - started, 0.001)
                    speed = (downloaded - existing) / elapsed
                    progress = (
                        f"{downloaded / total:6.1%}" if total else "   ?  "
                    )
                    print(
                        f"\r{progress}  {format_bytes(downloaded)}  "
                        f"{format_bytes(int(speed))}/s",
                        end="",
                        flush=True,
                    )
                    last_update = now

    print()
    partial.replace(destination)
    print(f"Downloaded: {destination} ({format_bytes(destination.stat().st_size)})")


def main() -> None:
    args = parse_args()
    filename = FILES[args.variant]
    try:
        download(f"{BASE_URL}/{filename}", OUTPUT_DIR / filename, args.force)
    except KeyboardInterrupt:
        print("\nDownload interrupted; rerun the command to resume.", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
