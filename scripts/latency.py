"""p50/p95 latency of POST /recommend over random queries against a running API.

Usage (API running via `make api`):
    uv --directory backend run python ../scripts/latency.py --n 100
Interests are random article titles from the DEV subset (1-3 per query), surprise random.
Writes reports/api_latency.md.
"""

import argparse
import random
import time
from pathlib import Path

import httpx
import numpy as np
import polars as pl

from rabbithole.config import REPO_ROOT, get_settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    settings = get_settings()
    ids = settings.data_dir / "processed" / settings.dataset_mode / "article_ids.parquet"
    titles = [t.replace("_", " ") for t in pl.read_parquet(ids)["title"].to_list()]
    rng = random.Random(0)
    times: list[float] = []
    errors = 0
    with httpx.Client(base_url=args.url, timeout=30) as client:
        client.post("/recommend", json={"interests": ["warm up"]})
        for _ in range(args.n):
            body = {
                "interests": rng.sample(titles, rng.randint(1, 3)),
                "surprise": round(rng.random(), 2),
            }
            start = time.perf_counter()
            r = client.post("/recommend", json=body)
            times.append((time.perf_counter() - start) * 1000)
            errors += r.status_code != 200
    p50, p95, p99 = np.percentile(times, [50, 95, 99])
    verdict = "PASS" if p95 < 800 else "FAIL"
    text = (
        f"# /recommend latency ({settings.dataset_mode.upper()}, local)\n\n"
        f"{args.n} random queries (1-3 article titles as interests, random surprise), "
        f"sequential, end to end over HTTP, encoder on CPU.\n\n"
        f"| p50 | p95 | p99 | max | non-200 |\n|---|---|---|---|---|\n"
        f"| {p50:.0f} ms | {p95:.0f} ms | {p99:.0f} ms | {max(times):.0f} ms | {errors} |\n\n"
        f"Target p95 < 800 ms: **{verdict}**\n"
    )
    out: Path = REPO_ROOT / "reports" / "api_latency.md"
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
