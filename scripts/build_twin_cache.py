#!/usr/bin/env python3
"""Build twin forecast cache for the test partition of each cluster.

Writes data/twin_cache/<cluster>_forecasts.parquet (git-ignored).
Uses vectorised causal features — target <10 min per cluster.
"""

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ttwin.twin.engine import TwinEngine


def build_cache(cluster: str):
    t0 = time.time()
    print(f"[{cluster}] Loading engine...")
    engine = TwinEngine(cluster)

    out_dir = ROOT / "data" / "twin_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    hosts = engine.hosts()
    print(f"[{cluster}] Computing forecasts for {len(hosts)} hosts...")

    all_frames = []
    for i, host in enumerate(hosts):
        df = engine.forecast_vectorised(host)
        if len(df) > 0:
            all_frames.append(df)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(hosts)} hosts done")

    if not all_frames:
        print(f"[{cluster}] No forecasts produced!")
        return

    cache = pd.concat(all_frames, ignore_index=True)
    out_path = out_dir / f"{cluster}_forecasts.parquet"
    cache.to_parquet(out_path, index=False)

    elapsed = time.time() - t0
    print(f"[{cluster}] Cache: {len(cache):,} rows, {out_path.stat().st_size / 1024 / 1024:.1f} MB, {elapsed:.1f}s")


if __name__ == "__main__":
    for cluster in ["d1", "d2"]:
        build_cache(cluster)
