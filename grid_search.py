#!/usr/bin/env python3
"""Grid search over FTS boost parameters for sediment.

Runs retrieval-only benchmarks across (FTS_BOOST_MAX, FTS_GAMMA) combinations
and prints a results table sorted by R@1 then MRR.

Usage:
  python grid_search.py --sediment-bin /path/to/sediment
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from adapters.sediment import SedimentAdapter
from run import load_memories, load_queries, run_retrieval_phase


@dataclass
class GridResult:
    boost_max: float
    gamma: float
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_5: float


async def run_single(
    sediment_bin: str,
    boost_max: float,
    gamma: float,
    memories: list[dict],
    queries: list[dict],
) -> GridResult:
    """Run a single retrieval benchmark with the given FTS parameters."""
    extra_env = {
        "SEDIMENT_FTS_BOOST_MAX": str(boost_max),
        "SEDIMENT_FTS_GAMMA": str(gamma),
    }
    adapter = SedimentAdapter(sediment_bin=sediment_bin, extra_env=extra_env)
    await adapter.setup()
    try:
        result = await run_retrieval_phase(adapter, memories, queries)
    finally:
        await adapter.teardown()

    m = result["metrics"]
    return GridResult(
        boost_max=boost_max,
        gamma=gamma,
        recall_at_1=m["recall_at_1"]["aggregate"],
        recall_at_5=m["recall_at_5"]["aggregate"],
        recall_at_10=m["recall_at_10"]["aggregate"],
        mrr=m["mrr"]["aggregate"],
        ndcg_at_5=m["ndcg_at_5"]["aggregate"],
    )


async def main_async(args: argparse.Namespace) -> None:
    boost_values = [0.04, 0.06, 0.08, 0.10, 0.12]
    gamma_values = [1.0, 1.5, 2.0, 3.0, 4.0]

    combos = list(product(boost_values, gamma_values))
    print(f"Grid search: {len(combos)} combinations")
    print(f"  FTS_BOOST_MAX: {boost_values}")
    print(f"  FTS_GAMMA:     {gamma_values}")
    print()

    # Load dataset once
    memories = load_memories()
    queries = load_queries()
    print(f"  {len(memories)} memories, {len(queries)} queries\n")

    results: list[GridResult] = []
    for i, (boost_max, gamma) in enumerate(combos):
        label = f"[{i + 1}/{len(combos)}] boost={boost_max:.2f} gamma={gamma:.1f}"
        print(f"--- {label} ---", flush=True)
        r = await run_single(args.sediment_bin, boost_max, gamma, memories, queries)
        results.append(r)
        print(
            f"  => R@1={r.recall_at_1:.1%}  R@5={r.recall_at_5:.1%}  "
            f"R@10={r.recall_at_10:.1%}  MRR={r.mrr:.1%}  nDCG@5={r.ndcg_at_5:.1%}\n",
            flush=True,
        )

    # Sort by R@1 desc, then MRR desc
    results.sort(key=lambda r: (r.recall_at_1, r.mrr), reverse=True)

    # Print results table
    print("\n" + "=" * 80)
    print("GRID SEARCH RESULTS (sorted by R@1, then MRR)")
    print("=" * 80)
    header = f"{'Boost':>6}  {'Gamma':>5}  {'R@1':>6}  {'R@5':>6}  {'R@10':>6}  {'MRR':>6}  {'nDCG@5':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.boost_max:>6.2f}  {r.gamma:>5.1f}  "
            f"{r.recall_at_1:>5.1%}  {r.recall_at_5:>5.1%}  "
            f"{r.recall_at_10:>5.1%}  {r.mrr:>5.1%}  {r.ndcg_at_5:>5.1%}"
        )

    best = results[0]
    print(f"\nBest: boost={best.boost_max:.2f} gamma={best.gamma:.1f}")
    print(f"  R@1={best.recall_at_1:.1%}  R@5={best.recall_at_5:.1%}  MRR={best.mrr:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Grid search FTS boost parameters")
    parser.add_argument(
        "--sediment-bin",
        required=True,
        help="Path to sediment bench build binary",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
