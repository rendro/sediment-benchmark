#!/usr/bin/env python3
"""Benchmark runner: stores memories, runs queries, scores results.

Usage:
  python run.py [--systems sediment,chromadb] [--phases retrieval] [--seed 42]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from adapters.base import MemoryAdapter, MemoryItem
from metrics.dedup import DedupResult, compute_dedup_metrics
from metrics.latency import compute_latency_report
from metrics.retrieval import QueryResult, compute_retrieval_metrics
from metrics.temporal import TemporalResult, compute_temporal_metrics

DATASET_DIR = Path(__file__).parent / "dataset"
RESULTS_DIR = Path(__file__).parent / "results" / "raw"

OPERATION_TIMEOUT = 120  # seconds (increased for rate-limit retries)


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

ADAPTERS: dict[str, type[MemoryAdapter]] = {}


def _register_adapters() -> None:
    from adapters.chromadb_baseline import ChromaDBAdapter
    from adapters.letta_adapter import LettaAdapter
    from adapters.mem0_adapter import Mem0Adapter
    from adapters.sediment import SedimentAdapter

    ADAPTERS["chromadb"] = ChromaDBAdapter
    ADAPTERS["letta"] = LettaAdapter
    ADAPTERS["mem0"] = Mem0Adapter
    ADAPTERS["sediment"] = SedimentAdapter


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_memories() -> list[dict]:
    return load_jsonl(DATASET_DIR / "memories.jsonl")


def load_queries() -> list[dict]:
    return load_jsonl(DATASET_DIR / "queries.jsonl")


def load_temporal() -> list[dict]:
    return load_jsonl(DATASET_DIR / "temporal.jsonl")


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------


async def _timed(coro):
    """Run a coroutine with a timeout, return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = await asyncio.wait_for(coro, timeout=OPERATION_TIMEOUT)
    elapsed = time.perf_counter() - start
    return result, elapsed


async def run_retrieval_phase(
    adapter: MemoryAdapter,
    memories: list[dict],
    queries: list[dict],
) -> dict:
    """Store all memories, run all queries, compute retrieval metrics."""
    print(f"  [{adapter.name}] Retrieval: resetting...", flush=True)
    await adapter.reset()

    # Build content→dataset_id map for systems that generate their own IDs
    content_to_id: dict[str, str] = {mem["content"]: mem["id"] for mem in memories}

    # Store all memories
    print(f"  [{adapter.name}] Retrieval: storing {len(memories)} memories...", flush=True)
    store_errors = 0
    store_timings: list[float] = []
    for i, mem in enumerate(memories):
        try:
            _, elapsed = await _timed(
                adapter.store(
                    MemoryItem(
                        id=mem["id"],
                        content=mem["content"],
                        scope=mem.get("scope"),
                        metadata={
                            k: v for k, v in mem.items()
                            if k not in ("id", "content", "scope")
                            and isinstance(v, (str, int, float, bool))
                        },
                    )
                )
            )
            store_timings.append(elapsed)
        except Exception as e:
            store_errors += 1
            if store_errors <= 3:
                print(f"    WARN: store error on {mem['id']}: {e}", file=sys.stderr)
        if (i + 1) % 200 == 0:
            print(f"    stored {i + 1}/{len(memories)}", flush=True)

    stored_count = await adapter.count()
    print(f"  [{adapter.name}] Retrieval: {stored_count} items stored, running {len(queries)} queries...", flush=True)

    # Run queries
    query_results: list[QueryResult] = []
    recall_timings: list[float] = []
    query_errors = 0
    for i, q in enumerate(queries):
        try:
            results, elapsed = await _timed(adapter.recall(q["query"], limit=10))
            recall_timings.append(elapsed)
            # Map returned IDs back to dataset IDs via content matching.
            # Systems like ChromaDB preserve our IDs; Sediment generates UUIDs.
            returned_ids = []
            for r in results:
                dataset_id = content_to_id.get(r.content, r.id)
                returned_ids.append(dataset_id)
            query_results.append(
                QueryResult(
                    query_id=q["id"],
                    returned_ids=returned_ids,
                    expected_ids=q["expected"],
                    category=q.get("category"),
                )
            )
        except Exception as e:
            query_errors += 1
            if query_errors <= 3:
                print(f"    WARN: query error on {q['id']}: {e}", file=sys.stderr)
            query_results.append(
                QueryResult(
                    query_id=q["id"],
                    returned_ids=[],
                    expected_ids=q["expected"],
                    category=q.get("category"),
                )
            )
        if (i + 1) % 50 == 0:
            print(f"    queried {i + 1}/{len(queries)}", flush=True)

    # Compute metrics
    report = compute_retrieval_metrics(query_results)

    latency = compute_latency_report(store_timings, recall_timings)

    return {
        "phase": "retrieval",
        "stored_count": stored_count,
        "store_errors": store_errors,
        "query_count": len(queries),
        "query_errors": query_errors,
        "metrics": {
            "recall_at_1": _scores_to_dict(report.recall_at_1),
            "recall_at_3": _scores_to_dict(report.recall_at_3),
            "recall_at_5": _scores_to_dict(report.recall_at_5),
            "recall_at_10": _scores_to_dict(report.recall_at_10),
            "mrr": _scores_to_dict(report.mrr),
            "ndcg_at_5": _scores_to_dict(report.ndcg_at_5),
        },
        "latency": _latency_to_dict(latency),
        "per_query": [
            {
                "query_id": qr.query_id,
                "returned_ids": qr.returned_ids,
                "expected_ids": qr.expected_ids,
                "category": qr.category,
            }
            for qr in query_results
        ],
    }


def _scores_to_dict(scores) -> dict:
    return {
        "aggregate": round(scores.aggregate, 4),
        "by_category": {k: round(v, 4) for k, v in scores.by_category.items()},
    }


def _stats_to_dict(stats) -> dict:
    return {
        "count": stats.count,
        "mean": round(stats.mean, 6),
        "min": round(stats.min, 6),
        "max": round(stats.max, 6),
        "p50": round(stats.p50, 6),
        "p95": round(stats.p95, 6),
        "p99": round(stats.p99, 6),
    }


def _latency_to_dict(report) -> dict:
    return {
        "store": _stats_to_dict(report.store),
        "recall": _stats_to_dict(report.recall),
    }


TEMPORAL_SPACING_DAYS = 30  # Days between each version in a temporal sequence


async def run_temporal_phase(
    adapter: MemoryAdapter,
    temporal: list[dict],
) -> dict:
    """Run temporal correctness phase: does the system return the latest fact?"""
    print(f"  [{adapter.name}] Temporal: running {len(temporal)} sequences...", flush=True)

    now = int(time.time())

    results: list[TemporalResult] = []
    store_timings: list[float] = []
    recall_timings: list[float] = []
    errors = 0

    for i, seq in enumerate(temporal):
        seq_id = seq["id"]
        sequence = seq["sequence"]
        query = seq["query"]
        expected_idx = seq["expected_rank_1"]

        try:
            await adapter.reset()

            # Backdate items: newest gets `now`, each older version is
            # TEMPORAL_SPACING_DAYS further in the past.
            max_ts = max(item["timestamp"] for item in sequence)

            content_to_id: dict[str, str] = {}
            for item in sequence:
                item_id = f"{seq_id}_v{item['timestamp']}"
                content_to_id[item["content"]] = item_id
                age_days = (max_ts - item["timestamp"]) * TEMPORAL_SPACING_DAYS
                created_at = now - age_days * 86400
                _, elapsed = await _timed(
                    adapter.store(
                        MemoryItem(
                            id=item_id,
                            content=item["content"],
                            created_at=created_at,
                        )
                    )
                )
                store_timings.append(elapsed)

            # Query and find rank of the expected (newest) item
            recall_results, elapsed = await _timed(
                adapter.recall(query, limit=len(sequence))
            )
            recall_timings.append(elapsed)

            expected_content = sequence[expected_idx]["content"]
            rank = None
            for r_idx, r in enumerate(recall_results, start=1):
                if r.content == expected_content:
                    rank = r_idx
                    break

            results.append(
                TemporalResult(
                    sequence_id=seq_id,
                    expected_content=expected_content,
                    rank=rank,
                )
            )
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"    WARN: temporal error on {seq_id}: {e}", file=sys.stderr)
            results.append(
                TemporalResult(
                    sequence_id=seq_id,
                    expected_content=sequence[expected_idx]["content"],
                    rank=None,
                )
            )

        if (i + 1) % 10 == 0:
            print(f"    processed {i + 1}/{len(temporal)} sequences", flush=True)

    report = compute_temporal_metrics(results)
    latency = compute_latency_report(store_timings, recall_timings)

    return {
        "phase": "temporal",
        "sequence_count": len(temporal),
        "errors": errors,
        "metrics": {
            "recency_at_1": round(report.recency_at_1, 4),
            "recency_at_3": round(report.recency_at_3, 4),
            "mrr": round(report.mrr, 4),
            "mean_rank": round(report.mean_rank, 4),
        },
        "latency": _latency_to_dict(latency),
        "per_sequence": [
            {
                "sequence_id": r.sequence_id,
                "expected_content": r.expected_content,
                "rank": r.rank,
            }
            for r in results
        ],
    }


def generate_paraphrase(content: str, index: int) -> str:
    """Generate a deterministic paraphrase of memory content."""
    variant = index % 4
    if variant == 0:
        return f"Note: {content}"
    elif variant == 1:
        return f"Remember that {content.lower()}"
    elif variant == 2:
        return f"{content} (confirmed)"
    else:
        return f"{content} — this is the current approach"


async def run_dedup_phase(
    adapter: MemoryAdapter,
    memories: list[dict],
) -> dict:
    """Run dedup phase: how does the system handle near-duplicate memories?"""
    pair_count = min(50, len(memories))
    print(f"  [{adapter.name}] Dedup: generating {pair_count} pairs...", flush=True)

    await adapter.reset()

    # Store originals and paraphrases
    store_errors = 0
    store_timings: list[float] = []
    pairs: list[tuple[str, str, str]] = []  # (pair_id, original, paraphrase)
    for i in range(pair_count):
        mem = memories[i]
        original = mem["content"]
        paraphrase = generate_paraphrase(original, i)
        pair_id = f"d_{i + 1:03d}"
        pairs.append((pair_id, original, paraphrase))

        try:
            _, elapsed = await _timed(
                adapter.store(MemoryItem(id=f"{pair_id}_orig", content=original))
            )
            store_timings.append(elapsed)
        except Exception as e:
            store_errors += 1
            if store_errors <= 3:
                print(f"    WARN: store error on {pair_id}_orig: {e}", file=sys.stderr)

        try:
            _, elapsed = await _timed(
                adapter.store(MemoryItem(id=f"{pair_id}_dup", content=paraphrase))
            )
            store_timings.append(elapsed)
        except Exception as e:
            store_errors += 1
            if store_errors <= 3:
                print(f"    WARN: store error on {pair_id}_dup: {e}", file=sys.stderr)

    expected_count = pair_count * 2
    stored_count = await adapter.count()
    print(
        f"  [{adapter.name}] Dedup: {stored_count}/{expected_count} items stored, "
        f"checking retrieval...",
        flush=True,
    )

    # Check retrievability of originals
    results: list[DedupResult] = []
    recall_timings: list[float] = []
    query_errors = 0
    for pair_id, original, paraphrase in pairs:
        try:
            recall_results, elapsed = await _timed(adapter.recall(original, limit=1))
            recall_timings.append(elapsed)
            # Check if original content is retrievable (exact or close match)
            retrieved = any(r.content == original for r in recall_results)
            results.append(
                DedupResult(
                    pair_id=pair_id,
                    original_content=original,
                    duplicate_content=paraphrase,
                    original_retrieved=retrieved,
                )
            )
        except Exception as e:
            query_errors += 1
            if query_errors <= 3:
                print(f"    WARN: recall error on {pair_id}: {e}", file=sys.stderr)
            results.append(
                DedupResult(
                    pair_id=pair_id,
                    original_content=original,
                    duplicate_content=paraphrase,
                    original_retrieved=False,
                )
            )

    report = compute_dedup_metrics(results, stored_count, expected_count)
    latency = compute_latency_report(store_timings, recall_timings)

    return {
        "phase": "dedup",
        "pair_count": pair_count,
        "expected_count": expected_count,
        "stored_count": stored_count,
        "store_errors": store_errors,
        "query_errors": query_errors,
        "metrics": {
            "consolidation_rate": round(report.consolidation_rate, 4),
            "recall_after_dedup": round(report.recall_after_dedup, 4),
        },
        "latency": _latency_to_dict(latency),
    }


async def run_latency_phase(
    adapter: MemoryAdapter,
    memories: list[dict],
) -> dict:
    """Standalone latency phase: focused 100-store + 50-recall workload."""
    store_count = min(100, len(memories))
    query_count = min(50, store_count)

    print(f"  [{adapter.name}] Latency: resetting...", flush=True)
    await adapter.reset()

    # Store first N memories, collecting timings
    print(f"  [{adapter.name}] Latency: storing {store_count} memories...", flush=True)
    store_timings: list[float] = []
    store_errors = 0
    stored_contents: list[str] = []
    for i in range(store_count):
        mem = memories[i]
        try:
            _, elapsed = await _timed(
                adapter.store(
                    MemoryItem(id=mem["id"], content=mem["content"])
                )
            )
            store_timings.append(elapsed)
            stored_contents.append(mem["content"])
        except Exception as e:
            store_errors += 1
            if store_errors <= 3:
                print(f"    WARN: store error on {mem['id']}: {e}", file=sys.stderr)

    # Recall using first 80 chars of stored content as queries (guarantees hits)
    print(f"  [{adapter.name}] Latency: running {query_count} queries...", flush=True)
    recall_timings: list[float] = []
    query_errors = 0
    for i in range(query_count):
        query = stored_contents[i][:80] if i < len(stored_contents) else memories[i]["content"][:80]
        try:
            _, elapsed = await _timed(adapter.recall(query, limit=5))
            recall_timings.append(elapsed)
        except Exception as e:
            query_errors += 1
            if query_errors <= 3:
                print(f"    WARN: recall error: {e}", file=sys.stderr)

    latency = compute_latency_report(store_timings, recall_timings)

    return {
        "phase": "latency",
        "store_count": store_count,
        "query_count": query_count,
        "store_errors": store_errors,
        "query_errors": query_errors,
        "metrics": _latency_to_dict(latency),
    }


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------


def get_system_info() -> dict:
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_system(
    name: str,
    adapter: MemoryAdapter,
    phases: list[str],
    memories: list[dict],
    queries: list[dict],
    temporal: list[dict],
) -> dict:
    """Run all requested phases for one system."""
    result: dict = {
        "system": name,
        "phases": {},
        "system_info": get_system_info(),
    }

    try:
        print(f"[{name}] Setting up...", flush=True)
        await adapter.setup()
    except Exception as e:
        print(f"[{name}] Setup FAILED: {e}", file=sys.stderr)
        result["error"] = f"setup failed: {e}"
        return result

    try:
        if "retrieval" in phases:
            try:
                phase_result = await run_retrieval_phase(adapter, memories, queries)
                result["phases"]["retrieval"] = phase_result
                m = phase_result["metrics"]
                print(
                    f"  [{name}] Retrieval done: "
                    f"Recall@1={m['recall_at_1']['aggregate']:.3f}  "
                    f"Recall@5={m['recall_at_5']['aggregate']:.3f}  "
                    f"MRR={m['mrr']['aggregate']:.3f}  "
                    f"nDCG@5={m['ndcg_at_5']['aggregate']:.3f}",
                    flush=True,
                )
            except Exception as e:
                print(f"  [{name}] Retrieval phase FAILED: {e}", file=sys.stderr)
                result["phases"]["retrieval"] = {"error": str(e)}

        if "temporal" in phases:
            try:
                phase_result = await run_temporal_phase(adapter, temporal)
                result["phases"]["temporal"] = phase_result
                m = phase_result["metrics"]
                print(
                    f"  [{name}] Temporal done: "
                    f"Recency@1={m['recency_at_1']:.3f}  "
                    f"Recency@3={m['recency_at_3']:.3f}  "
                    f"MRR={m['mrr']:.3f}  "
                    f"MeanRank={m['mean_rank']:.2f}",
                    flush=True,
                )
            except Exception as e:
                print(f"  [{name}] Temporal phase FAILED: {e}", file=sys.stderr)
                result["phases"]["temporal"] = {"error": str(e)}

        if "dedup" in phases:
            try:
                phase_result = await run_dedup_phase(adapter, memories)
                result["phases"]["dedup"] = phase_result
                m = phase_result["metrics"]
                print(
                    f"  [{name}] Dedup done: "
                    f"Consolidation={m['consolidation_rate']:.3f}  "
                    f"RecallAfterDedup={m['recall_after_dedup']:.3f}",
                    flush=True,
                )
            except Exception as e:
                print(f"  [{name}] Dedup phase FAILED: {e}", file=sys.stderr)
                result["phases"]["dedup"] = {"error": str(e)}

        if "latency" in phases:
            try:
                phase_result = await run_latency_phase(adapter, memories)
                result["phases"]["latency"] = phase_result
                m = phase_result["metrics"]
                print(
                    f"  [{name}] Latency done: "
                    f"store p50={m['store']['p50']*1000:.1f}ms  "
                    f"recall p50={m['recall']['p50']*1000:.1f}ms",
                    flush=True,
                )
            except Exception as e:
                print(f"  [{name}] Latency phase FAILED: {e}", file=sys.stderr)
                result["phases"]["latency"] = {"error": str(e)}
    finally:
        print(f"[{name}] Tearing down...", flush=True)
        try:
            await adapter.teardown()
        except Exception as e:
            print(f"[{name}] Teardown error: {e}", file=sys.stderr)

    return result


def write_results(system_name: str, data: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"{system_name}_results_{ts}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Results written to {path}", flush=True)
    return path


async def main_async(args: argparse.Namespace) -> None:
    _register_adapters()

    systems = [s.strip() for s in args.systems.split(",")]
    phases = [p.strip() for p in args.phases.split(",")]
    if "all" in phases:
        phases = ["retrieval", "temporal", "dedup", "latency"]

    # Validate systems
    for s in systems:
        if s not in ADAPTERS:
            print(f"Unknown system: {s}. Available: {list(ADAPTERS.keys())}", file=sys.stderr)
            sys.exit(1)

    # Load dataset
    print("Loading dataset...", flush=True)
    memories = load_memories()
    queries = load_queries()
    temporal = load_temporal()
    print(f"  {len(memories)} memories, {len(queries)} queries, {len(temporal)} temporal sequences", flush=True)

    # Run each system
    result_files = []
    for system_name in systems:
        print(f"\n{'='*60}", flush=True)
        print(f"  Running: {system_name}", flush=True)
        print(f"{'='*60}", flush=True)

        if system_name == "sediment":
            adapter = ADAPTERS[system_name](sediment_bin=args.sediment_bin)
        elif system_name == "letta":
            adapter = ADAPTERS[system_name](base_url=args.letta_url)
        else:
            adapter = ADAPTERS[system_name]()
        result = await run_system(system_name, adapter, phases, memories, queries, temporal)
        path = write_results(system_name, result)
        result_files.append(path)

    print(f"\nAll done. Result files:", flush=True)
    for p in result_files:
        print(f"  {p}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run memory system benchmark")
    parser.add_argument(
        "--systems",
        default="chromadb,sediment",
        help="Comma-separated list of systems to benchmark",
    )
    parser.add_argument(
        "--phases",
        default="all",
        help="Comma-separated phases: retrieval,temporal,dedup,latency or 'all'",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--sediment-bin",
        default="sediment",
        help="Path to sediment binary (use bench build for temporal phase)",
    )
    parser.add_argument(
        "--letta-url",
        default="http://localhost:8283",
        help="Base URL of the Letta server",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
