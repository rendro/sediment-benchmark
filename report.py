#!/usr/bin/env python3
"""Generate comparison report from raw benchmark results.

Reads results/raw/*.json, produces:
  - results/report.md   (human-readable markdown)
  - results/data.json   (machine-readable aggregate)

Usage:
  python report.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RAW_DIR = RESULTS_DIR / "raw"
DATASET_DIR = Path(__file__).parent / "dataset"

# Metrics where lower is better (latency, rank); everything else is higher-is-better
_LOWER_IS_BETTER = {"mean_rank", "p50", "p95", "p99", "mean", "min", "max"}


def _display_name(system: str) -> str:
    """Capitalize system name for display (chromadb -> ChromaDB, mem0 -> Mem0)."""
    special = {"chromadb": "ChromaDB", "mem0": "Mem0"}
    return special.get(system, system.capitalize())


def _dataset_counts() -> tuple[int, int, int]:
    """Return (memories, queries, temporal_sequences) counts from dataset files."""
    counts = []
    for name in ("memories.jsonl", "queries.jsonl", "temporal.jsonl"):
        p = DATASET_DIR / name
        counts.append(sum(1 for _ in p.open()) if p.exists() else 0)
    return counts[0], counts[1], counts[2]


def load_latest_results() -> dict[str, dict]:
    """Load results with phase-level merging across files per system."""
    results: dict[str, dict] = {}
    if not RAW_DIR.exists():
        return results

    for f in sorted(RAW_DIR.glob("*_results_*.json")):
        data = json.loads(f.read_text())
        system = data.get("system", f.stem.split("_")[0])
        if system not in results:
            results[system] = data
        else:
            # Merge phases — later files overwrite same phase, but preserve others
            for phase, phase_data in data.get("phases", {}).items():
                results[system].setdefault("phases", {})[phase] = phase_data
            # Update system_info to latest
            if "system_info" in data:
                results[system]["system_info"] = data["system_info"]

    return results


def format_pct(value: float, bold: bool = False) -> str:
    s = f"{value * 100:.1f}%"
    return f"**{s}**" if bold else s


def format_ms(seconds: float, bold: bool = False) -> str:
    """Format seconds as milliseconds string, e.g. '12.3ms'."""
    s = f"{seconds * 1000:.1f}ms"
    return f"**{s}**" if bold else s


def _best_indices(values: list[float | None], lower_is_better: bool) -> set[int]:
    """Return indices of the best value(s). Only highlights when 2+ systems have data."""
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(valid) < 2:
        return set()
    best = min(v for _, v in valid) if lower_is_better else max(v for _, v in valid)
    return {i for i, v in valid if v == best}


def _inline_latency_note(all_results: dict[str, dict], phase: str, systems: list[str]) -> str | None:
    """Build an italic inline latency note for a phase, or None if no data."""
    parts = []
    for s in systems:
        latency = all_results[s].get("phases", {}).get(phase, {}).get("latency")
        if latency:
            store_p50 = format_ms(latency["store"]["p50"])
            recall_p50 = format_ms(latency["recall"]["p50"])
            parts.append(f"{_display_name(s)} store p50={store_p50}, recall p50={recall_p50}")
    if parts:
        return f"*Latency: {' | '.join(parts)}*"
    return None


def generate_markdown(all_results: dict[str, dict]) -> str:
    lines: list[str] = []
    lines.append("# Memory System Benchmark Results\n")

    systems = sorted(all_results.keys())
    if not systems:
        lines.append("No results found.\n")
        return "\n".join(lines)

    # Dataset info
    n_mem, n_query, n_temporal = _dataset_counts()
    lines.append(f"*{n_mem} memories, {n_query} queries, {n_temporal} temporal sequences · {len(systems)} systems compared*\n")

    display_names = [_display_name(s) for s in systems]

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    lines.append("## Summary\n")

    # Collect retrieval metrics for each system
    has_retrieval = any(
        "retrieval" in all_results[s].get("phases", {})
        and "error" not in all_results[s]["phases"]["retrieval"]
        for s in systems
    )

    if has_retrieval:
        lines.append("| Metric | " + " | ".join(display_names) + " |")
        lines.append("| --- | " + " | ".join(["---"] * len(systems)) + " |")

        metric_labels = [
            ("recall_at_1", "Recall@1"),
            ("recall_at_3", "Recall@3"),
            ("recall_at_5", "Recall@5"),
            ("recall_at_10", "Recall@10"),
            ("mrr", "MRR"),
            ("ndcg_at_5", "nDCG@5"),
        ]

        for key, label in metric_labels:
            values = []
            for s in systems:
                phase = all_results[s].get("phases", {}).get("retrieval", {})
                values.append(phase.get("metrics", {}).get(key, {}).get("aggregate"))
            best = _best_indices(values, lower_is_better=False)
            row = [f"**{label}**"]
            for i, val in enumerate(values):
                row.append(format_pct(val, bold=i in best) if val is not None else "—")
            lines.append("| " + " | ".join(row) + " |")

        note = _inline_latency_note(all_results, "retrieval", systems)
        if note:
            lines.append("")
            lines.append(note)

        lines.append("")

    # -----------------------------------------------------------------------
    # Per-category breakdown
    # -----------------------------------------------------------------------
    if has_retrieval:
        lines.append("## Retrieval Quality by Category\n")

        # Get all categories from the first system that has data
        categories = set()
        for s in systems:
            phase = all_results[s].get("phases", {}).get("retrieval", {})
            r5 = phase.get("metrics", {}).get("recall_at_5", {}).get("by_category", {})
            categories.update(r5.keys())
        categories = sorted(categories)

        if categories:
            lines.append("### Recall@5 by Category\n")
            lines.append("| Category | " + " | ".join(display_names) + " |")
            lines.append("| --- | " + " | ".join(["---"] * len(systems)) + " |")

            for cat in categories:
                values = []
                for s in systems:
                    phase = all_results[s].get("phases", {}).get("retrieval", {})
                    values.append(phase.get("metrics", {}).get("recall_at_5", {}).get("by_category", {}).get(cat))
                best = _best_indices(values, lower_is_better=False)
                row = [f"`{cat}`"]
                for i, val in enumerate(values):
                    row.append(format_pct(val, bold=i in best) if val is not None else "—")
                lines.append("| " + " | ".join(row) + " |")

            lines.append("")

            lines.append("### MRR by Category\n")
            lines.append("| Category | " + " | ".join(display_names) + " |")
            lines.append("| --- | " + " | ".join(["---"] * len(systems)) + " |")

            for cat in categories:
                values = []
                for s in systems:
                    phase = all_results[s].get("phases", {}).get("retrieval", {})
                    values.append(phase.get("metrics", {}).get("mrr", {}).get("by_category", {}).get(cat))
                best = _best_indices(values, lower_is_better=False)
                row = [f"`{cat}`"]
                for i, val in enumerate(values):
                    row.append(format_pct(val, bold=i in best) if val is not None else "—")
                lines.append("| " + " | ".join(row) + " |")

            lines.append("")

    # -----------------------------------------------------------------------
    # Temporal correctness
    # -----------------------------------------------------------------------
    has_temporal = any(
        "temporal" in all_results[s].get("phases", {})
        and "error" not in all_results[s]["phases"]["temporal"]
        for s in systems
    )

    if has_temporal:
        lines.append("## Temporal Correctness\n")
        lines.append("| Metric | " + " | ".join(display_names) + " |")
        lines.append("| --- | " + " | ".join(["---"] * len(systems)) + " |")

        temporal_metrics = [
            ("recency_at_1", "Recency@1"),
            ("recency_at_3", "Recency@3"),
            ("mrr", "MRR"),
            ("mean_rank", "Mean Rank"),
        ]

        for key, label in temporal_metrics:
            values = []
            for s in systems:
                phase = all_results[s].get("phases", {}).get("temporal", {})
                values.append(phase.get("metrics", {}).get(key))
            lower = key in _LOWER_IS_BETTER
            best = _best_indices(values, lower_is_better=lower)
            row = [f"**{label}**"]
            for i, val in enumerate(values):
                if val is not None:
                    if key == "mean_rank":
                        txt = f"{val:.2f}"
                        row.append(f"**{txt}**" if i in best else txt)
                    else:
                        row.append(format_pct(val, bold=i in best))
                else:
                    row.append("—")
            lines.append("| " + " | ".join(row) + " |")

        note = _inline_latency_note(all_results, "temporal", systems)
        if note:
            lines.append("")
            lines.append(note)

        lines.append("")

    # -----------------------------------------------------------------------
    # Dedup / consolidation
    # -----------------------------------------------------------------------
    has_dedup = any(
        "dedup" in all_results[s].get("phases", {})
        and "error" not in all_results[s]["phases"]["dedup"]
        for s in systems
    )

    if has_dedup:
        lines.append("## Dedup / Consolidation\n")
        lines.append("| Metric | " + " | ".join(display_names) + " |")
        lines.append("| --- | " + " | ".join(["---"] * len(systems)) + " |")

        dedup_metrics = [
            ("consolidation_rate", "Consolidation Rate"),
            ("recall_after_dedup", "Recall After Dedup"),
        ]

        for key, label in dedup_metrics:
            values = []
            for s in systems:
                phase = all_results[s].get("phases", {}).get("dedup", {})
                values.append(phase.get("metrics", {}).get(key))
            best = _best_indices(values, lower_is_better=False)
            row = [f"**{label}**"]
            for i, val in enumerate(values):
                row.append(format_pct(val, bold=i in best) if val is not None else "—")
            lines.append("| " + " | ".join(row) + " |")

        # Add stored counts row
        row = ["**Stored / Expected**"]
        for s in systems:
            phase = all_results[s].get("phases", {}).get("dedup", {})
            stored = phase.get("stored_count")
            expected = phase.get("expected_count")
            if stored is not None and expected is not None:
                row.append(f"{stored} / {expected}")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")

        note = _inline_latency_note(all_results, "dedup", systems)
        if note:
            lines.append("")
            lines.append(note)

        lines.append("")

    # -----------------------------------------------------------------------
    # Latency
    # -----------------------------------------------------------------------
    has_latency = any(
        "latency" in all_results[s].get("phases", {})
        and "error" not in all_results[s]["phases"]["latency"]
        for s in systems
    )

    if has_latency:
        lines.append("## Latency\n")

        latency_keys = [("p50", "p50"), ("p95", "p95"), ("p99", "p99"), ("mean", "Mean"), ("min", "Min"), ("max", "Max")]

        for op, op_label in [("store", "Store Latency"), ("recall", "Recall Latency")]:
            lines.append(f"### {op_label}\n")
            lines.append("| Metric | " + " | ".join(display_names) + " |")
            lines.append("| --- | " + " | ".join(["---"] * len(systems)) + " |")

            for key, label in latency_keys:
                values = []
                for s in systems:
                    phase = all_results[s].get("phases", {}).get("latency", {})
                    values.append(phase.get("metrics", {}).get(op, {}).get(key))
                best = _best_indices(values, lower_is_better=True)
                row = [f"**{label}**"]
                for i, val in enumerate(values):
                    row.append(format_ms(val, bold=i in best) if val is not None else "—")
                lines.append("| " + " | ".join(row) + " |")

            # Sample size row
            row = ["**Samples**"]
            for s in systems:
                phase = all_results[s].get("phases", {}).get("latency", {})
                count = phase.get("metrics", {}).get(op, {}).get("count")
                row.append(str(count) if count is not None else "—")
            lines.append("| " + " | ".join(row) + " |")

            lines.append("")

    # -----------------------------------------------------------------------
    # Errors / notes
    # -----------------------------------------------------------------------
    notes: list[str] = []
    for s in systems:
        data = all_results[s]
        dn = _display_name(s)
        if "error" in data:
            notes.append(f"- **{dn}**: {data['error']}")
        else:
            for phase_name, phase_data in data.get("phases", {}).items():
                se = phase_data.get("store_errors", 0)
                qe = phase_data.get("query_errors", 0)
                errs = phase_data.get("errors", 0)
                if se or qe:
                    notes.append(f"- **{dn}** ({phase_name}): {se} store errors, {qe} query errors")
                elif errs:
                    notes.append(f"- **{dn}** ({phase_name}): {errs} errors")

    if notes:
        lines.append("## Notes\n")
        lines.extend(notes)

    # -----------------------------------------------------------------------
    # Methodology
    # -----------------------------------------------------------------------
    lines.append("\n## Methodology\n")
    for s in systems:
        dn = _display_name(s)
        info = all_results[s].get("system_info", {})
        if info:
            lines.append(f"- **{dn}**: {info.get('platform', '?')}, Python {info.get('python_version', '?')}")
            lines.append(f"  - Run at: {info.get('timestamp', '?')}")

    # Generation timestamp
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append("")
    lines.append(f"*Generated: {now}*")
    lines.append("")
    return "\n".join(lines)


def generate_data_json(all_results: dict[str, dict]) -> dict:
    """Machine-readable aggregate: just metrics per system."""
    data = {}
    for system, result in all_results.items():
        entry: dict = {"system": system}
        retrieval = result.get("phases", {}).get("retrieval", {})
        if "metrics" in retrieval:
            entry["retrieval"] = {
                k: v["aggregate"]
                for k, v in retrieval["metrics"].items()
            }
        if "latency" in retrieval:
            entry.setdefault("phase_latency", {})["retrieval"] = retrieval["latency"]
        temporal = result.get("phases", {}).get("temporal", {})
        if "metrics" in temporal:
            entry["temporal"] = temporal["metrics"]
        if "latency" in temporal:
            entry.setdefault("phase_latency", {})["temporal"] = temporal["latency"]
        dedup = result.get("phases", {}).get("dedup", {})
        if "metrics" in dedup:
            entry["dedup"] = dedup["metrics"]
        if "latency" in dedup:
            entry.setdefault("phase_latency", {})["dedup"] = dedup["latency"]
        latency = result.get("phases", {}).get("latency", {})
        if "metrics" in latency:
            entry["latency"] = latency["metrics"]
        data[system] = entry
    return data


def main():
    print("Loading results...", flush=True)
    all_results = load_latest_results()

    if not all_results:
        print("No result files found in results/raw/", file=sys.stderr)
        sys.exit(1)

    print(f"Found results for: {list(all_results.keys())}", flush=True)

    # Generate markdown report
    md = generate_markdown(all_results)
    report_path = RESULTS_DIR / "report.md"
    report_path.write_text(md)
    print(f"Report written to {report_path}", flush=True)

    # Generate data.json
    data = generate_data_json(all_results)
    data_path = RESULTS_DIR / "data.json"
    data_path.write_text(json.dumps(data, indent=2))
    print(f"Data written to {data_path}", flush=True)


if __name__ == "__main__":
    main()
