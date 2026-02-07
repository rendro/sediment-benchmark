# Sediment Benchmark

Benchmark harness for AI agent memory systems — measures retrieval quality, temporal awareness, deduplication, and latency across multiple backends.

## Overview

This suite evaluates how well memory systems store and retrieve developer knowledge (architecture decisions, code patterns, troubleshooting notes, etc.) using a synthetic dataset of 1,000 memories and 200 queries across six categories.

**Systems under test:**

| System | Description |
|--------|-------------|
| **ChromaDB** | In-process vector database with default embeddings (baseline) |
| **Sediment** | MCP-based memory system with hybrid vector + FTS search |
| **Mem0** | AI memory layer with local Qdrant backend |
| **Letta** | REST-based agent framework with passage storage |

**Benchmark phases:**

| Phase | What it measures |
|-------|-----------------|
| **Retrieval** | Recall@k (k=1,3,5,10), MRR, nDCG@5 across difficulty levels and categories |
| **Temporal** | Whether the system returns the most recent version of updated facts (Recency@1/3, MRR) |
| **Dedup** | Consolidation of near-duplicate memories while preserving recall |
| **Latency** | Store and recall operation timing at p50, p95, p99 percentiles |

## Quick Start

```bash
# Clone and install
git clone https://github.com/rendro/sediment-benchmark.git
cd sediment-benchmark
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run benchmark (ChromaDB baseline)
python run.py --systems chromadb

# Run all systems, all phases
python run.py --systems chromadb,sediment,mem0,letta

# Generate report
python report.py
# Output: results/report.md, results/data.json
```

### System-specific requirements

- **Sediment**: Requires the `sediment` binary on PATH (or use `--sediment-bin`). For temporal/dedup phases, build with `cargo build --release --features bench`.
- **Mem0**: Installs with pip dependencies. Uses local Qdrant and HuggingFace embeddings.
- **Letta**: Requires a running Letta server (default `http://localhost:8283`).

## CLI Reference

### `run.py` — Benchmark runner

```
python run.py [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--systems` | `chromadb,sediment` | Comma-separated list: `chromadb`, `sediment`, `mem0`, `letta` |
| `--phases` | `all` | Comma-separated phases: `retrieval`, `temporal`, `dedup`, `latency` (or `all`) |
| `--seed` | `42` | Random seed for reproducibility |
| `--sediment-bin` | `sediment` | Path to Sediment binary |
| `--letta-url` | `http://localhost:8283` | Letta server URL |

Results are written to `results/raw/{system}_results_{timestamp}.json`.

### `report.py` — Report generator

```
python report.py
```

Reads all files from `results/raw/`, merges by system, and outputs:
- `results/report.md` — Markdown tables with best-in-row bolding
- `results/data.json` — Machine-readable aggregate

### `grid_search.py` — FTS parameter tuning

```
python grid_search.py --sediment-bin /path/to/sediment
```

Runs a 5×5 grid over `FTS_BOOST_MAX` and `FTS_GAMMA` values, outputting retrieval metrics for each combination.

## Project Structure

```
sediment-benchmark/
├── adapters/
│   ├── base.py                 # Abstract base class (MemoryAdapter)
│   ├── chromadb_baseline.py    # ChromaDB adapter
│   ├── sediment.py             # Sediment MCP adapter
│   ├── mem0_adapter.py         # Mem0 adapter
│   └── letta_adapter.py        # Letta REST adapter
├── dataset/
│   ├── generate_dataset.py     # Dataset generation (requires Anthropic API key)
│   ├── memories.jsonl          # 1,000 synthetic memories
│   ├── queries.jsonl           # 200 search queries with expected results
│   └── temporal.jsonl          # 50 temporal update sequences
├── metrics/
│   ├── retrieval.py            # Recall@k, MRR, nDCG
│   ├── temporal.py             # Recency, temporal MRR
│   ├── dedup.py                # Consolidation rate, post-dedup recall
│   └── latency.py              # Percentile latency stats
├── tests/                      # pytest suite for metrics and adapters
├── results/
│   ├── raw/                    # Per-run JSON result files
│   ├── report.md               # Generated markdown report
│   └── data.json               # Generated aggregate data
├── run.py                      # Main benchmark runner
├── report.py                   # Report generator
├── grid_search.py              # FTS parameter grid search
└── pyproject.toml              # Dependencies and project config
```

## Dataset

The benchmark ships with pre-generated datasets in `dataset/`. No API key needed to run benchmarks.

### `memories.jsonl` — 1,000 memories

Each entry has an `id`, `content`, `category`, `scope`, and `tags`:

```json
{"id": "mem_0001", "content": "...", "category": "architecture", "scope": "project-alpha", "tags": ["microservices"]}
```

**Categories:** architecture (200), code_patterns (200), project_facts (200), user_preferences (150), troubleshooting (150), cross_project (100)

### `queries.jsonl` — 200 queries

Each query specifies expected memory IDs and difficulty:

```json
{"id": "q_001", "query": "...", "expected": ["mem_0050"], "category": "architecture", "difficulty": "easy"}
```

**Difficulty split:** 50 easy, 100 medium, 50 hard

### `temporal.jsonl` — 50 sequences

Each sequence contains 2–4 versions of a fact, ordered oldest to newest:

```json
{"id": "t_001", "sequence": [{"content": "Uses React 16", "timestamp": 0}, {"content": "Upgraded to React 18", "timestamp": 1}], "query": "what React version", "expected_rank_1": 1}
```

### Regenerating the dataset

```bash
pip install -e ".[generate]"
ANTHROPIC_API_KEY=sk-... python dataset/generate_dataset.py
```

## Adding an Adapter

Implement the `MemoryAdapter` ABC from `adapters/base.py`:

```python
class MemoryAdapter(ABC):
    name: str

    async def setup(self) -> None: ...       # Initialize services
    async def teardown(self) -> None: ...     # Clean up
    async def reset(self) -> None: ...        # Wipe all memories
    async def store(self, item: MemoryItem) -> None: ...
    async def recall(self, query: str, limit: int = 5) -> list[RecallResult]: ...
    async def count(self) -> int: ...
```

Then register it in the `SYSTEMS` dict in `run.py`.

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

Tests cover all metric calculations and adapter interfaces. Adapter tests that require external services are skipped when the service is unavailable.

## License

MIT
