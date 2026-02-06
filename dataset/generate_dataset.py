#!/usr/bin/env python3
"""Generate benchmark dataset using Claude API.

Produces three files:
  - memories.jsonl  (1000 memory items across 6 categories)
  - queries.jsonl   (200 queries with ground-truth memory IDs)
  - temporal.jsonl  (50 fact-evolution sequences)

Usage:
  pip install anthropic
  ANTHROPIC_API_KEY=sk-... python dataset/generate_dataset.py [--seed 42]

The generated files are committed to the repo so the benchmark can run
without an API key.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import anthropic

MODEL = "claude-sonnet-4-5-20250929"
DATASET_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------

CATEGORIES = {
    "architecture": {
        "count": 200,
        "description": "Architecture and design decisions",
        "scope_pool": ["project-alpha", "project-beta", "project-gamma", "project-delta"],
        "tag_pool": [
            "database", "postgres", "mysql", "redis", "caching", "api",
            "graphql", "rest", "microservices", "monolith", "event-driven",
            "queue", "kafka", "aws", "gcp", "deployment", "scaling",
            "consistency", "availability", "networking", "cdn",
        ],
        "prompt": (
            "Generate {count} realistic software architecture decision memories. "
            "Each should describe a concrete technical choice and brief rationale. "
            "Cover databases, APIs, infrastructure, scaling, caching, messaging, "
            "deployment strategies, and service architecture. Vary the specificity — "
            "some short (1 sentence), some detailed (2-3 sentences). "
            "Make them sound like notes an engineer would save for future reference."
        ),
    },
    "code_patterns": {
        "count": 200,
        "description": "Code patterns and conventions",
        "scope_pool": ["project-alpha", "project-beta", "project-gamma"],
        "tag_pool": [
            "typescript", "python", "rust", "patterns", "middleware",
            "auth", "validation", "error-handling", "testing", "logging",
            "dependency-injection", "factory", "observer", "state-machine",
            "generics", "async", "concurrency", "orm", "serialization",
        ],
        "prompt": (
            "Generate {count} realistic code pattern and convention memories. "
            "Each should describe a specific pattern, idiom, or convention used "
            "in a codebase. Cover middleware patterns, error handling, auth flows, "
            "testing conventions, naming standards, module structure, type patterns, "
            "async patterns, and design patterns. Mix languages (TypeScript, Python, "
            "Rust, Go). Sound like notes from a working engineer."
        ),
    },
    "user_preferences": {
        "count": 150,
        "description": "User preferences and working style",
        "scope_pool": [None],  # preferences are often global
        "tag_pool": [
            "style", "formatting", "tooling", "editor", "terminal",
            "workflow", "testing", "documentation", "types", "functional",
            "oop", "naming", "git", "review", "debugging",
        ],
        "prompt": (
            "Generate {count} realistic developer preference memories. "
            "Each should express a specific preference about coding style, "
            "tooling, workflow, or conventions. Cover formatting preferences, "
            "language style (functional vs OOP), error handling philosophy, "
            "testing approach, documentation style, git workflow, review preferences, "
            "and editor/terminal setup. Sound like things a developer would tell "
            "an AI assistant to remember."
        ),
    },
    "project_facts": {
        "count": 200,
        "description": "Project facts and configuration",
        "scope_pool": ["project-alpha", "project-beta", "project-gamma", "project-delta", "project-epsilon"],
        "tag_pool": [
            "deploy", "ci-cd", "env", "config", "secrets", "domain",
            "staging", "production", "monitoring", "alerts", "dns",
            "ssl", "ports", "endpoints", "versions", "dependencies",
            "team", "schedule", "compliance", "sla",
        ],
        "prompt": (
            "Generate {count} realistic project fact memories. "
            "Each should state a concrete fact about a project's configuration, "
            "infrastructure, team, or processes. Cover deploy targets, CI/CD setup, "
            "environment variables, service endpoints, team contacts, SLAs, "
            "compliance requirements, release schedules, and dependency versions. "
            "Make them specific with realistic values (URLs, regions, ports, names)."
        ),
    },
    "troubleshooting": {
        "count": 150,
        "description": "Troubleshooting notes and fixes",
        "scope_pool": ["project-alpha", "project-beta", "project-gamma"],
        "tag_pool": [
            "bug", "fix", "workaround", "webpack", "docker", "node",
            "python", "memory-leak", "timeout", "crash", "performance",
            "race-condition", "ssl", "cors", "dependency", "migration",
            "build", "ci", "flaky-test",
        ],
        "prompt": (
            "Generate {count} realistic troubleshooting note memories. "
            "Each should describe a specific problem encountered and how it was "
            "fixed or worked around. Cover build failures, runtime errors, "
            "performance issues, dependency conflicts, Docker problems, "
            "CI failures, flaky tests, memory leaks, and configuration gotchas. "
            "Include the symptom and the fix. Sound like war stories."
        ),
    },
    "cross_project": {
        "count": 100,
        "description": "Cross-project knowledge and reusable patterns",
        "scope_pool": [None],  # cross-project by nature
        "tag_pool": [
            "reusable", "shared", "library", "pattern", "migration",
            "cross-team", "platform", "internal-tool", "api-gateway",
            "auth-service", "logging-service", "rate-limiter",
        ],
        "prompt": (
            "Generate {count} realistic cross-project knowledge memories. "
            "Each should describe something learned in one project that's useful "
            "in another. Cover reusable libraries, shared patterns, lessons from "
            "migrations, cross-team conventions, platform capabilities, and "
            "internal tools. Reference specific project names like project-alpha, "
            "project-beta, etc. Sound like wisdom worth sharing across teams."
        ),
    },
}

# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def call_claude(client: anthropic.Anthropic, prompt: str, system: str) -> str:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=8192,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def generate_memories(client: anthropic.Anthropic, rng: random.Random) -> list[dict]:
    """Generate all 1000 memory items."""
    system = (
        "You are generating a synthetic dataset for benchmarking memory systems. "
        "Return ONLY a JSON array of objects, each with a single \"content\" field. "
        "No markdown fences, no commentary. Just the JSON array."
    )

    all_memories = []
    mem_idx = 1

    for category, cfg in CATEGORIES.items():
        count = cfg["count"]
        prompt = cfg["prompt"].format(count=count)

        # Generate in batches of 50 to stay within output limits
        batch_size = 50
        generated = []
        while len(generated) < count:
            remaining = count - len(generated)
            n = min(batch_size, remaining)
            batch_prompt = prompt.replace(f"{count} realistic", f"{n} realistic")

            raw = call_claude(client, batch_prompt, system)
            try:
                items = json.loads(raw)
            except json.JSONDecodeError:
                # Try to extract JSON array from response
                start = raw.find("[")
                end = raw.rfind("]") + 1
                if start >= 0 and end > start:
                    items = json.loads(raw[start:end])
                else:
                    print(f"  WARNING: Failed to parse batch for {category}, retrying...", file=sys.stderr)
                    continue

            generated.extend(items)
            print(f"  {category}: {len(generated)}/{count}", file=sys.stderr)

        # Trim to exact count and assign IDs
        generated = generated[:count]
        for item in generated:
            scope = rng.choice(cfg["scope_pool"])
            tags = rng.sample(cfg["tag_pool"], k=rng.randint(1, 3))
            all_memories.append({
                "id": f"mem_{mem_idx:04d}",
                "content": item["content"],
                "category": category,
                "scope": scope,
                "tags": tags,
            })
            mem_idx += 1

    return all_memories


def generate_queries(
    client: anthropic.Anthropic, memories: list[dict], rng: random.Random
) -> list[dict]:
    """Generate 200 queries with ground-truth memory IDs."""
    system = (
        "You are generating search queries for a memory retrieval benchmark. "
        "You will be given a set of memory items. For each query you generate, "
        "return a JSON object with: \"query\" (the search query), "
        "\"expected_ids\" (array of memory IDs that are relevant), "
        "\"difficulty\" (\"easy\", \"medium\", or \"hard\"), "
        "and \"category\" (the memory category being targeted). "
        "Return ONLY a JSON array. No markdown fences, no commentary."
    )

    # Group memories by category for targeted query generation
    by_category = {}
    for m in memories:
        by_category.setdefault(m["category"], []).append(m)

    # Difficulty distribution: 50 easy, 100 medium, 50 hard
    difficulty_quotas = {"easy": 50, "medium": 100, "hard": 50}

    all_queries = []
    query_idx = 1

    for difficulty, quota in difficulty_quotas.items():
        # Spread queries across categories proportionally
        categories = list(CATEGORIES.keys())
        per_cat = quota // len(categories)
        remainder = quota % len(categories)

        for i, category in enumerate(categories):
            n = per_cat + (1 if i < remainder else 0)
            if n == 0:
                continue

            # Sample memories to reference
            cat_memories = by_category[category]
            sample = rng.sample(cat_memories, k=min(30, len(cat_memories)))
            memories_json = json.dumps(sample, indent=2)

            if difficulty == "easy":
                diff_instruction = (
                    "Generate EASY queries that nearly match the wording of a memory. "
                    "Each query should clearly map to 1-2 specific memories. "
                    "A simple keyword match would find them."
                )
            elif difficulty == "medium":
                diff_instruction = (
                    "Generate MEDIUM queries that require semantic understanding. "
                    "Rephrase concepts — don't reuse exact words from the memories. "
                    "Each query should map to 1-3 specific memories."
                )
            else:
                diff_instruction = (
                    "Generate HARD queries that require cross-concept reasoning. "
                    "Ask about combining ideas from different memories, or use "
                    "abstract descriptions that require inference to match. "
                    "Each query should map to 2-4 specific memories."
                )

            prompt = (
                f"{diff_instruction}\n\n"
                f"Generate exactly {n} queries targeting the '{category}' category.\n\n"
                f"Available memories:\n{memories_json}"
            )

            raw = call_claude(client, prompt, system)
            try:
                items = json.loads(raw)
            except json.JSONDecodeError:
                start = raw.find("[")
                end = raw.rfind("]") + 1
                if start >= 0 and end > start:
                    items = json.loads(raw[start:end])
                else:
                    print(f"  WARNING: Failed to parse queries for {category}/{difficulty}, retrying...", file=sys.stderr)
                    continue

            for item in items[:n]:
                all_queries.append({
                    "id": f"q_{query_idx:03d}",
                    "query": item["query"],
                    "expected": item.get("expected_ids", item.get("expected", [])),
                    "category": category,
                    "difficulty": difficulty,
                })
                query_idx += 1

            print(f"  queries ({difficulty}/{category}): {len(all_queries)}/200", file=sys.stderr)

    return all_queries[:200]


def generate_temporal(client: anthropic.Anthropic) -> list[dict]:
    """Generate 50 temporal fact-evolution sequences."""
    system = (
        "You are generating temporal fact sequences for a memory benchmark. "
        "Each sequence represents a fact that changes over time (2-4 versions). "
        "Return ONLY a JSON array where each element has: "
        "\"topic\" (brief label), "
        "\"sequence\" (array of {\"content\": str, \"timestamp\": int} from oldest to newest), "
        "\"query\" (a question whose answer is the LATEST version). "
        "No markdown fences, no commentary."
    )

    prompt = (
        "Generate 50 temporal fact-evolution sequences about software projects. "
        "Each sequence should have 2-4 versions of a fact that changes over time. "
        "Cover: API endpoints migrating, library versions upgrading, config values "
        "changing, team members rotating, deployment targets moving, database schemas "
        "evolving, feature flags toggling, pricing tiers updating, domain names changing, "
        "and architecture decisions being revised.\n\n"
        "Example:\n"
        "{\n"
        "  \"topic\": \"API endpoint\",\n"
        "  \"sequence\": [\n"
        "    {\"content\": \"API endpoint is https://api.v1.example.com\", \"timestamp\": 0},\n"
        "    {\"content\": \"API endpoint migrated to https://api.v2.example.com\", \"timestamp\": 1}\n"
        "  ],\n"
        "  \"query\": \"what is the current API endpoint\"\n"
        "}\n\n"
        "Make each sequence realistic and distinct. Vary the number of versions."
    )

    # Generate in two batches of 25
    all_sequences = []
    for batch_num in range(2):
        n = 25
        batch_prompt = prompt.replace("50 temporal", f"{n} temporal")
        if batch_num == 1:
            batch_prompt += "\n\nMake these different from typical API/database examples. Cover less obvious evolving facts."

        raw = call_claude(client, batch_prompt, system)
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                items = json.loads(raw[start:end])
            else:
                print(f"  WARNING: Failed to parse temporal batch {batch_num}", file=sys.stderr)
                continue

        all_sequences.extend(items)
        print(f"  temporal: {len(all_sequences)}/50", file=sys.stderr)

    # Assign IDs and normalize
    result = []
    for i, seq in enumerate(all_sequences[:50]):
        last_idx = len(seq["sequence"]) - 1
        result.append({
            "id": f"t_{i + 1:03d}",
            "sequence": seq["sequence"],
            "query": seq["query"],
            "expected_rank_1": last_idx,
        })

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def write_jsonl(path: Path, items: list[dict]) -> None:
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(items)} items to {path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    client = anthropic.Anthropic()

    print("=== Generating memories ===", file=sys.stderr)
    memories = generate_memories(client, rng)
    write_jsonl(DATASET_DIR / "memories.jsonl", memories)

    print("\n=== Generating queries ===", file=sys.stderr)
    queries = generate_queries(client, memories, rng)
    write_jsonl(DATASET_DIR / "queries.jsonl", queries)

    print("\n=== Generating temporal sequences ===", file=sys.stderr)
    temporal = generate_temporal(client)
    write_jsonl(DATASET_DIR / "temporal.jsonl", temporal)

    # Summary
    print(f"\nDone! Generated:", file=sys.stderr)
    print(f"  {len(memories)} memories", file=sys.stderr)
    print(f"  {len(queries)} queries", file=sys.stderr)
    print(f"  {len(temporal)} temporal sequences", file=sys.stderr)


if __name__ == "__main__":
    main()
