from __future__ import annotations

import asyncio
import shutil
import tempfile

from mem0 import Memory

from .base import MemoryAdapter, MemoryItem, RecallResult

USER_ID = "bench"


class Mem0Adapter(MemoryAdapter):
    name = "mem0"

    def __init__(self) -> None:
        self._memory: Memory | None = None
        self._tmp_dir: str | None = None
        self._store_count: int = 0

    def _build_config(self) -> dict:
        assert self._tmp_dir is not None
        return {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "benchmark",
                    "embedding_model_dims": 384,
                    "path": self._tmp_dir,
                    "on_disk": False,
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "embedding_dims": 384,
                },
            },
            "llm": {
                "provider": "litellm",
                "config": {"model": "gpt-4.1-nano-2025-04-14"},
            },
            "history_db_path": f"{self._tmp_dir}/history.db",
        }

    async def setup(self) -> None:
        self._tmp_dir = tempfile.mkdtemp(prefix="mem0_bench_")
        self._memory = await asyncio.to_thread(Memory.from_config, self._build_config())

    async def teardown(self) -> None:
        self._memory = None
        if self._tmp_dir:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None

    async def reset(self) -> None:
        assert self._memory is not None, "Call setup() first"
        await asyncio.to_thread(self._memory.reset)
        self._store_count = 0

    async def store(self, item: MemoryItem) -> None:
        assert self._memory is not None, "Call setup() first"
        await asyncio.to_thread(
            self._memory.add, item.content, user_id=USER_ID, infer=False
        )
        self._store_count += 1

    async def recall(self, query: str, limit: int = 5) -> list[RecallResult]:
        assert self._memory is not None, "Call setup() first"
        raw = await asyncio.to_thread(
            self._memory.search, query, user_id=USER_ID, limit=limit
        )

        results = []
        entries = raw.get("results", []) if isinstance(raw, dict) else raw
        for entry in entries:
            if isinstance(entry, dict):
                rid = entry.get("id", "")
                content = entry.get("memory", entry.get("content", ""))
                score = entry.get("score")
            else:
                rid = getattr(entry, "id", "")
                content = getattr(entry, "memory", getattr(entry, "content", ""))
                score = getattr(entry, "score", None)
            results.append(RecallResult(id=str(rid), content=content, score=score))
        return results

    async def count(self) -> int:
        return self._store_count
