from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from letta_client import Letta

from .base import MemoryAdapter, MemoryItem, RecallResult


class LettaAdapter(MemoryAdapter):
    name = "letta"

    def __init__(self, base_url: str = "http://localhost:8283") -> None:
        self._base_url = base_url
        self._client: Letta | None = None
        self._agent_id: str | None = None
        self._store_count: int = 0

    def _unique_name(self) -> str:
        return f"bench_{uuid.uuid4().hex[:12]}"

    async def setup(self) -> None:
        self._client = await asyncio.to_thread(
            Letta, base_url=self._base_url
        )
        agent = await asyncio.to_thread(
            self._client.agents.create,
            name=self._unique_name(),
            embedding="letta/letta-free",
            include_base_tools=False,
        )
        self._agent_id = agent.id
        self._store_count = 0

    async def teardown(self) -> None:
        if self._client and self._agent_id:
            try:
                await asyncio.to_thread(
                    self._client.agents.delete, self._agent_id
                )
            except Exception:
                pass
        self._client = None
        self._agent_id = None

    async def reset(self) -> None:
        assert self._client is not None, "Call setup() first"
        # Delete current agent and create a fresh one
        if self._agent_id:
            try:
                await asyncio.to_thread(
                    self._client.agents.delete, self._agent_id
                )
            except Exception:
                pass
        agent = await asyncio.to_thread(
            self._client.agents.create,
            name=self._unique_name(),
            embedding="letta/letta-free",
            include_base_tools=False,
        )
        self._agent_id = agent.id
        self._store_count = 0

    async def store(self, item: MemoryItem) -> None:
        assert self._client is not None and self._agent_id is not None, "Call setup() first"
        kwargs: dict = {"text": item.content}
        if item.created_at is not None:
            kwargs["created_at"] = datetime.fromtimestamp(
                item.created_at, tz=timezone.utc
            )
        await asyncio.to_thread(
            lambda: self._client.agents.passages.create(  # type: ignore[union-attr]
                self._agent_id,  # type: ignore[arg-type]
                **kwargs,
            )
        )
        self._store_count += 1

    async def recall(self, query: str, limit: int = 5) -> list[RecallResult]:
        assert self._client is not None and self._agent_id is not None, "Call setup() first"
        raw = await asyncio.to_thread(
            lambda: self._client.agents.passages.search(  # type: ignore[union-attr]
                self._agent_id,  # type: ignore[arg-type]
                query=query,
                top_k=limit,
            )
        )

        results: list[RecallResult] = []
        # PassageSearchResponse has .results list of Result objects
        entries = getattr(raw, "results", []) if not isinstance(raw, list) else raw
        for entry in entries:
            if isinstance(entry, dict):
                rid = entry.get("id", "")
                content = entry.get("content", entry.get("text", ""))
            else:
                rid = getattr(entry, "id", "")
                content = getattr(entry, "content", getattr(entry, "text", ""))
            results.append(RecallResult(id=str(rid), content=content, score=None))
        return results

    async def count(self) -> int:
        return self._store_count
