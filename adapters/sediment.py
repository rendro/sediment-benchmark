from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .base import MemoryAdapter, MemoryItem, RecallResult

# Retry settings for rate-limited operations
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0  # seconds


class SedimentAdapter(MemoryAdapter):
    name = "sediment"

    def __init__(
        self,
        sediment_bin: str = "sediment",
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self._bin = sediment_bin
        self._extra_env = extra_env or {}
        self._db_dir: str | None = None
        self._stdio_cm: object | None = None
        self._session_cm: object | None = None
        self._session: ClientSession | None = None
        self._store_count: int = 0

    @property
    def session(self) -> ClientSession:
        assert self._session is not None, "Call setup() first"
        return self._session

    async def _connect(self) -> None:
        """Start sediment subprocess and establish MCP session."""
        assert self._db_dir is not None
        server_params = StdioServerParameters(
            command=self._bin,
            env={**os.environ, "SEDIMENT_DB": self._db_dir, **self._extra_env},
        )
        self._stdio_cm = stdio_client(server_params)
        read, write = await self._stdio_cm.__aenter__()
        self._session_cm = ClientSession(read, write)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()

    async def _disconnect(self) -> None:
        """Close MCP session and stop subprocess."""
        try:
            if self._session_cm is not None:
                await self._session_cm.__aexit__(None, None, None)
        except Exception:
            pass
        finally:
            self._session_cm = None
            self._session = None

        try:
            if self._stdio_cm is not None:
                await self._stdio_cm.__aexit__(None, None, None)
        except Exception:
            pass
        finally:
            self._stdio_cm = None

    async def _call_with_retry(self, tool: str, args: dict) -> object:
        """Call an MCP tool with exponential backoff on rate limit errors."""
        backoff = INITIAL_BACKOFF
        for attempt in range(MAX_RETRIES + 1):
            result = await self.session.call_tool(tool, args)
            if not result.isError:
                return result
            text = result.content[0].text if result.content else ""
            if "rate limit" not in text.lower():
                raise RuntimeError(f"Sediment {tool} failed: {text}")
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Sediment {tool} rate limited after {MAX_RETRIES} retries")
            await asyncio.sleep(backoff)
            backoff *= 2
        return result  # unreachable

    async def setup(self) -> None:
        self._db_dir = tempfile.mkdtemp(prefix="sediment_bench_")
        await self._connect()

    async def teardown(self) -> None:
        await self._disconnect()
        if self._db_dir and os.path.exists(self._db_dir):
            shutil.rmtree(self._db_dir, ignore_errors=True)
        self._db_dir = None

    async def reset(self) -> None:
        await self._disconnect()
        if self._db_dir and os.path.exists(self._db_dir):
            shutil.rmtree(self._db_dir, ignore_errors=True)
        os.makedirs(self._db_dir, exist_ok=True)
        self._store_count = 0
        await self._connect()

    async def store(self, item: MemoryItem) -> None:
        args: dict = {"content": item.content}
        if item.created_at is not None:
            args["created_at"] = item.created_at
        await self._call_with_retry("store", args)
        self._store_count += 1

    async def recall(self, query: str, limit: int = 5) -> list[RecallResult]:
        result = await self._call_with_retry("recall", {"query": query, "limit": limit})
        text = result.content[0].text
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []

        items = []
        for r in data.get("results", []):
            score = float(r["similarity"]) if "similarity" in r else None
            items.append(RecallResult(id=r["id"], content=r["content"], score=score))
        return items

    async def count(self) -> int:
        """Query Sediment's list tool for the real item count."""
        try:
            result = await self._call_with_retry("list", {"limit": 1, "scope": "all"})
            data = json.loads(result.content[0].text)
            return data.get("count", self._store_count)
        except Exception:
            return self._store_count
