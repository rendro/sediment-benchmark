import pytest

from adapters.base import MemoryItem
from adapters.mem0_adapter import Mem0Adapter


@pytest.fixture
async def adapter():
    a = Mem0Adapter()
    await a.setup()
    yield a
    await a.teardown()


@pytest.mark.asyncio
async def test_store_and_count(adapter: Mem0Adapter):
    assert await adapter.count() == 0
    await adapter.store(MemoryItem(id="m1", content="Postgres is the primary database"))
    assert await adapter.count() == 1


@pytest.mark.asyncio
async def test_recall_returns_relevant(adapter: Mem0Adapter):
    await adapter.store(MemoryItem(id="m1", content="We use Postgres for the main database"))
    await adapter.store(MemoryItem(id="m2", content="Deploy target is us-east-1"))
    await adapter.store(MemoryItem(id="m3", content="Auth uses JWT tokens with RS256"))

    results = await adapter.recall("which database do we use", limit=2)
    assert len(results) == 2
    assert "Postgres" in results[0].content or "database" in results[0].content
    assert results[0].score is not None


@pytest.mark.asyncio
async def test_reset_clears_data(adapter: Mem0Adapter):
    await adapter.store(MemoryItem(id="m1", content="some fact"))
    assert await adapter.count() == 1
    await adapter.reset()
    assert await adapter.count() == 0


@pytest.mark.asyncio
async def test_store_with_metadata(adapter: Mem0Adapter):
    await adapter.store(
        MemoryItem(
            id="m1",
            content="Use Result types for error handling",
            scope="project-alpha",
            metadata={"category": "preferences"},
        )
    )
    assert await adapter.count() == 1
    results = await adapter.recall("error handling approach")
    assert len(results) >= 1
    assert "Result types" in results[0].content
