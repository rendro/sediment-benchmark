import httpx
import pytest

from adapters.base import MemoryItem
from adapters.letta_adapter import LettaAdapter

LETTA_URL = "http://localhost:8283"


def _server_available() -> bool:
    try:
        r = httpx.get(f"{LETTA_URL}/v1/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _server_available(), reason="Letta server not running"
)


@pytest.fixture
async def adapter():
    a = LettaAdapter(base_url=LETTA_URL)
    await a.setup()
    yield a
    await a.teardown()


@pytest.mark.asyncio
async def test_store_and_count(adapter: LettaAdapter):
    assert await adapter.count() == 0
    await adapter.store(MemoryItem(id="m1", content="Postgres is the primary database"))
    assert await adapter.count() == 1


@pytest.mark.asyncio
async def test_recall_returns_relevant(adapter: LettaAdapter):
    await adapter.store(MemoryItem(id="m1", content="We use Postgres for the main database"))
    await adapter.store(MemoryItem(id="m2", content="Deploy target is us-east-1"))
    await adapter.store(MemoryItem(id="m3", content="Auth uses JWT tokens with RS256"))

    results = await adapter.recall("which database do we use", limit=2)
    assert len(results) == 2
    # Content-based assertion since Letta generates its own IDs
    assert "Postgres" in results[0].content or "database" in results[0].content
    # Letta search doesn't return similarity scores
    assert results[0].score is None


@pytest.mark.asyncio
async def test_reset_clears_data(adapter: LettaAdapter):
    await adapter.store(MemoryItem(id="m1", content="some fact"))
    assert await adapter.count() == 1
    await adapter.reset()
    assert await adapter.count() == 0


@pytest.mark.asyncio
async def test_store_with_metadata(adapter: LettaAdapter):
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
