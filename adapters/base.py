from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class MemoryItem:
    id: str
    content: str
    scope: str | None = None
    metadata: dict | None = None
    created_at: int | None = None  # Unix timestamp override (bench builds only)


@dataclass
class RecallResult:
    id: str
    content: str
    score: float | None = None


class MemoryAdapter(ABC):
    name: str

    @abstractmethod
    async def setup(self) -> None:
        """Start services, initialize DB. Called once per run."""

    @abstractmethod
    async def teardown(self) -> None:
        """Stop services, clean up. Called once per run."""

    @abstractmethod
    async def reset(self) -> None:
        """Wipe all stored memories. Called between test phases."""

    @abstractmethod
    async def store(self, item: MemoryItem) -> None:
        """Store a single memory item."""

    @abstractmethod
    async def recall(self, query: str, limit: int = 5) -> list[RecallResult]:
        """Semantic recall. Return up to `limit` results, best first."""

    @abstractmethod
    async def count(self) -> int:
        """Return number of stored items."""
