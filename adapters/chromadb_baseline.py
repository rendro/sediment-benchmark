import chromadb

from .base import MemoryAdapter, MemoryItem, RecallResult

COLLECTION_NAME = "benchmark"


class ChromaDBAdapter(MemoryAdapter):
    name = "chromadb"

    def __init__(self) -> None:
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    @property
    def collection(self) -> chromadb.Collection:
        assert self._collection is not None, "Call setup() first"
        return self._collection

    async def setup(self) -> None:
        self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(name=COLLECTION_NAME)

    async def teardown(self) -> None:
        if self._client:
            self._client.delete_collection(name=COLLECTION_NAME)
        self._client = None
        self._collection = None

    async def reset(self) -> None:
        assert self._client is not None, "Call setup() first"
        self._client.delete_collection(name=COLLECTION_NAME)
        self._collection = self._client.create_collection(name=COLLECTION_NAME)

    async def store(self, item: MemoryItem) -> None:
        metadatas = None
        if item.metadata or item.scope:
            meta = dict(item.metadata) if item.metadata else {}
            if item.scope:
                meta["scope"] = item.scope
            metadatas = [meta] if meta else None

        self.collection.add(
            ids=[item.id],
            documents=[item.content],
            metadatas=metadatas,
        )

    async def recall(self, query: str, limit: int = 5) -> list[RecallResult]:
        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
        )

        items = []
        ids = results["ids"][0]
        documents = results["documents"][0]
        distances = results["distances"][0] if results.get("distances") else [None] * len(ids)

        for id_, doc, dist in zip(ids, documents, distances):
            # ChromaDB returns L2 distances (lower = more similar).
            # Convert to a similarity score in [0, 1].
            score = 1.0 / (1.0 + dist) if dist is not None else None
            items.append(RecallResult(id=id_, content=doc, score=score))

        return items

    async def count(self) -> int:
        return self.collection.count()
