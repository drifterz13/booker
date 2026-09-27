import json
from pathlib import Path
from typing import cast

import chromadb

from booker.model.embedded_chunk import EmbeddedChunk
from booker.model.search_hit import SearchHit


class ChromaStore:
    """Persist dense embeddings with their source metadata locally."""

    def __init__(
        self, directory: Path, *, model_name: str = "bge-m3", collection_name: str
    ) -> None:
        client = chromadb.PersistentClient(path=str(directory))
        self._collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={"embedding_model": model_name},
            configuration={"hnsw": {"space": "cosine"}},
        )
        if (self._collection.metadata or {}).get("embedding_model") != model_name:
            raise ValueError("Collection uses a different embedding model")

    def sync(self, source: Path, items: list[EmbeddedChunk]) -> None:
        """Replace stored items for a source, removing any stale entries."""

        source_id = str(source.resolve())
        old_ids = set(
            self._collection.get(where={"source": source_id}, include=[])["ids"]
        )
        ids = [
            f"{source_id}:{item.chunk.segment_index}:{item.chunk.chunk_index}"
            for item in items
        ]

        for start in range(0, len(items), 100):
            batch = items[start : start + 100]
            self._collection.upsert(
                ids=ids[start : start + 100],
                embeddings=[item.vector for item in batch],
                documents=[item.chunk.text for item in batch],
                metadatas=[
                    {
                        "source": source_id,
                        "path": json.dumps(item.chunk.path, ensure_ascii=False),
                        "pages": json.dumps(item.chunk.pages),
                    }
                    for item in batch
                ],
            )

        if stale_ids := old_ids - set(ids):
            self._collection.delete(ids=list(stale_ids))

    def search(
        self,
        query_vector: list[float],
        *,
        limit: int = 5,
        source: Path | None = None,
    ) -> list[SearchHit]:
        """Search by a query vector from the same embedding model."""

        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            where={"source": str(source.resolve())} if source else None,
            include=["documents", "metadatas", "distances"],
        )

        documents = cast(list[list[str]], result["documents"])
        metadatas = cast(list[list[dict[str, str]]], result["metadatas"])
        distances = cast(list[list[float]], result["distances"])

        return [
            SearchHit(
                source=Path(metadata["source"]),
                path=tuple(json.loads(metadata["path"])),
                pages=tuple(json.loads(metadata["pages"])),
                text=document,
                distance=distance,
            )
            for document, metadata, distance in zip(
                documents[0], metadatas[0], distances[0], strict=True
            )
        ]
