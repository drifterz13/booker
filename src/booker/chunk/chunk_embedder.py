from langchain_core.embeddings import Embeddings

from booker.model.chunk import Chunk
from booker.model.embedded_chunk import EmbeddedChunk


class ChunkEmbedder:
    """Embed book chunks in small batches without losing their source metadata."""

    def __init__(self, embeddings: Embeddings, *, batch_size: int = 8) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        self._embeddings = embeddings
        self._batch_size = batch_size

    def embed(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        results: list[EmbeddedChunk] = []

        for start in range(0, len(chunks), self._batch_size):
            print(f"Embedding in-progress ({start}/{len(chunks)})")

            batch = chunks[start : start + self._batch_size]
            vectors = self._embeddings.embed_documents(
                [chunk.embedding_text for chunk in batch]
            )
            results.extend(
                EmbeddedChunk(chunk, vector)
                for chunk, vector in zip(batch, vectors, strict=True)
            )

        return results
