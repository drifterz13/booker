from dataclasses import dataclass

from booker.model.chunk import Chunk


@dataclass(slots=True)
class EmbeddedChunk:
    """A chunk and its dense vector, ready for a vector store."""

    chunk: Chunk
    vector: list[float]
