from pathlib import Path

from langchain_core.embeddings import Embeddings

from booker.chunk.chunk_embedder import ChunkEmbedder
from booker.chunk.chunker import Chunker
from booker.extractor.content import ContentExtractor
from booker.extractor.outline import OutlineExtractor
from booker.store.chroma import ChromaStore


def index_book(
    source: Path,
    *,
    store: ChromaStore,
    embeddings: Embeddings,
) -> int:
    """Index an outlined PDF and return its chunk count."""
    book = OutlineExtractor(src=source).extract()
    if not book.sections:
        raise ValueError("The PDF has no bookmarks to identify its sections")

    segments = ContentExtractor().extract(book)
    chunks = Chunker(chunk_size=2000, chunk_overlap=200).chunk(segments)
    if not chunks:
        raise ValueError("The PDF has no extractable section text")

    embedded = ChunkEmbedder(embeddings).embed(chunks)
    store.sync(source, embedded)
    return len(chunks)
