# Booker

Extract a PDF's bookmark hierarchy and turn it into non-overlapping content
segments. Outline levels remain structural data instead of being interpreted as
fixed part, chapter, or section types.

```python
from pathlib import Path

from booker.chunk.chunk_embedder import ChunkEmbedder
from booker.chunk.chunker import Chunker
from booker.extractor.content import ContentExtractor
from booker.extractor.outline import OutlineExtractor
from booker.store.chroma import ChromaStore
from langchain_ollama import OllamaEmbeddings


source = Path("book.pdf")
book = OutlineExtractor(src=source).extract()

for section in book.walk():
    print(
        section.level,
        section.title,
        section.start_page,
        section.end_page,
    )

segments = ContentExtractor().extract(book)
chunks = Chunker(chunk_size=2000, chunk_overlap=200).chunk(segments)
embeddings = OllamaEmbeddings(model="bge-m3")
embedded = ChunkEmbedder(embeddings).embed(chunks)
store = ChromaStore(Path(".booker/chroma"), collection_name="booker_chunks")
store.sync(source, embedded)

for segment in segments:
    print(segment.path, segment.start_page, segment.end_page, segment.text[:100])
    print([fragment.page_number for fragment in segment.fragments])

for chunk in chunks:
    print(chunk.path, chunk.start_page, chunk.end_page, chunk.word_count)

for item in embedded:
    print(item.chunk.path, item.chunk.pages, len(item.vector))

for hit in store.search(embeddings.embed_query("What is this book about?")):
    print(hit.path, hit.start_page, hit.end_page, hit.text[:100])
```

Internally, section boundaries are half-open PDF positions: `start` is included
and `end` is excluded. Positions use zero-based physical page indexes, while the
`start_page` and `end_page` convenience properties return one-based page numbers.

Outline depth is unrestricted. Hierarchical section ranges may overlap their
descendants, but content segments never overlap: each segment runs from its TOC
entry to the immediately following entry and carries its complete ancestor path.
Segment text is retained as page-level fragments so future chunks can preserve
their physical PDF page provenance.

Chunking uses LangChain's recursive character splitter, preferring paragraph
and line boundaries. Sizes are measured in characters by default. Chunks can
span PDF pages but never cross content-segment boundaries; their page metadata
is mapped from the original page fragments. `chunk.embedding_text` prefixes
the chunk with its hierarchy path.

Embedding uses LangChain's Ollama integration and batches requests to avoid
sending an entire book at once. Run `ollama pull bge-m3` and start Ollama before
running the embedding example. `EmbeddedChunk` retains each chunk's text, path,
and page references alongside its dense vector. Chroma persists those vectors,
the original chunk text, and its section/page metadata under `.booker/chroma`.
Re-indexing the same source updates its chunks and removes stale ones. Query
vectors must come from the same embedding model. Ollama's embedding endpoint
provides BGE-M3's dense vectors, not its sparse or multi-vector outputs.

Run the tests with:

```shell
uv run python -m unittest discover -s tests -v
```

Run the lint and formatting checks with:

```shell
uv run ruff check src tests
uv run ruff format --check src tests
```
