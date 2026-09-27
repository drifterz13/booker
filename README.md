# Booker

Extract a PDF's bookmark hierarchy and turn it into non-overlapping content
segments. Outline levels remain structural data instead of being interpreted as
fixed part, chapter, or section types.

```python
from pathlib import Path

from booker.utils.chunker import Chunker
from booker.utils.content_extractor import ContentExtractor
from booker.utils.outline_extractor import OutlineExtractor


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

for segment in segments:
    print(segment.path, segment.start_page, segment.end_page, segment.text[:100])
    print([fragment.page_number for fragment in segment.fragments])

for chunk in chunks:
    print(chunk.path, chunk.start_page, chunk.end_page, chunk.word_count)
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

Run the tests with:

```shell
python -m unittest discover -s tests -v
```
