# Booker

Extract a PDF's bookmark hierarchy and turn it into non-overlapping content
segments. Outline levels remain structural data instead of being interpreted as
fixed part, chapter, or section types.

```python
from pathlib import Path

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

for segment in segments:
    print(segment.path, segment.start_page, segment.end_page, segment.text[:100])
```

Internally, section boundaries are half-open PDF positions: `start` is included
and `end` is excluded. Positions use zero-based physical page indexes, while the
`start_page` and `end_page` convenience properties return one-based page numbers.

Outline depth is unrestricted. Hierarchical section ranges may overlap their
descendants, but content segments never overlap: each segment runs from its TOC
entry to the immediately following entry and carries its complete ancestor path.

Run the tests with:

```shell
python -m unittest discover -s tests -v
```
