import pymupdf

from booker.model.book import Book, Position
from booker.model.content import ContentSegment
from booker.utils.outline_extractor import BookExtractionError


class ContentExtractor:
    """Extract non-overlapping text segments from a book outline."""

    def extract(self, book: Book) -> list[ContentSegment]:
        entries = list(book.walk_with_path())
        if not entries:
            return []

        with pymupdf.open(book.source) as pdf:
            document_end = Position(page=pdf.page_count, y=0.0)
            segments: list[ContentSegment] = []

            for index, (section, path) in enumerate(entries):
                end = (
                    entries[index + 1][0].start
                    if index + 1 < len(entries)
                    else document_end
                )
                text = self._extract_span(pdf, section.start, end)
                segments.append(
                    ContentSegment(
                        title=section.title,
                        level=section.level,
                        path=path,
                        start=section.start,
                        end=end,
                        text=text,
                    )
                )

            return segments

    @staticmethod
    def _extract_span(
        pdf: pymupdf.Document,
        start: Position,
        end: Position,
    ) -> str:
        if end < start:
            raise BookExtractionError(
                f"Content end {end!r} appears before start {start!r}"
            )

        pieces: list[str] = []
        last_page = min(end.page, pdf.page_count - 1)

        for page_index in range(start.page, last_page + 1):
            page = pdf[page_index]
            top = start.y if page_index == start.page else page.rect.y0
            bottom = end.y if page_index == end.page else page.rect.y1

            top = min(max(top, page.rect.y0), page.rect.y1)
            bottom = min(max(bottom, page.rect.y0), page.rect.y1)
            if bottom <= top:
                continue

            clip = pymupdf.Rect(page.rect.x0, top, page.rect.x1, bottom)
            text = page.get_text("text", clip=clip, sort=True).strip()
            if text:
                pieces.append(text)

        return "\n\n".join(pieces)
