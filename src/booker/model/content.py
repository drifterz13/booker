from dataclasses import dataclass

from booker.model.book import Position


@dataclass(frozen=True, slots=True)
class PageFragment:
    """Text extracted from one zero-based physical PDF page."""

    page: int
    text: str

    @property
    def page_number(self) -> int:
        return self.page + 1


@dataclass(frozen=True, slots=True)
class ContentSegment:
    """Non-overlapping text owned by one outline entry."""

    title: str
    level: int
    path: tuple[str, ...]
    start: Position
    end: Position
    fragments: tuple[PageFragment, ...]

    @property
    def text(self) -> str:
        return "\n\n".join(fragment.text for fragment in self.fragments)

    @property
    def start_page(self) -> int:
        return self.start.page_number

    @property
    def end_page(self) -> int:
        """Return the last one-based physical page containing segment text."""

        if self.end.page > self.start.page and self.end.y <= 0:
            return self.end.page
        return self.end.page_number
