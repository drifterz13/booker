from dataclasses import dataclass

from booker.model.book import Position


@dataclass(frozen=True, slots=True)
class ContentSegment:
    """Non-overlapping text owned by one outline entry."""

    title: str
    level: int
    path: tuple[str, ...]
    start: Position
    end: Position
    text: str

    @property
    def start_page(self) -> int:
        return self.start.page_number

    @property
    def end_page(self) -> int:
        """Return the last one-based physical page containing segment text."""

        if self.end.page > self.start.page and self.end.y <= 0:
            return self.end.page
        return self.end.page_number
