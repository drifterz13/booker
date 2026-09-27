from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A stored passage returned by vector search."""

    source: Path
    path: tuple[str, ...]
    pages: tuple[int, ...]
    text: str
    distance: float

    @property
    def start_page(self) -> int:
        return self.pages[0] + 1

    @property
    def end_page(self) -> int:
        return self.pages[-1] + 1
