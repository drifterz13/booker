from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf

from booker.extractor.content import ContentExtractor
from booker.extractor.outline import OutlineExtractor
from booker.model.book import Book, BookSection, Position
from booker.model.content import PageFragment


class OutlineExtractorTests(unittest.TestCase):
    def test_builds_tree_and_assigns_same_or_shallower_boundaries(self) -> None:
        pdf = _FakeDocument(
            page_count=5,
            toc=[
                _toc(1, "Part A", 0, 0),
                _toc(2, "Chapter 1", 0, 100),
                _toc(3, "Section 1", 1, 0),
                _toc(3, "Section 2", 1, 300),
                _toc(2, "Chapter 2", 2, 0),
                _toc(1, "Part B", 4, 0),
            ],
        )
        extractor = OutlineExtractor(src=Path("unused.pdf"))

        roots = extractor._build_sections(pdf)  # type: ignore[arg-type]
        book = Book(Path("unused.pdf"), pdf.page_count, roots)
        flat = list(book.walk())

        self.assertEqual([node.title for node in roots], ["Part A", "Part B"])
        self.assertEqual(
            [node.title for node in roots[0].children],
            ["Chapter 1", "Chapter 2"],
        )
        self.assertEqual(
            [node.title for node in roots[0].children[0].children],
            ["Section 1", "Section 2"],
        )
        self.assertEqual(flat[0].end, Position(4, 0))
        self.assertEqual(flat[1].end, Position(2, 0))
        self.assertEqual(flat[2].end, Position(1, 300))
        self.assertEqual(flat[3].end, Position(2, 0))
        self.assertEqual(flat[-1].end, Position(5, 0))

    def test_accepts_deep_outline_and_produces_paths(self) -> None:
        pdf = _FakeDocument(
            page_count=1,
            toc=[
                _toc(1, "Part", 0, 0),
                _toc(2, "Chapter", 0, 100),
                _toc(3, "Section", 0, 200),
                _toc(4, "Topic", 0, 300),
            ],
        )
        extractor = OutlineExtractor(src=Path("unused.pdf"))
        book = Book(
            source=Path("unused.pdf"),
            page_count=pdf.page_count,
            sections=extractor._build_sections(pdf),  # type: ignore[arg-type]
        )

        self.assertEqual(book.max_level, 4)
        self.assertEqual(
            list(book.walk_with_path())[-1][1],
            ("Part", "Chapter", "Section", "Topic"),
        )

    def test_section_children_are_not_shared(self) -> None:
        first = BookSection("First", 1, Position(0))
        second = BookSection("Second", 1, Position(1))

        first.children.append(BookSection("Child", 2, Position(0, 100)))

        self.assertEqual(second.children, [])


class ContentExtractorTests(unittest.TestCase):
    def test_creates_non_overlapping_segments_with_ancestor_paths(self) -> None:
        chapter = BookSection("Chapter", 2, Position(0, 100), Position(1, 0))
        part = BookSection(
            "Part A",
            1,
            Position(0, 0),
            Position(1, 0),
            children=[chapter],
        )
        part_b = BookSection("Part B", 1, Position(1, 0), Position(2, 0))
        book = Book(Path("book.pdf"), 2, [part, part_b])
        pdf = _FakeDocument(page_count=2, toc=[])

        with patch("booker.extractor.content.pymupdf.open", return_value=pdf):
            segments = ContentExtractor().extract(book)

        self.assertEqual(
            [(segment.start, segment.end) for segment in segments],
            [
                (Position(0, 0), Position(0, 100)),
                (Position(0, 100), Position(1, 0)),
                (Position(1, 0), Position(2, 0)),
            ],
        )
        self.assertEqual(segments[1].path, ("Part A", "Chapter"))
        self.assertEqual(
            segments[1].fragments,
            (PageFragment(0, "page=0 top=100.0 bottom=800.0"),),
        )
        self.assertEqual(segments[1].text, "page=0 top=100.0 bottom=800.0")

    def test_extracts_page_fragment_with_same_page_clipping(self) -> None:
        pdf = _FakeDocument(page_count=1, toc=[])

        fragments = ContentExtractor._extract_fragments(
            pdf,  # type: ignore[arg-type]
            Position(page=0, y=100),
            Position(page=0, y=300),
        )

        self.assertEqual(
            fragments,
            (PageFragment(0, "page=0 top=100.0 bottom=300.0"),),
        )
        self.assertEqual(pdf[0].clips, [(100.0, 300.0)])

    def test_preserves_page_for_each_fragment(self) -> None:
        pdf = _FakeDocument(page_count=3, toc=[])

        fragments = ContentExtractor._extract_fragments(
            pdf,  # type: ignore[arg-type]
            Position(page=0, y=100),
            Position(page=2, y=300),
        )

        self.assertEqual([fragment.page for fragment in fragments], [0, 1, 2])
        self.assertEqual([fragment.page_number for fragment in fragments], [1, 2, 3])


class _FakeDocument:
    def __init__(self, *, page_count: int, toc: list[list[object]]) -> None:
        self.page_count = page_count
        self._toc = toc
        self._pages = [_FakePage(index) for index in range(page_count)]

    def __enter__(self) -> _FakeDocument:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def get_toc(self, *, simple: bool) -> list[list[object]]:
        if simple:
            raise AssertionError("Outline extraction requires detailed destinations")
        return self._toc

    def __getitem__(self, page_index: int) -> _FakePage:
        return self._pages[page_index]


class _FakePage:
    def __init__(self, page_index: int) -> None:
        self.page_index = page_index
        self.rect = pymupdf.Rect(0, 0, 600, 800)
        self.clips: list[tuple[float, float]] = []

    def get_text(
        self,
        output: str,
        *,
        clip: pymupdf.Rect,
        sort: bool,
    ) -> str:
        if output != "text" or not sort:
            raise AssertionError("Expected sorted plain-text extraction")
        self.clips.append((clip.y0, clip.y1))
        return f"page={self.page_index} top={clip.y0} bottom={clip.y1}"


def _toc(level: int, title: str, page: int, y: float) -> list[object]:
    return [
        level,
        title,
        page + 1,
        {"page": page, "to": pymupdf.Point(0, y)},
    ]


if __name__ == "__main__":
    unittest.main()
