import unittest

from booker.chunk.chunker import Chunker
from booker.model.book import Position
from booker.model.content import ContentSegment, PageFragment


class ChunkerTests(unittest.TestCase):
    def test_chunk_can_span_pages_and_preserves_page_provenance(self) -> None:
        segment = _segment(
            PageFragment(0, "one two three four"),
            PageFragment(1, "five six seven eight"),
        )

        chunks = Chunker(chunk_size=40, chunk_overlap=5).chunk([segment])

        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            chunks[0].text,
            "one two three four\n\nfive six seven eight",
        )
        self.assertEqual(chunks[0].pages, (0, 1))
        self.assertEqual((chunks[0].start_page, chunks[0].end_page), (1, 2))

    def test_prefers_paragraph_boundaries(self) -> None:
        segment = _segment(
            PageFragment(0, "Alpha one two.\n\nBeta three four.\n\nGamma five six.")
        )

        chunks = Chunker(chunk_size=30, chunk_overlap=5).chunk([segment])

        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["Alpha one two.", "Beta three four.", "Gamma five six."],
        )

    def test_repeated_text_maps_to_its_actual_page(self) -> None:
        segment = _segment(
            PageFragment(0, "same same same"),
            PageFragment(1, "same same same"),
        )

        chunks = Chunker(chunk_size=16, chunk_overlap=0).chunk([segment])

        self.assertEqual([chunk.pages for chunk in chunks], [(0,), (1,)])

    def test_does_not_cross_segment_boundaries(self) -> None:
        first = _segment(PageFragment(0, "one two"), title="First")
        second = _segment(PageFragment(1, "three four"), title="Second")

        chunks = Chunker(chunk_size=50, chunk_overlap=0).chunk([first, second])

        self.assertEqual([chunk.text for chunk in chunks], ["one two", "three four"])
        self.assertEqual([chunk.segment_index for chunk in chunks], [0, 1])
        self.assertEqual([chunk.chunk_index for chunk in chunks], [0, 0])

    def test_adds_hierarchy_to_embedding_text(self) -> None:
        segment = _segment(PageFragment(2, "chapter content"))

        chunk = Chunker(chunk_size=50, chunk_overlap=0).chunk([segment])[0]

        self.assertEqual(chunk.path, ("Part", "Chapter"))
        self.assertEqual(
            chunk.embedding_text,
            "Part > Chapter\n\nchapter content",
        )
        self.assertEqual((chunk.start_page, chunk.end_page), (3, 3))

    def test_skips_empty_segments(self) -> None:
        segment = _segment()

        self.assertEqual(Chunker().chunk([segment]), [])

    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            Chunker(chunk_size=0)
        with self.assertRaises(ValueError):
            Chunker(chunk_size=10, chunk_overlap=11)


def _segment(
    *fragments: PageFragment,
    title: str = "Chapter",
) -> ContentSegment:
    return ContentSegment(
        title=title,
        level=2,
        path=("Part", title),
        start=Position(0),
        end=Position(3),
        fragments=fragments,
    )


if __name__ == "__main__":
    unittest.main()
