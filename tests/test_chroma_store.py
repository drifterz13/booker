import tempfile
import unittest
from pathlib import Path

from booker.model.chunk import Chunk
from booker.model.embedded_chunk import EmbeddedChunk
from booker.store.chroma import ChromaStore


class ChromaStoreTests(unittest.TestCase):
    def test_persists_and_reopens_source_with_page_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path("book.pdf")
            store = ChromaStore(Path(directory), collection_name="booker_chunks")
            store.sync(source, [_item(0, (0, 1), [1.0, 0.0])])

            reopened = ChromaStore(Path(directory), collection_name="booker_chunks")
            hits = reopened.search([1.0, 0.0], source=source)

            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0].source, source.resolve())
            self.assertEqual(hits[0].path, ("Part", "Chapter"))
            self.assertEqual(hits[0].pages, (0, 1))
            self.assertEqual((hits[0].start_page, hits[0].end_page), (1, 2))
            self.assertEqual(hits[0].text, "Content 0")

    def test_sync_removes_stale_chunks_without_touching_other_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ChromaStore(Path(directory), collection_name="booker_chunks")
            source = Path("first.pdf")
            other = Path("second.pdf")
            store.sync(
                source,
                [_item(0, (0,), [1.0, 0.0]), _item(1, (1,), [0.0, 1.0])],
            )
            store.sync(other, [_item(0, (2,), [0.0, 1.0])])

            store.sync(source, [_item(0, (0,), [1.0, 0.0], "Revised")])

            hits = store.search([1.0, 0.0], source=source)
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0].text, "Revised")
            self.assertEqual(len(store.search([0.0, 1.0], source=other)), 1)

            store.sync(source, [])
            self.assertEqual(store.search([1.0, 0.0], source=source), [])
            self.assertEqual(len(store.search([0.0, 1.0], source=other)), 1)

    def test_rejects_different_model_for_existing_collection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ChromaStore(
                Path(directory), model_name="bge-m3", collection_name="booker_chunks"
            )

            with self.assertRaises(ValueError):
                ChromaStore(
                    Path(directory),
                    model_name="another-model",
                    collection_name="booker_chunks",
                )


def _item(
    index: int,
    pages: tuple[int, ...],
    vector: list[float],
    text: str | None = None,
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=Chunk(
            segment_index=0,
            chunk_index=index,
            path=("Part", "Chapter"),
            level=2,
            pages=pages,
            text=text or f"Content {index}",
            word_count=2,
        ),
        vector=vector,
    )


if __name__ == "__main__":
    unittest.main()
