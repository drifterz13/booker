import os
import unittest
from pathlib import Path
from unittest.mock import patch

import chromadb

from booker.config.env import EnvConfig
from booker.model.chunk import Chunk
from booker.model.embedded_chunk import EmbeddedChunk
from booker.store.chroma import ChromaStore


class ChromaStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        client = chromadb.EphemeralClient()
        cloud_client = patch(
            "booker.store.chroma.chromadb.CloudClient", return_value=client
        )
        self.cloud_client = cloud_client.start()
        self.addCleanup(cloud_client.stop)
        with patch.dict(
            os.environ,
            {
                "CHROMA_API_KEY": "test-key",
                "CHROMA_TENANT": "test-tenant",
                "CHROMA_DATABASE": "test-database",
            },
        ):
            self.config = EnvConfig()

    def test_returns_source_with_page_metadata(self) -> None:
        source = Path("book.pdf")
        store = ChromaStore(config=self.config)
        self.addCleanup(store.close)
        store.sync(source, [_item(0, (0, 1), [1.0, 0.0])])

        hits = store.search([1.0, 0.0], source=source)

        self.cloud_client.assert_called_once_with(
            api_key="test-key", tenant="test-tenant", database="test-database"
        )
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].source, source.resolve())
        self.assertEqual(hits[0].path, ("Part", "Chapter"))
        self.assertEqual(hits[0].pages, (0, 1))
        self.assertEqual((hits[0].start_page, hits[0].end_page), (1, 2))
        self.assertEqual(hits[0].text, "Content 0")

    def test_sync_removes_stale_chunks_without_touching_other_sources(self) -> None:
        store = ChromaStore(config=self.config)
        self.addCleanup(store.close)
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

    def test_chat_collections_are_isolated(self) -> None:
        source = Path("book.pdf")
        first = ChromaStore(config=self.config)
        second = ChromaStore(config=self.config)
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        first.sync(source, [_item(0, (0,), [1.0, 0.0], "First chat")])
        second.sync(source, [_item(0, (0,), [1.0, 0.0], "Second chat")])

        self.assertEqual(first.search([1.0, 0.0], source=source)[0].text, "First chat")
        self.assertEqual(
            second.search([1.0, 0.0], source=source)[0].text, "Second chat"
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
