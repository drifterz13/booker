import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import chromadb
import pymupdf

from booker.config.env import EnvConfig
from booker.ingest.indexer import index_book
from booker.store.chroma import ChromaStore


class IndexBookTests(unittest.TestCase):
    def test_indexes_a_pdf_into_chroma(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "book.pdf"
            _create_pdf(source, text="A passage about book indexing.", bookmarks=True)

            embeddings = Mock()
            embeddings.embed_documents.side_effect = lambda texts: [
                [1.0, 0.0] for _ in texts
            ]
            with (
                patch.dict(
                    os.environ,
                    {
                        "CHROMA_API_KEY": "test-key",
                        "CHROMA_TENANT": "test-tenant",
                        "CHROMA_DATABASE": "test-database",
                    },
                ),
                patch(
                    "booker.store.chroma.chromadb.CloudClient",
                    return_value=chromadb.EphemeralClient(),
                ),
            ):
                store = ChromaStore(config=EnvConfig())
            self.addCleanup(store.close)
            count = index_book(source, store=store, embeddings=embeddings)
            hits = store.search([1.0, 0.0], source=source)

            self.assertGreater(count, 0)
            self.assertIn("book indexing", hits[0].text)
            self.assertEqual(hits[0].start_page, 1)

    def test_rejects_pdf_without_bookmarks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "book.pdf"
            _create_pdf(source, text="A passage.", bookmarks=False)
            with self.assertRaisesRegex(ValueError, "no bookmarks"):
                index_book(source, store=Mock(), embeddings=Mock())

    def test_rejects_pdf_without_extractable_text(self) -> None:
        store = Mock()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "book.pdf"
            _create_pdf(source, text=None, bookmarks=True)
            with self.assertRaisesRegex(ValueError, "no extractable section text"):
                index_book(source, store=store, embeddings=Mock())

        store.sync.assert_not_called()


def _create_pdf(source: Path, *, text: str | None, bookmarks: bool) -> None:
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        if text:
            page.insert_text((72, 72), text)
        if bookmarks:
            pdf.set_toc([[1, "Introduction", 1]])
        pdf.save(source)
