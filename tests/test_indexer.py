import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pymupdf

from booker.ingest.indexer import index_book
from booker.store.chroma import ChromaStore


class IndexBookTests(unittest.TestCase):
    def test_indexes_a_pdf_into_chroma(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "book.pdf"
            with pymupdf.open() as pdf:
                page = pdf.new_page()
                page.insert_text((72, 72), "A passage about book indexing.")
                pdf.set_toc([[1, "Introduction", 1]])
                pdf.save(source)

            embeddings = Mock()
            embeddings.embed_documents.side_effect = lambda texts: [
                [1.0, 0.0] for _ in texts
            ]
            store = ChromaStore(Path(directory) / "chroma", collection_name="books")
            count = index_book(source, store=store, embeddings=embeddings)
            hits = store.search([1.0, 0.0], source=source)

            self.assertGreater(count, 0)
            self.assertIn("book indexing", hits[0].text)
            self.assertEqual(hits[0].start_page, 1)

    def test_indexes_chunks_for_an_outlined_pdf(self) -> None:
        source = Path("book.pdf")
        store = Mock()
        embeddings = Mock()
        book = Mock(sections=[Mock()])
        segments = [Mock()]
        chunks = [Mock(), Mock()]
        embedded = [Mock(), Mock()]

        with (
            patch("booker.ingest.indexer.OutlineExtractor") as outline,
            patch("booker.ingest.indexer.ContentExtractor") as content,
            patch("booker.ingest.indexer.Chunker") as chunker,
            patch("booker.ingest.indexer.ChunkEmbedder") as embedder,
        ):
            outline.return_value.extract.return_value = book
            content.return_value.extract.return_value = segments
            chunker.return_value.chunk.return_value = chunks
            embedder.return_value.embed.return_value = embedded
            count = index_book(source, store=store, embeddings=embeddings)

        self.assertEqual(count, 2)
        outline.assert_called_once_with(src=source)
        content.return_value.extract.assert_called_once_with(book)
        chunker.return_value.chunk.assert_called_once_with(segments)
        embedder.assert_called_once_with(embeddings)
        embedder.return_value.embed.assert_called_once_with(chunks)
        store.sync.assert_called_once_with(source, embedded)

    def test_rejects_pdf_without_bookmarks(self) -> None:
        with patch("booker.ingest.indexer.OutlineExtractor") as outline:
            outline.return_value.extract.return_value = Mock(sections=[])
            with self.assertRaisesRegex(ValueError, "no bookmarks"):
                index_book(Path("book.pdf"), store=Mock(), embeddings=Mock())

    def test_rejects_pdf_without_extractable_text(self) -> None:
        store = Mock()
        with (
            patch("booker.ingest.indexer.OutlineExtractor") as outline,
            patch("booker.ingest.indexer.ContentExtractor") as content,
            patch("booker.ingest.indexer.Chunker") as chunker,
        ):
            outline.return_value.extract.return_value = Mock(sections=[Mock()])
            content.return_value.extract.return_value = []
            chunker.return_value.chunk.return_value = []
            with self.assertRaisesRegex(ValueError, "no extractable section text"):
                index_book(Path("book.pdf"), store=store, embeddings=Mock())

        store.sync.assert_not_called()
