import unittest

from langchain_core.embeddings import Embeddings

from booker.chunk.chunk_embedder import ChunkEmbedder
from booker.model.chunk import Chunk


class RecordingEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


class ChunkEmbedderTests(unittest.TestCase):
    def test_embeds_text_with_section_context_in_batches(self) -> None:
        embeddings = RecordingEmbeddings()
        chunks = [_chunk(0), _chunk(1), _chunk(2)]

        results = ChunkEmbedder(embeddings, batch_size=2).embed(chunks)

        self.assertEqual(
            embeddings.calls,
            [
                [chunks[0].embedding_text, chunks[1].embedding_text],
                [chunks[2].embedding_text],
            ],
        )
        self.assertEqual([result.chunk for result in results], chunks)
        self.assertEqual(
            [result.vector for result in results],
            [[float(len(chunk.embedding_text))] for chunk in chunks],
        )

    def test_empty_input_does_not_call_model(self) -> None:
        embeddings = RecordingEmbeddings()

        self.assertEqual(ChunkEmbedder(embeddings).embed([]), [])
        self.assertEqual(embeddings.calls, [])

    def test_rejects_invalid_batch_size(self) -> None:
        with self.assertRaises(ValueError):
            ChunkEmbedder(RecordingEmbeddings(), batch_size=0)


def _chunk(index: int) -> Chunk:
    return Chunk(
        segment_index=0,
        chunk_index=index,
        path=("Part", "Chapter"),
        level=2,
        pages=(index,),
        text=f"Content {index}",
        word_count=2,
    )


if __name__ == "__main__":
    unittest.main()
