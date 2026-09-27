from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

from booker.chunk.chunk_embedder import ChunkEmbedder
from booker.chunk.chunker import Chunker
from booker.extractor.content import ContentExtractor
from booker.extractor.outline import OutlineExtractor
from booker.store.chroma import ChromaStore


def index_book(source: Path):
    book = OutlineExtractor(src=source).extract()

    for section in book.walk():
        print(
            section.level,
            section.title,
            section.start_page,
            section.end_page,
        )

    segments = ContentExtractor().extract(book)
    chunks = Chunker(chunk_size=2000, chunk_overlap=200).chunk(segments)
    embeddings = OllamaEmbeddings(model="bge-m3")
    embedded = ChunkEmbedder(embeddings).embed(chunks)
    store = ChromaStore(Path(".store/chroma"), collection_name="books")
    store.sync(source, embedded)


def main() -> None:
    print("Loading env...")
    load_dotenv()

    src = Path("WorkingEffectivelyWithLegacyCode.pdf")
    print(f"Indexing book {src}")
    index_book(src)
