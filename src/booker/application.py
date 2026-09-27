from collections.abc import AsyncIterator
from hashlib import file_digest
from pathlib import Path
from shutil import copyfile

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from booker.ai.agent import BookAgent, build_book_agent
from booker.ai.tools.summarizer import BookSummarizer
from booker.ai.tools.web_searcher import WebSearcher
from booker.ingest.indexer import index_book
from booker.store.chroma import ChromaStore


class Session:
    """One selected book and its in-memory conversation."""

    def __init__(self, source: Path, chunk_count: int, agent: BookAgent) -> None:
        self.source = source
        self.chunk_count = chunk_count
        self._agent = agent
        self._messages: list[dict[str, str]] = []

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        messages = [*self._messages, {"role": "user", "content": prompt}]
        parts: list[str] = []
        async for token in self._agent.stream(messages):
            parts.append(token)
            yield token

        if parts:
            self._messages.extend(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "".join(parts)},
                ]
            )


class Application:
    """Index uploaded PDFs and create book-bound chat sessions."""

    def __init__(
        self,
        *,
        books_dir: Path,
        store: ChromaStore,
        embeddings: Embeddings,
        model: BaseChatModel,
    ) -> None:
        self._books_dir = books_dir.resolve()
        self._store = store
        self._embeddings = embeddings
        self._model = model

    def start(self, upload: Path) -> Session:
        source = self._save_pdf(upload)
        chunk_count = index_book(
            source,
            store=self._store,
            embeddings=self._embeddings,
        )
        agent = build_book_agent(
            model=self._model,
            source=source,
            store=self._store,
            embeddings=self._embeddings,
            summarizer=BookSummarizer(self._model),
            web_searcher=WebSearcher(self._model),
        )
        return Session(source, chunk_count, agent)

    def _save_pdf(self, upload: Path) -> Path:
        with upload.open("rb") as file:
            digest = file_digest(file, "sha256").hexdigest()

        self._books_dir.mkdir(parents=True, exist_ok=True)
        source = self._books_dir / f"{digest}.pdf"
        if not source.exists():
            copyfile(upload, source)
        return source
