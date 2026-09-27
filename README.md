# Booker

Booker is a local app for asking questions about PDF books. Upload a book with
bookmarks, and Booker extracts its section structure, chunks and embeds the
text, and lets you chat with it. The agent can search the book, summarize it,
or search the web for external information.

## Technology

- **PDF processing:** PyMuPDF for bookmarks and page text.
- **Chunking and AI:** LangChain, Ollama with `bge-m3` embeddings, and xAI's
  `grok-4.3` chat model.
- **Storage and UI:** Persistent Chroma for vectors and Chainlit for the chat UI.
- **Development:** Python, uv, unittest, and Ruff.

## Prerequisites

- Python 3.11.4 or newer and [uv](https://docs.astral.sh/uv/).
- [Ollama](https://ollama.com/) running locally with `bge-m3` pulled.
- An xAI API key with access to the configured chat model.
- A PDF with bookmarks and extractable text. Scanned PDFs without OCR and PDFs
  without bookmarks are not supported yet.

## Install

```sh
uv sync --locked
ollama pull bge-m3
cp .env.example .env
```

Set `XAI_API_KEY` in `.env`, then ensure Ollama is running.

## Run

```sh
uv run chainlit run src/booker/chainlit_app.py
```

Open the URL printed by Chainlit and upload a PDF. Indexing may take a while
for a large book. Each chat keeps its conversation in memory. Booker saves PDFs
under `.store/books` and vectors under `.store/chroma`; Chainlit uses `.files/`
for session uploads. A new chat currently requires another upload and re-index.
The saved store is shared, so this prototype is not yet set up for private
multi-user hosting.

## Tests and checks

```sh
uv run python -m unittest discover -s tests -v
uv run ruff check src tests
uv run ruff format --check src tests
```
