from pathlib import Path

import chainlit as cl
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

from booker.ai.model.xai import create_xai_model
from booker.application import Application, Session
from booker.store.chroma import ChromaStore

DATA_DIR = Path(".store")

load_dotenv()


@cl.on_chat_start
async def on_chat_start() -> None:
    files = await cl.AskFileMessage(
        content="Upload a book (.pdf) to begin.",
        accept=["application/pdf"],
        max_size_mb=100,
        max_files=1,
        timeout=300,
    ).send()
    if not files:
        await cl.Message(
            content="Upload timed out. Start a new chat to try again."
        ).send()
        return

    uploaded = files[0]
    if Path(uploaded.name).suffix.lower() != ".pdf":
        await cl.Message(content="Please upload a PDF file.").send()
        return

    status = await cl.Message(content=f"Indexing {uploaded.name}...").send()

    try:
        application = Application(
            books_dir=DATA_DIR / "books",
            store=ChromaStore(DATA_DIR / "chroma", collection_name="books"),
            embeddings=OllamaEmbeddings(model="bge-m3"),
            model=create_xai_model(),
        )
        session = await cl.make_async(application.start)(Path(uploaded.path))
    except Exception as exc:  # noqa: BLE001 - report indexing failures in the UI
        status.content = f"Could not index {uploaded.name}: {exc}"
        await status.update()
        return

    cl.user_session.set("session", session)
    status.content = (
        f"Indexed {uploaded.name} ({session.chunk_count} chunks). Ask about the book!"
    )
    await status.update()
    await cl.Pdf(path=str(session.source), name=uploaded.name, display="side").send(
        for_id=status.id
    )


@cl.on_message
async def on_message(message: cl.Message) -> None:
    session: Session | None = cl.user_session.get("session")
    if session is None:
        await cl.Message(content="Upload a book in a new chat first.").send()
        return

    answer = await cl.Message(content="Working on your answer...").send()
    started = False
    try:
        async for token in session.stream(message.content):
            if not started:
                answer.content = ""
                await answer.update()
                started = True
            await answer.stream_token(token)
    except Exception as exc:  # noqa: BLE001 - report agent failures in the UI
        answer.content = f"Could not answer: {exc}"
        await answer.update()
        return

    if not started:
        answer.content = "No answer was generated."
    await answer.update()
