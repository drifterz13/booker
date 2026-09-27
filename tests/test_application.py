import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from booker.application import Application, Session


class ApplicationTests(unittest.TestCase):
    def test_indexes_upload_at_a_stable_content_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            upload = Path(directory) / "upload.pdf"
            upload.write_bytes(b"%PDF-1.7\nexample")
            store = Mock()
            embeddings = Mock()
            model = Mock()
            application = Application(
                books_dir=Path(directory) / "books",
                store=store,
                embeddings=embeddings,
                model=model,
            )

            with (
                patch("booker.application.index_book", return_value=3) as index,
                patch("booker.application.build_book_agent") as build_agent,
            ):
                first = application.start(upload)
                second = application.start(upload)

            self.assertEqual(first.source, second.source)
            self.assertEqual(first.source.read_bytes(), upload.read_bytes())
            self.assertEqual(first.chunk_count, 3)
            index.assert_called_with(first.source, store=store, embeddings=embeddings)
            self.assertEqual(build_agent.call_count, 2)
            self.assertEqual(build_agent.call_args.kwargs["source"], first.source)


class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_keeps_history_between_questions(self) -> None:
        inputs: list[list[dict[str, str]]] = []
        agent = Mock()

        async def stream(messages: list[dict[str, str]]):
            inputs.append(messages)
            yield "Book answer"

        agent.stream.side_effect = stream
        session = Session(Path("book.pdf"), 3, agent)

        first = "".join([part async for part in session.stream("First question")])
        second = "".join([part async for part in session.stream("Follow-up")])

        self.assertEqual(first, "Book answer")
        self.assertEqual(second, "Book answer")
        self.assertEqual(
            inputs[1],
            [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "Book answer"},
                {"role": "user", "content": "Follow-up"},
            ],
        )
