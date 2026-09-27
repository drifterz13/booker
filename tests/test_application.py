import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from booker.application import Application, Session


class ApplicationTests(unittest.TestCase):
    def test_indexes_each_upload_using_its_path(self) -> None:
        upload = Path("upload.pdf")
        store = Mock()
        embeddings = Mock()
        application = Application(store=store, embeddings=embeddings, model=Mock())

        with (
            patch("booker.application.index_book") as index,
            patch("booker.application.build_book_agent"),
        ):
            first = application.start(upload)
            second = application.start(upload)

        self.assertEqual(first.source, upload)
        self.assertEqual(second.source, upload)
        self.assertEqual(index.call_count, 2)
        for call in index.call_args_list:
            self.assertEqual(call.args, (upload,))
            self.assertEqual(call.kwargs, {"store": store, "embeddings": embeddings})


class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_keeps_history_between_questions(self) -> None:
        inputs: list[list[dict[str, str]]] = []
        agent = Mock()

        async def stream(messages: list[dict[str, str]]):
            inputs.append(messages)
            yield "Book answer"

        agent.stream.side_effect = stream
        session = Session(Path("book.pdf"), agent)

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
