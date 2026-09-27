import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage

from booker.ai.tools.summarizer import BookSummarizer
from booker.model.book import Book, Position
from booker.model.content import ContentSegment, PageFragment


class BookSummarizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = Path("book.pdf")
        self.book = Book(source=self.source, page_count=2)
        self.segments = [
            ContentSegment(
                title="First",
                level=1,
                path=("First",),
                start=Position(0),
                end=Position(1),
                fragments=(PageFragment(0, "First section text."),),
            ),
            ContentSegment(
                title="Second",
                level=1,
                path=("Second",),
                start=Position(1),
                end=Position(2),
                fragments=(PageFragment(1, "Second section text."),),
            ),
        ]

    def test_summarizes_five_or_fewer_chunks_in_one_call(self) -> None:
        model = Mock()
        model.invoke.return_value = AIMessage(content="Whole book summary.")
        with (
            patch(
                "booker.ai.tools.summarizer.OutlineExtractor.extract",
                return_value=self.book,
            ),
            patch(
                "booker.ai.tools.summarizer.ContentExtractor.extract",
                return_value=self.segments,
            ),
        ):
            result = BookSummarizer(model).summarize(self.source)

        self.assertEqual(result, "Whole book summary.")
        model.invoke.assert_called_once()
        self.assertIn("First section text", model.invoke.call_args.args[0][1][1])
        self.assertIn("Second section text", model.invoke.call_args.args[0][1][1])

    def test_combines_summaries_of_multiple_batches(self) -> None:
        segments = [
            ContentSegment(
                title=f"Section {index}",
                level=1,
                path=(f"Section {index}",),
                start=Position(index),
                end=Position(index + 1),
                fragments=(PageFragment(index, f"Text from section {index}."),),
            )
            for index in range(6)
        ]
        model = Mock()
        model.invoke.side_effect = [
            AIMessage(content="Summary of sections 0–4."),
            AIMessage(content="Summary of section 5."),
            AIMessage(content="Whole book summary."),
        ]
        with (
            patch(
                "booker.ai.tools.summarizer.OutlineExtractor.extract",
                return_value=self.book,
            ),
            patch(
                "booker.ai.tools.summarizer.ContentExtractor.extract",
                return_value=segments,
            ),
        ):
            result = BookSummarizer(model).summarize(self.source)

        self.assertEqual(result, "Whole book summary.")
        self.assertEqual(model.invoke.call_count, 3)
        first_prompt = model.invoke.call_args_list[0].args[0][1][1]
        second_prompt = model.invoke.call_args_list[1].args[0][1][1]
        self.assertIn("Text from section 0.", first_prompt)
        self.assertIn("Text from section 4.", first_prompt)
        self.assertNotIn("Text from section 5.", first_prompt)
        self.assertIn("Text from section 5.", second_prompt)
        self.assertIn("Summary of sections 0–4.", model.invoke.call_args.args[0][1][1])

    def test_returns_message_when_no_text_is_extracted(self) -> None:
        model = Mock()
        with (
            patch(
                "booker.ai.tools.summarizer.OutlineExtractor.extract",
                return_value=self.book,
            ),
            patch(
                "booker.ai.tools.summarizer.ContentExtractor.extract",
                return_value=[],
            ),
        ):
            result = BookSummarizer(model).summarize(self.source)

        self.assertEqual(result, "No extractable text found in the book.")
        model.invoke.assert_not_called()

    def test_rejects_invalid_group_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "group_size must be at least 2"):
            BookSummarizer(Mock(), group_size=1)
