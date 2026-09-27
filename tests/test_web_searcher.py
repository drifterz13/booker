import unittest
from unittest.mock import Mock

from langchain_core.messages import AIMessage

from booker.ai.web_searcher import WebSearcher


class WebSearcherTests(unittest.TestCase):
    def test_uses_xai_web_search(self) -> None:
        model = Mock()
        model.bind_tools.return_value.invoke.return_value = AIMessage(
            content="A result: https://example.com"
        )

        result = WebSearcher(model).search("current practice")

        self.assertEqual(result, "A result: https://example.com")
        model.bind_tools.assert_called_once_with([{"type": "web_search"}])
        self.assertIn(
            "current practice", model.bind_tools.return_value.invoke.call_args.args[0]
        )
