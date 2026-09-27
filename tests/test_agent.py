import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from booker.ai.agent import BookAgent, build_book_agent
from booker.model.search_hit import SearchHit


class BookAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_runs_with_supplied_tools(self) -> None:
        @tool
        def search_book(query: str) -> str:
            """Find a passage in the selected book."""
            return f"Book passage for {query}"

        model = "xai:grok-4.3"
        with patch("booker.ai.agent.create_agent") as create_agent:
            create_agent.return_value.ainvoke = AsyncMock(
                return_value={"messages": [AIMessage(content="Answer, page 3")]}
            )
            agent = BookAgent(model=model, tools=[search_book])
            answer = await agent.run("What does the book say?")

        self.assertEqual(answer, "Answer, page 3")
        self.assertEqual(create_agent.call_args.kwargs["model"], model)
        self.assertEqual(create_agent.call_args.kwargs["tools"], [search_book])
        self.assertIn("Cite PDF pages", create_agent.call_args.kwargs["system_prompt"])
        create_agent.return_value.ainvoke.assert_awaited_once_with(
            {"messages": [{"role": "user", "content": "What does the book say?"}]}
        )

    def test_builder_binds_book_and_service_tools(self) -> None:
        source = Path("book.pdf")
        store = Mock()
        store.search.return_value = [
            SearchHit(
                source=source,
                path=("Part", "Chapter"),
                pages=(0, 1),
                text="A relevant passage",
                distance=0.1,
            )
        ]
        embeddings = Mock()
        embeddings.embed_query.return_value = [1.0, 0.0]
        summarizer = Mock()
        summarizer.summarize.return_value = "Book summary"
        web_searcher = Mock()
        web_searcher.search.return_value = "News: https://example.com"

        with patch("booker.ai.agent.BookAgent") as agent_class:
            result = build_book_agent(
                model="xai:grok-4.3",
                source=source,
                store=store,
                embeddings=embeddings,
                summarizer=summarizer,
                web_searcher=web_searcher,
            )

        self.assertIs(result, agent_class.return_value)
        tools = {item.name: item for item in agent_class.call_args.kwargs["tools"]}
        self.assertIn(
            "A relevant passage", tools["search_book"].invoke({"query": "pooling"})
        )
        self.assertEqual(tools["summarize_book"].invoke({}), "Book summary")
        self.assertEqual(
            tools["search_web"].invoke({"query": "latest news"}),
            "News: https://example.com",
        )
        store.search.assert_called_once_with([1.0, 0.0], source=source, limit=5)
        summarizer.summarize.assert_called_once_with(source)
        web_searcher.search.assert_called_once_with("latest news")
