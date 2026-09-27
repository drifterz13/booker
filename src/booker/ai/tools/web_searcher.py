from langchain_core.language_models import BaseChatModel


class WebSearcher:
    """Use xAI's built-in web search through a chat model."""

    def __init__(self, model: BaseChatModel) -> None:
        self._model = model.bind_tools([{"type": "web_search"}])

    def search(self, query: str) -> str:
        return self._model.invoke(
            f"Search the web for: {query}\nInclude source URLs in your answer."
        ).text
