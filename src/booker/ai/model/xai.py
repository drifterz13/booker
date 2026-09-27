from langchain_core.language_models import BaseChatModel
from langchain_xai import ChatXAI


def create_xai_model(
    *,
    model_name: str = "grok-4.3",
    temperature: float = 0,
) -> BaseChatModel:
    return ChatXAI(
        model=model_name,
        temperature=temperature,
    )
