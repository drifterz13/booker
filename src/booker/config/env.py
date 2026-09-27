import os


class EnvConfig:
    def __init__(self) -> None:
        xai_api_key = os.getenv("XAI_API_KEY")
        if not xai_api_key:
            raise ValueError("Missing required environment variable: 'XAI_API_KEY'")

        self._xai_api_key = xai_api_key

    @property
    def xai_api_key(self) -> str:
        return self._xai_api_key
