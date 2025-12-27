from typing import Optional

from together import Together

from .base_parser import BaseParser


class DeepSeekParser(BaseParser):
    """Call LLM model via Together API on text/markdown to produce JSON extractions."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "openai/gpt-oss-120b",
        max_tokens: Optional[int] = None,
    ):
        if not api_key:
            raise ValueError("Together API key is required for DeepSeekParser.")
        self.client = Together(api_key=api_key)
        self.model_name = model_name
        self.max_tokens = max_tokens

    def parse_text(
        self,
        text: str,
        prompt: str,
    ) -> str:
        """Send text content to LLM via Together API and return the response."""
        request_kwargs = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            "temperature": 0.0,
        }
        if self.max_tokens is not None:
            request_kwargs["max_tokens"] = self.max_tokens

        response = self.client.chat.completions.create(**request_kwargs)
        message = response.choices[0].message
        content = (
            message.content
            if isinstance(message.content, str)
            else str(message.content)
        )
        return content
