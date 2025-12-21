import base64
import io
from pathlib import Path
from typing import Iterable, List, Optional

from PIL import Image
from together import Together

from .base_parser import BaseParser, ParseResult

DEFAULT_PROMPT = """Convert the following scientific article page into polished Markdown:
- Use # / ## headings that mirror the document structure.
- Preserve tables using Markdown table syntax.
- Keep bullet or numbered lists intact.
- Retain mathematical expressions, figure references, and inline citations.
- Respond with Markdown only."""


def _encode_image_to_data_url(image) -> str:
    """Serialize a PIL image to a data URL for Together's image_url payload."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


class QwenParser(BaseParser):
    """Call Together's Qwen2.5-VL 72B model on page images to produce Markdown."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct",
        max_tokens: Optional[int] = None,
    ):
        if not api_key:
            raise ValueError("Together API key is required for QwenParser.")
        self.client = Together(api_key=api_key)
        self.model_name = model_name
        self.max_tokens = max_tokens

    def parse_page(
        self,
        page_image,
        prompt: Optional[str] = None,
        logprobs: Optional[int] = None,
        return_logprobs: bool = False,
    ):
        """Upload a page image to Qwen2.5-VL via chat.completions and return Markdown."""
        data_url = _encode_image_to_data_url(page_image)
        prompt_text = prompt or DEFAULT_PROMPT

        request_kwargs = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": 0.0,
        }
        if self.max_tokens is not None:
            request_kwargs["max_tokens"] = self.max_tokens
        if logprobs is not None:
            request_kwargs["logprobs"] = logprobs

        response = self.client.chat.completions.create(
            **request_kwargs,
        )
        message = response.choices[0].message
        content = (
            message.content
            if isinstance(message.content, str)
            else str(message.content)
        )
        if return_logprobs:
            logprobs = getattr(response.choices[0], "logprobs", None)
            return content, logprobs
        return content

    def _load_images_from_dir(self, image_dir: Path) -> List[Image.Image]:
        image_paths = sorted(
            [
                *image_dir.glob("*.png"),
                *image_dir.glob("*.jpg"),
                *image_dir.glob("*.jpeg"),
                *image_dir.glob("*.webp"),
            ]
        )
        if not image_paths:
            raise ValueError(f"No images found in {image_dir}")

        images: List[Image.Image] = []
        for image_path in image_paths:
            with Image.open(image_path) as img:
                images.append(img.convert("RGB"))
        return images

    def _load_images(self, source: str | Iterable[str]) -> List[Image.Image]:
        if isinstance(source, (list, tuple)):
            images: List[Image.Image] = []
            for image_path in source:
                with Image.open(image_path) as img:
                    images.append(img.convert("RGB"))
            if not images:
                raise ValueError("No images provided in the list of paths.")
            return images

        path = Path(source)
        if path.is_dir():
            return self._load_images_from_dir(path)

        if path.suffix.lower() == ".pdf":
            raise ValueError(
                f"Expected an image directory or list of image paths, not a PDF: {path}"
            )

        if path.is_file():
            with Image.open(path) as img:
                return [img.convert("RGB")]

        raise FileNotFoundError(f"Image path not found: {path}")

    def parse(self, image_source: str | Iterable[str]) -> ParseResult:
        """Parse images into Markdown by processing each page image with Qwen VL."""
        images = self._load_images(image_source)

        page_markdown = []
        for img in images:
            page_text = self.parse_page(img)
            if page_text:
                page_markdown.append(page_text.strip())

        content = "\n\n".join(page_markdown)
        return ParseResult(
            content=content,
            metadata={"parser": "qwen", "model": self.model_name},
        )
