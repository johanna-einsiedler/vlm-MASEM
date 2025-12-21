"""Minimal base parser definitions shared across document parsers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ParseResult:
    content: str
    metadata: Dict[str, Any]


class BaseParser:
    """Base class for document parsers."""

    def parse(self, file_path: str) -> ParseResult:  # pragma: no cover - interface stub
        raise NotImplementedError
