"""String manipulation utilities for text processing."""
import re
from typing import Iterator


def camel_to_snake(name: str) -> str:
    """Convert camelCase or PascalCase to snake_case."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def snake_to_camel(name: str, pascal: bool = False) -> str:
    """Convert snake_case to camelCase or PascalCase."""
    parts = name.split("_")
    if pascal:
        return "".join(p.capitalize() for p in parts)
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to max_length, appending suffix if truncated."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def slugify(text: str) -> str:
    """Convert text to URL-friendly slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return re.sub(r"-+", "-", text).strip("-")


def word_wrap(text: str, width: int = 80) -> str:
    """Wrap text at word boundaries."""
    lines = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split():
            if current and len(current) + 1 + len(word) > width:
                lines.append(current)
                current = word
            else:
                current = f"{current} {word}" if current else word
        if current:
            lines.append(current)
    return "\n".join(lines)


def extract_emails(text: str) -> list[str]:
    """Extract email addresses from text."""
    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    return re.findall(pattern, text)


def count_words(text: str) -> dict[str, int]:
    """Count word frequencies in text."""
    words = re.findall(r"\b\w+\b", text.lower())
    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    return counts
