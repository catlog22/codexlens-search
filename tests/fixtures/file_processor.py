"""File processing utilities for batch operations."""
import os
import hashlib
from pathlib import Path
from dataclasses import dataclass


@dataclass
class FileInfo:
    """Metadata about a processed file."""
    path: str
    size: int
    checksum: str
    line_count: int
    extension: str


def compute_checksum(filepath: str | Path, algorithm: str = "sha256") -> str:
    """Compute file checksum using the specified hash algorithm."""
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def count_lines(filepath: str | Path) -> int:
    """Count the number of lines in a text file."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)


def get_file_info(filepath: str | Path) -> FileInfo:
    """Gather metadata about a file."""
    p = Path(filepath)
    return FileInfo(
        path=str(p),
        size=p.stat().st_size,
        checksum=compute_checksum(p),
        line_count=count_lines(p),
        extension=p.suffix,
    )


def find_duplicates(directory: str | Path) -> dict[str, list[str]]:
    """Find files with identical content by checksum."""
    checksums: dict[str, list[str]] = {}
    for root, _, files in os.walk(directory):
        for name in files:
            path = os.path.join(root, name)
            try:
                cs = compute_checksum(path)
                checksums.setdefault(cs, []).append(path)
            except (OSError, PermissionError):
                continue
    return {k: v for k, v in checksums.items() if len(v) > 1}


def batch_rename(directory: str | Path, pattern: str, replacement: str) -> list[tuple[str, str]]:
    """Rename files matching a pattern in a directory."""
    import re
    renamed = []
    d = Path(directory)
    for f in d.iterdir():
        if f.is_file() and re.search(pattern, f.name):
            new_name = re.sub(pattern, replacement, f.name)
            new_path = f.parent / new_name
            f.rename(new_path)
            renamed.append((str(f), str(new_path)))
    return renamed
