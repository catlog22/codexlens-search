from __future__ import annotations

import enum
import json
from dataclasses import dataclass


class EntityKind(enum.Enum):
    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    MODULE = "module"


@dataclass(frozen=True)
class EntityId:
    file_path: str
    symbol_name: str
    kind: EntityKind
    start_line: int
    end_line: int

    def to_key(self) -> str:
        return "\t".join(
            (
                self.file_path,
                self.kind.value,
                self.symbol_name,
                str(int(self.start_line)),
                str(int(self.end_line)),
            )
        )

    @staticmethod
    def from_key(key: str) -> "EntityId":
        raw = (key or "").strip()
        if raw.startswith("{"):
            payload = json.loads(raw)
            return EntityId(
                file_path=str(payload.get("file_path", "")),
                symbol_name=str(payload.get("symbol_name", "")),
                kind=EntityKind(str(payload.get("kind", EntityKind.FILE.value))),
                start_line=int(payload.get("start_line", 0) or 0),
                end_line=int(payload.get("end_line", 0) or 0),
            )

        parts = raw.split("\t")
        if len(parts) != 5:
            raise ValueError(f"Invalid EntityId key: {key!r}")
        file_path, kind, symbol_name, start_line, end_line = parts
        return EntityId(
            file_path=file_path,
            symbol_name=symbol_name,
            kind=EntityKind(kind),
            start_line=int(start_line or 0),
            end_line=int(end_line or 0),
        )
