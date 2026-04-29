from __future__ import annotations

import enum


class Format(enum.Enum):
    GEMMA3 = "gemma3"
    GEMMA4 = "gemma4"

    def from_gemma4(self, text: str) -> str:
        return text


class Conversation:  # minimal compatibility shim
    pass

