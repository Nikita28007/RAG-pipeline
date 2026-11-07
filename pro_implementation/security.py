"""Helpers that treat retrieved documents as untrusted input."""

import re

INJECTION_PATTERNS = {
    "ignore_instructions": re.compile(
        r"\b(ignore|disregard|forget)\b.{0,40}\b(previous|prior|above|earlier|all)\b"
        r".{0,40}\b(instructions?|prompts?|rules?)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "reveal_prompt": re.compile(
        r"\b(reveal|print|show|repeat)\b.{0,40}\b(system|hidden)\s+(prompt|instructions?)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "role_override": re.compile(
        r"\byou are now\b|\bnew instructions?\s*:", re.IGNORECASE
    ),
    "chat_markers": re.compile(
        r"<\|?(system|im_start|im_end)\|?>|\[/?INST\]|^\s*(system|assistant)\s*:",
        re.IGNORECASE | re.MULTILINE,
    ),
}

_DELIMITER_TAG = re.compile(r"<(/?)\s*(document|context)\b", re.IGNORECASE)


def find_injection_patterns(text: str) -> list[str]:
    """Names of the heuristic injection patterns that match the text."""
    return [name for name, pattern in INJECTION_PATTERNS.items() if pattern.search(text)]


def neutralize_markup(text: str) -> str:
    """Stop a document from closing or forging the <document> delimiters."""
    return _DELIMITER_TAG.sub(lambda m: "&lt;" + m.group(1) + m.group(2), text)