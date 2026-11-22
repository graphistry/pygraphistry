"""Shared text contracts for Cypher exact integer parsing."""

from __future__ import annotations

import re
import sys
import unicodedata

ASCII_EXACT_INT64_PATTERN = r"^[+-]?[0-9]+$"
UNICODE_EXACT_INT64_PATTERN = r"^[+-]?\d+$"

ASCII_EXACT_INT64_RE = re.compile(ASCII_EXACT_INT64_PATTERN)
UNICODE_EXACT_INT64_RE = re.compile(UNICODE_EXACT_INT64_PATTERN)

_DECIMAL_DIGIT_REPLACEMENTS = {
    character: str(unicodedata.decimal(character))
    for codepoint in range(sys.maxunicode + 1)
    if unicodedata.category(character := chr(codepoint)) == "Nd"
}

PANDAS_DIGIT_TRANSLATION = str.maketrans(_DECIMAL_DIGIT_REPLACEMENTS)

POLARS_DIGIT_REPLACEMENTS = _DECIMAL_DIGIT_REPLACEMENTS


def normalize_supported_decimal_digits(text: str) -> str:
    """Normalize inherited Java-compatible decimal-digit forms to ASCII."""
    return text.translate(PANDAS_DIGIT_TRANSLATION)
