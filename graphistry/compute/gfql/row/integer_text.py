"""Shared text contracts for Cypher exact integer parsing."""

from __future__ import annotations

import re

ASCII_EXACT_INT64_PATTERN = r"^[+-]?[0-9]+$"
UNICODE_EXACT_INT64_PATTERN = r"^[+-]?\d+$"

ASCII_EXACT_INT64_RE = re.compile(ASCII_EXACT_INT64_PATTERN)
UNICODE_EXACT_INT64_RE = re.compile(UNICODE_EXACT_INT64_PATTERN)

_SUPPORTED_DECIMAL_DIGITS = (
    "٠١٢٣٤٥٦٧٨٩",
    "০১২৩৪৫৬৭৮৯",
)

PANDAS_DIGIT_TRANSLATION = str.maketrans({
    digit: str(offset)
    for digits in _SUPPORTED_DECIMAL_DIGITS
    for offset, digit in enumerate(digits)
})

POLARS_DIGIT_REPLACEMENTS = {
    digit: str(offset)
    for digits in _SUPPORTED_DECIMAL_DIGITS
    for offset, digit in enumerate(digits)
}


def normalize_supported_decimal_digits(text: str) -> str:
    """Normalize inherited Java-compatible decimal-digit forms to ASCII."""
    return text.translate(PANDAS_DIGIT_TRANSLATION)
