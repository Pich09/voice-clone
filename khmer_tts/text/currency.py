"""
Khmer currency verbalizer.

Handles USD ($) and Riel (៛) amounts, converting them into spoken Khmer,
e.g.:
    $10        -> ដប់ដុល្លារ
    $10.50     -> ដប់ដុល្លារ ហាសិប សេន
    ៛5000      -> ប្រាំពាន់រៀល
    5000 រៀល   -> ប្រាំពាន់រៀល

This module should run BEFORE the generic number verbalizer
(numbers.verbalize_numbers_in_text) so currency-prefixed numbers are
claimed here first and not double-processed.
"""

import re
from .numbers import number_to_khmer_words, khmer_digits_to_arabic

_DOLLAR_RE = re.compile(
    r"\$\s*(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)"
)
_RIEL_SYMBOL_SUFFIX_RE = re.compile(
    r"(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*៛"
)
_RIEL_SYMBOL_PREFIX_RE = re.compile(
    r"៛\s*(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
)
_RIEL_WORD_RE = re.compile(
    r"(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*រៀល"
)


def _dollar_to_words(match: re.Match) -> str:
    raw = match.group(1).replace(",", "")
    if "." in raw:
        dollars_str, cents_str = raw.split(".")
        dollars = int(dollars_str)
        cents = int(cents_str.ljust(2, "0")[:2])
        dollar_words = number_to_khmer_words(dollars) + "ដុល្លារ"
        if cents:
            cent_words = number_to_khmer_words(cents) + "សេន"
            return f"{dollar_words} {cent_words}"
        return dollar_words
    return number_to_khmer_words(int(raw)) + "ដុល្លារ"


def _riel_to_words(match: re.Match) -> str:
    raw = match.group(1).replace(",", "")
    value = float(raw) if "." in raw else int(raw)
    return number_to_khmer_words(value) + "រៀល"


def verbalize_currency_in_text(text: str) -> str:
    """Replace $-prefixed and ៛/រៀល-suffixed amounts with spoken Khmer."""
    text = khmer_digits_to_arabic_preserving(text)
    text = _DOLLAR_RE.sub(_dollar_to_words, text)
    text = _RIEL_SYMBOL_PREFIX_RE.sub(_riel_to_words, text)
    text = _RIEL_SYMBOL_SUFFIX_RE.sub(_riel_to_words, text)
    text = _RIEL_WORD_RE.sub(_riel_to_words, text)
    return text


def khmer_digits_to_arabic_preserving(text: str) -> str:
    """
    Convert Khmer digits to arabic ONLY when adjacent to a currency
    marker ($, ៛, រៀល), so plain Khmer-digit numbers elsewhere in the
    text are left for numbers.py to handle on its own pass.
    """
    def repl(m: re.Match) -> str:
        return khmer_digits_to_arabic(m.group(0))

    # Khmer digits immediately preceded by $ or ៛, or followed by ៛ / រៀល
    text = re.sub(r"(?<=[\$៛])[០-៩,\.]+", repl, text)
    text = re.sub(r"[០-៩,\.]+(?=\s*៛)", repl, text)
    text = re.sub(r"[០-៩,\.]+(?=\s*រៀល)", repl, text)
    return text


if __name__ == "__main__":
    tests = [
        "ខ្ញុំមាន $10 នៅឆ្នាំ 2026។",
        "តម្លៃសរុបគឺ $15.50។",
        "តម្លៃ ៛5000 សម្រាប់ម្ហូបនេះ។",
        "គាត់បង់ 20000 រៀល។",
    ]
    for t in tests:
        print(t, "->", verbalize_currency_in_text(t))
