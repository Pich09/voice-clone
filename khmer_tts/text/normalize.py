"""
Khmer text normalizer for TTS.

Pipeline (order matters):
  1. Unicode normalization (NFC) + zero-width space / control char removal
  2. Whitespace cleanup (collapse duplicate spaces, strip)
  3. Khmer character reordering (fix common Unicode encoding issues where
     dependent vowels / coeng sequences are typed out of canonical order)
  4. Currency verbalization ($10, ៛5000, 20000 រៀល)
  5. Date verbalization (ថ្ងៃទី.../dd-mm-yyyy/ឆ្នាំ 2026)
  6. Percentage verbalization (25% -> ម្ភៃប្រាំភាគរយ)
  7. Generic number verbalization (whatever numbers remain)
  8. Latin symbol / punctuation cleanup
  9. Sentence splitting (exposed separately via split_sentences)

Use `normalize_khmer_text(text)` as the single entry point that should
run identically before training, validation, and inference.
"""

import re
import unicodedata

from .currency import verbalize_currency_in_text
from .dates import verbalize_dates_in_text
from .numbers import number_to_khmer_words, verbalize_numbers_in_text, khmer_digits_to_arabic

ZERO_WIDTH_CHARS = "\u200b\u200c\u200d\ufeff"
_ZERO_WIDTH_RE = re.compile("[" + ZERO_WIDTH_CHARS + "]")

_WHITESPACE_RE = re.compile(r"[ \t\u00a0]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{2,}")

_PERCENT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")

# Common Latin punctuation -> Khmer/neutral equivalents for TTS purposes.
# We keep sentence-final khan (។) and question mark since Fish Speech /
# most TTS front-ends use these as phrase-break signals.
_PUNCT_MAP = {
    "…": "។",
    "..": ".",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
}

# Coeng (U+17D2) must always be immediately followed by a consonant.
# A common encoding mistake is a stray coeng at end of a cluster with no
# following consonant, or double coeng. This is a conservative cleanup,
# not a full grammar-based re-orderer.
_DOUBLE_COENG_RE = re.compile("\u17d2\u17d2+")


def _fix_khmer_ordering(text: str) -> str:
    """Conservative fixes for common Khmer Unicode ordering issues."""
    text = _DOUBLE_COENG_RE.sub("\u17d2", text)
    # Drop a coeng that has nothing after it (trailing/dangling coeng)
    text = re.sub("\u17d2(?=\\s|$)", "", text)
    return text


def _clean_unicode_and_whitespace(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n", text)
    return text.strip()


def _normalize_punctuation(text: str) -> str:
    for src, dst in _PUNCT_MAP.items():
        text = text.replace(src, dst)
    return text


def _verbalize_percent(text: str) -> str:
    def repl(m: re.Match) -> str:
        num = m.group(1)
        value = float(num) if "." in num else int(num)
        return number_to_khmer_words(value) + "ភាគរយ"

    return _PERCENT_RE.sub(repl, text)


def normalize_khmer_text(text: str) -> str:
    """Full normalization pipeline. Idempotent: running it twice on
    already-normalized text should be a no-op."""
    if not text:
        return text

    text = _clean_unicode_and_whitespace(text)
    text = _fix_khmer_ordering(text)
    text = _normalize_punctuation(text)

    # Order matters: currency and dates must claim their numbers before
    # the generic number verbalizer runs.
    text = verbalize_currency_in_text(text)
    text = verbalize_dates_in_text(text)
    text = _verbalize_percent(text)
    text = verbalize_numbers_in_text(text)

    # Final whitespace cleanup after all the substitutions above.
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


# Khmer sentence-ending punctuation: khan (។), question mark, exclamation.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[។!?])\s*")


def split_sentences(text: str) -> list[str]:
    """Split normalized Khmer text into sentences for per-sentence TTS
    generation (see Section 16, 'Long text unstable' fix)."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return sentences


if __name__ == "__main__":
    raw = "ខ្ញុំមាន $10 នៅឆ្នាំ 2026។ តម្លៃបញ្ចុះ 25% សម្រាប់អតិថិជនថ្មី!"
    normalized = normalize_khmer_text(raw)
    print("RAW:       ", raw)
    print("NORMALIZED:", normalized)
    print("SENTENCES: ", split_sentences(normalized))
