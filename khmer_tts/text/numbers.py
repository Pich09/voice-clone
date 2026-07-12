"""
Khmer number verbalizer.

Converts Arabic digit sequences (int, float, or Khmer-digit strings) into
spoken Khmer number words, for use in TTS text normalization.

Design notes:
- Uses the "hundred / thousand / million / billion" grouping that is
  standard in modern spoken/written Khmer for TTS purposes (ម៉ឺន/សែន are
  traditional groupings but ពាន់/លាន/ពាន់លាន is what most contemporary
  Khmer speakers and news readers use, and what generalizes cleanly).
- Handles negative numbers and decimals ("ចុច" = "point").
- Exposes both `number_to_khmer_words` (arabic int/float) and
  `khmer_digits_to_arabic` / `arabic_to_khmer_digits` helpers.
"""

import re

KHMER_DIGITS = "០១២៣៤៥៦៧៨៩"
ARABIC_DIGITS = "0123456789"

_DIGIT_TRANS_KM_TO_AR = str.maketrans(KHMER_DIGITS, ARABIC_DIGITS)
_DIGIT_TRANS_AR_TO_KM = str.maketrans(ARABIC_DIGITS, KHMER_DIGITS)

_ONES = {
    0: "សូន្យ", 1: "មួយ", 2: "ពីរ", 3: "បី", 4: "បួន",
    5: "ប្រាំ", 6: "ប្រាំមួយ", 7: "ប្រាំពីរ", 8: "ប្រាំបី", 9: "ប្រាំបួន",
}

_TEENS_TENS = {
    10: "ដប់", 20: "ម្ភៃ", 30: "សាមសិប", 40: "សែសិប",
    50: "ហាសិប", 60: "ហុកសិប", 70: "ចិតសិប", 80: "ប៉ែតសិប", 90: "កៅសិប",
}

_SCALE = [
    (1_000_000_000, "ពាន់លាន"),
    (1_000_000, "លាន"),
    (1_000, "ពាន់"),
    (100, "រយ"),
]


def khmer_digits_to_arabic(text: str) -> str:
    """Convert any Khmer digits in text to arabic digits."""
    return text.translate(_DIGIT_TRANS_KM_TO_AR)


def arabic_to_khmer_digits(text: str) -> str:
    """Convert any arabic digits in text to Khmer digits (for display, not speech)."""
    return text.translate(_DIGIT_TRANS_AR_TO_KM)


def _two_digit_to_words(n: int) -> str:
    """n in [0, 99]."""
    if n < 10:
        return _ONES[n]
    if n in _TEENS_TENS:
        return _TEENS_TENS[n]
    tens = (n // 10) * 10
    ones = n % 10
    return _TEENS_TENS[tens] + _ONES[ones]


def _three_digit_to_words(n: int) -> str:
    """n in [0, 999]."""
    if n < 100:
        return _two_digit_to_words(n)
    hundreds = n // 100
    rest = n % 100
    out = _ONES[hundreds] + "រយ"
    if rest:
        out += _two_digit_to_words(rest)
    return out


def _int_to_words(n: int) -> str:
    if n == 0:
        return _ONES[0]
    if n < 0:
        return "ដក" + _int_to_words(-n)  # "negative"
    if n < 100:
        return _two_digit_to_words(n)
    if n < 1000:
        return _three_digit_to_words(n)

    parts = []
    remaining = n
    for scale_val, scale_word in _SCALE:
        if remaining >= scale_val:
            count = remaining // scale_val
            remaining = remaining % scale_val
            # count itself may be large (e.g. millions of billions) -> recurse
            count_words = _int_to_words(count) if count > 1 else "មួយ" if count == 1 else ""
            if count_words:
                parts.append(count_words + scale_word)
    if remaining:
        parts.append(_int_to_words(remaining))
    return "".join(parts)


def number_to_khmer_words(value) -> str:
    """
    Convert an int/float/numeric-string (arabic or Khmer digits) into
    spoken Khmer words.

    Examples:
        number_to_khmer_words(10)      -> "ដប់"
        number_to_khmer_words(2026)    -> "ពីរពាន់ម្ភៃប្រាំមួយ"
        number_to_khmer_words(15.5)    -> "ដប់ប្រាំចុចប្រាំ"
        number_to_khmer_words("១៥")    -> "ដប់ប្រាំ"
    """
    if isinstance(value, str):
        value = khmer_digits_to_arabic(value.strip())
        value = float(value) if "." in value else int(value)

    if isinstance(value, float):
        is_negative = value < 0
        value = abs(value)
        int_part = int(value)
        # Format decimal part as digit-by-digit (standard for TTS, avoids
        # ambiguity like 1.50 vs 1.5 meaning different magnitudes)
        frac_str = f"{value:.10f}".split(".")[1].rstrip("0")
        int_words = _int_to_words(int_part)
        if not frac_str:
            return ("ដក" if is_negative else "") + int_words
        frac_words = "".join(_ONES[int(d)] for d in frac_str)
        result = f"{int_words}ចុច{frac_words}"
        return ("ដក" + result) if is_negative else result

    return _int_to_words(int(value))


# Matches integers and decimals, with optional leading minus sign,
# and optional thousands separators (comma) which we strip.
_NUMBER_RE = re.compile(
    r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w])"
)
_KHMER_NUMBER_RE = re.compile(
    r"[០-៩]+(?:\.[០-៩]+)?"
)


def verbalize_numbers_in_text(text: str) -> str:
    """
    Find bare numbers (arabic or Khmer digit sequences) in `text` and
    replace them with spoken Khmer words. Does NOT handle currency,
    dates, or percentages -- see currency.py / dates.py, which should
    run BEFORE this function so they can claim their own number tokens
    first.
    """

    def _replace_arabic(m: re.Match) -> str:
        raw = m.group(0).replace(",", "")
        try:
            return number_to_khmer_words(raw)
        except ValueError:
            return m.group(0)

    def _replace_khmer(m: re.Match) -> str:
        try:
            return number_to_khmer_words(m.group(0))
        except ValueError:
            return m.group(0)

    text = _NUMBER_RE.sub(_replace_arabic, text)
    text = _KHMER_NUMBER_RE.sub(_replace_khmer, text)
    return text


if __name__ == "__main__":
    tests = [10, 2026, 100, 1500, 999, 0, -42, 15.5, "១៥", 1_234_567]
    for t in tests:
        print(t, "->", number_to_khmer_words(t))
