"""
Khmer date verbalizer.

Handles common date patterns and converts them to spoken Khmer:
    ថ្ងៃទី 5 ខែមករា ឆ្នាំ 2026  -> ថ្ងៃទីប្រាំ ខែមករា ឆ្នាំពីរពាន់ម្ភៃប្រាំមួយ
    2026                        -> ពីរពាន់ម្ភៃប្រាំមួយ  (bare 4-digit years
                                    are read digit-pair style, matching how
                                    Khmer speakers usually say years)
    05/01/2026                  -> ថ្ងៃទីប្រាំ ខែមករា ឆ្នាំពីរពាន់ម្ភៃប្រាំមួយ

Run this BEFORE the generic number verbalizer, since it needs to
recognize year/month/day tokens before they turn into plain digit words.
"""

import re
from .numbers import number_to_khmer_words, khmer_digits_to_arabic

_KHMER_MONTHS = {
    1: "មករា", 2: "កុម្ភៈ", 3: "មីនា", 4: "មេសា", 5: "ឧសភា", 6: "មិថុនា",
    7: "កក្កដា", 8: "សីហា", 9: "កញ្ញា", 10: "តុលា", 11: "វិច្ឆិកា", 12: "ធ្នូ",
}

# ថ្ងៃទី <day> ខែ<month word or number> ឆ្នាំ <year>
_FULL_DATE_RE = re.compile(
    r"ថ្ងៃទី\s*([0-9០-៩]{1,2})\s*ខែ\s*([0-9០-៩]{1,2}|[ក-អ]+)\s*ឆ្នាំ\s*([0-9០-៩]{4})"
)

# Numeric dd/mm/yyyy or dd-mm-yyyy
_NUMERIC_DATE_RE = re.compile(
    r"\b([0-3]?[0-9])[/\-]([01]?[0-9])[/\-]([12][0-9]{3})\b"
)

# Bare 4-digit year not already consumed above, e.g. "ឆ្នាំ 2026" or "ក្នុងឆ្នាំ2026"
_YEAR_RE = re.compile(r"ឆ្នាំ\s*([0-9០-៩]{4})")


def _year_to_words(year: int) -> str:
    """Khmer speakers typically read 4-digit years as a full cardinal
    number (ពីរពាន់ម្ភៃប្រាំមួយ for 2026), not split digit-pairs."""
    return number_to_khmer_words(year)


def _full_date_repl(m: re.Match) -> str:
    day_raw, month_raw, year_raw = m.groups()
    day = int(khmer_digits_to_arabic(day_raw))
    year = int(khmer_digits_to_arabic(year_raw))

    if month_raw.isdigit() or all(c in "0123456789០១២៣៤៥៦៧៨៩" for c in month_raw):
        month_num = int(khmer_digits_to_arabic(month_raw))
        month_word = _KHMER_MONTHS.get(month_num, month_raw)
    else:
        month_word = month_raw  # already a Khmer month name

    return f"ថ្ងៃទី{number_to_khmer_words(day)} ខែ{month_word} ឆ្នាំ{_year_to_words(year)}"


def _numeric_date_repl(m: re.Match) -> str:
    day, month, year = (int(x) for x in m.groups())
    month_word = _KHMER_MONTHS.get(month, str(month))
    return f"ថ្ងៃទី{number_to_khmer_words(day)} ខែ{month_word} ឆ្នាំ{_year_to_words(year)}"


def _year_only_repl(m: re.Match) -> str:
    year = int(khmer_digits_to_arabic(m.group(1)))
    return f"ឆ្នាំ{_year_to_words(year)}"


def verbalize_dates_in_text(text: str) -> str:
    text = _FULL_DATE_RE.sub(_full_date_repl, text)
    text = _NUMERIC_DATE_RE.sub(_numeric_date_repl, text)
    text = _YEAR_RE.sub(_year_only_repl, text)
    return text


if __name__ == "__main__":
    tests = [
        "ថ្ងៃទី 5 ខែមករា ឆ្នាំ 2026",
        "05/01/2026",
        "នៅឆ្នាំ 2026 បច្ចេកវិទ្យានឹងកាន់តែប្រសើរ។",
    ]
    for t in tests:
        print(t, "->", verbalize_dates_in_text(t))
