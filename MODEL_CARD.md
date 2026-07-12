# MODEL_CARD.md

Fill this in for each released checkpoint before production use.

## Model

- **Name:**
- **Version:**
- **Base architecture:** Fish Speech (fishaudio/fish-speech-1.5)
- **Stage:** [Khmer base | Voice clone]
- **Date trained:**
- **Checkpoint path:**

## Training data

- **Dataset(s):**
- **Total hours (A-grade / B-grade):**
- **Speaker(s):**
- **Consent records location:** (required for any real-person voice clone)

## Evaluation results

- **Pronunciation score:** _/5.0 (gate: ≥4.0)
- **Naturalness score:** _/5.0 (gate: ≥3.8)
- **Voice similarity score:** _/5.0 (gate: ≥4.0, voice-clone stage only)
- **Long-text stability:** pass/fail
- **Known failure modes:**

## Intended use

- Text → Khmer speech synthesis for [describe use case].
- **Not intended for:** impersonation without consent, voice authentication bypass, generating misleading audio of real individuals.

## Consent & safety

- Voice donor consent obtained: yes/no, date, scope of use.
- Any usage restrictions:
- Watermarking / provenance marking applied: yes/no.

## Limitations

- Pronunciation quality is bounded by training data coverage of Khmer
  dialects, numbers/dates/currency, and code-switched text.
- Not evaluated for [specific domains, e.g. medical terminology].
