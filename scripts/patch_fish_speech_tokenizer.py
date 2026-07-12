#!/usr/bin/env python3
"""
Patch vendor/fish-speech's FishTokenizer to load a bare tokenizer.tiktoken +
special_tokens.json pair directly via the `tiktoken` library, instead of
`transformers.AutoTokenizer.from_pretrained()`.

Why: FishTokenizer.__init__ calls
    AutoTokenizer.from_pretrained(f"{ckpt_dir}/tokenizer.tiktoken")
passing a *file* path. transformers has always required a *directory*
(containing tokenizer_config.json) for that call -- confirmed by checking the
actual files published in fishaudio/openaudio-s1-mini on Hugging Face, which
never included a tokenizer_config.json. This reproduces identically across
transformers 4.44.2, 4.56.1 (fish-speech's own uv.lock pin), and 4.57.3
(fish-speech's pyproject upper bound) -- it's a genuine upstream bug in
fish-speech's current main branch, not a version-pinning issue.

Run this once after cloning/updating vendor/fish-speech and before running
scripts/10_train_fish_khmer_base.sh. Idempotent -- safe to re-run.

Usage:
    python scripts/patch_fish_speech_tokenizer.py [--fish-dir vendor/fish-speech]
"""
import argparse
import os

MARKER = "# [khmer-voice-clone patch] tiktoken-file loader"

PATCH_IMPORTS = f"""{MARKER}
import os as _os


def _resolve_tiktoken_dir(model_path):
    \"\"\"FishTokenizer gets called two ways in fish-speech: with the tokenizer
    file itself (.../tokenizer.tiktoken) from the training config, and with
    the checkpoint *directory* from BaseTransformer.from_pretrained(). Handle
    both.\"\"\"
    if _os.path.isfile(model_path) and model_path.endswith(".tiktoken"):
        return _os.path.dirname(model_path)
    if _os.path.isdir(model_path) and _os.path.isfile(
        _os.path.join(model_path, "tokenizer.tiktoken")
    ):
        return model_path
    return None
"""

PATCH_INIT = '''    def __init__(self, model_path: str):
        _tiktoken_dir = _resolve_tiktoken_dir(model_path)
        if _tiktoken_dir is not None:
            self._tokenizer = _TiktokenFileBackend(_tiktoken_dir)
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.semantic_id_to_token_id = {}'''

BACKEND_CLASS = f'''

{MARKER}
class _TiktokenFileBackend:
    """Minimal AutoTokenizer-shaped wrapper around a raw tokenizer.tiktoken +
    special_tokens.json pair, for checkpoints (e.g. fishaudio/openaudio-s1-mini)
    that never shipped a tokenizer_config.json. Implements only the surface
    FishTokenizer and the training data pipeline actually touch: get_vocab,
    encode, decode, convert_tokens_to_ids, vocab_size, pad_token_id,
    eos_token_id, save_pretrained.

    Pre-tokenization regex: fish-speech doesn't publish the exact pat_str used
    to build this vocab, so this uses the standard cl100k-style regex (the
    common default for custom tiktoken BPEs of this shape). Token IDs/vocab
    membership are exact either way; only merge-boundary choices for
    never-before-seen text could differ slightly from the original training
    regex -- fine for fine-tuning, worth revisiting if you need bit-exact
    parity with upstream inference results.
    """

    _PAT_STR = (
        r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{{L}}\\p{{N}}]?\\p{{L}}+|\\p{{N}}{{1,3}}"""
        r"""| ?[^\\s\\p{{L}}\\p{{N}}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"""
    )

    def __init__(self, model_dir: str):
        import json as _json

        from tiktoken import Encoding as _Encoding
        from tiktoken.load import load_tiktoken_bpe as _load_tiktoken_bpe

        mergeable_ranks = _load_tiktoken_bpe(
            _os.path.join(model_dir, "tokenizer.tiktoken")
        )
        with open(
            _os.path.join(model_dir, "special_tokens.json"), encoding="utf-8"
        ) as f:
            special_tokens = _json.load(f)

        self._model_dir = model_dir
        self._special_tokens = special_tokens
        self._enc = _Encoding(
            name="fish-tiktoken-local",
            pat_str=self._PAT_STR,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        self._vocab = {{
            tok.decode("utf-8", errors="replace"): idx
            for tok, idx in mergeable_ranks.items()
        }}
        self._vocab.update(special_tokens)

    def get_vocab(self):
        return dict(self._vocab)

    @property
    def vocab_size(self):
        return self._enc.n_vocab

    @property
    def pad_token_id(self):
        return self._special_tokens.get("<|pad|>")

    @property
    def eos_token_id(self):
        return self._special_tokens.get("<|end_of_text|>") or self._special_tokens.get(
            "<|endoftext|>"
        )

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(token)

    def encode(self, text, add_special_tokens=False, allowed_special="all", **kwargs):
        return self._enc.encode(text, allowed_special=allowed_special)

    def decode(self, tokens, **kwargs):
        return self._enc.decode(tokens)

    def save_pretrained(self, path):
        import shutil

        os.makedirs(path, exist_ok=True)
        shutil.copy(
            _os.path.join(self._model_dir, "tokenizer.tiktoken"),
            _os.path.join(path, "tokenizer.tiktoken"),
        )
        shutil.copy(
            _os.path.join(self._model_dir, "special_tokens.json"),
            _os.path.join(path, "special_tokens.json"),
        )
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fish-dir", default="vendor/fish-speech")
    args = parser.parse_args()

    path = os.path.join(args.fish_dir, "fish_speech", "tokenizer.py")
    if not os.path.isfile(path):
        raise SystemExit(f"{path} not found -- clone vendor/fish-speech first.")

    with open(path, encoding="utf-8") as f:
        src = f.read()

    if MARKER in src:
        print(f"{path} already patched, skipping.")
        return

    old_init = '''    def __init__(self, model_path: str):
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.semantic_id_to_token_id = {}'''
    if old_init not in src:
        raise SystemExit(
            "Could not find the expected FishTokenizer.__init__ body to patch -- "
            "vendor/fish-speech's tokenizer.py may have changed upstream. "
            "Check scripts/patch_fish_speech_tokenizer.py and update it."
        )

    src = src.replace(
        "import torch\nfrom transformers import AutoTokenizer\n",
        "import torch\nfrom transformers import AutoTokenizer\n\n" + PATCH_IMPORTS,
        1,
    )
    src = src.replace(old_init, PATCH_INIT, 1)
    src = src.rstrip("\n") + "\n" + BACKEND_CLASS

    with open(path, "w", encoding="utf-8") as f:
        f.write(src)

    print(f"Patched {path}.")


if __name__ == "__main__":
    main()
