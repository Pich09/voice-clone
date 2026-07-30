#!/usr/bin/env python3
"""
Patch vendor/fish-speech's SemanticDataModule so num_workers=0 is a legal
configuration.

Why num_workers=0 is needed at all: Python 3.14 changed Linux's default
multiprocessing start method from `fork` to `forkserver`. forkserver PICKLES
the dataset to hand it to each worker, and fish-speech's dataset carries a
tiktoken-backed FishTokenizer, which is not picklable -- so every worker dies
during the handshake and the parent reports only

    BrokenPipeError: [Errno 32] Broken pipe

from popen_forkserver._launch, which says nothing about pickling. Under the old
fork default (Python <= 3.13, i.e. Kaggle and Colab today) worker memory was
copied and nothing was pickled, so this never appeared. Forcing fork back is
not an acceptable fix: fork with CUDA already initialised is unsafe, and
avoiding it is exactly why 3.14 changed the default. Loading batches in the
main process is the safe option.

Why this patch: SemanticDataModule.train_dataloader/val_dataloader hardcode

    persistent_workers=True

and torch raises `ValueError: persistent_workers option needs num_workers > 0`,
so setting data.num_workers=0 alone just trades one crash for another. Make the
flag follow the worker count instead -- persistent workers are meaningless with
no workers, and the behaviour is unchanged whenever num_workers > 0.

Idempotent -- safe to re-run after every clone/update.

Usage:
    python scripts/patch_fish_speech_dataloader.py [--fish-dir vendor/fish-speech]
"""
import argparse
import os

MARKER = "# [khmer-voice-clone patch] persistent_workers must follow num_workers"

OLD = """            num_workers=self.num_workers,
            persistent_workers=True,
        )"""

NEW = f"""            num_workers=self.num_workers,
            {MARKER}
            # torch rejects persistent_workers=True when num_workers == 0, and
            # num_workers=0 is the only safe setting on Python 3.14+ (forkserver
            # cannot pickle the tiktoken tokenizer). Identical behaviour for
            # num_workers > 0.
            persistent_workers=self.num_workers > 0,
        )"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fish-dir", default="vendor/fish-speech")
    args = ap.parse_args()

    target = os.path.join(args.fish_dir, "fish_speech", "datasets", "semantic.py")
    if not os.path.isfile(target):
        raise SystemExit(f"not found: {target}")

    with open(target, encoding="utf-8") as f:
        src = f.read()

    if MARKER in src:
        print(f"already patched: {target}")
        return

    n = src.count(OLD)
    if n != 2:
        raise SystemExit(
            f"expected 2 hardcoded persistent_workers=True dataloaders in "
            f"{target}, found {n} -- fish-speech has changed upstream; "
            "re-check this patch against the new source."
        )

    with open(target, "w", encoding="utf-8") as f:
        f.write(src.replace(OLD, NEW))
    print(f"patched: {target} (train_dataloader + val_dataloader)")


if __name__ == "__main__":
    main()
