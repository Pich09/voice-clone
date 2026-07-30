#!/usr/bin/env python3
"""
Patch the installed hydra-core so its argument parser survives Python 3.14.

Why: Python 3.14 added ArgumentParser._check_help(), which validates every
help value at add_argument() time by running

    if '%' not in help_string:

hydra's get_args_parser() registers --shell-completion with
help=LazyCompletionHelp(), an object that implements only __repr__ (it defers
building the completion text until render time). The membership test therefore
raises

    TypeError: argument of type 'LazyCompletionHelp' is not a container or iterable

which argparse re-raises as `ValueError: badly formed help string`. The effect
is total on 3.14: EVERY hydra entrypoint dies during argument-parser
construction, before any user code runs. For this repo that means
vendor/fish-speech/fish_speech/train.py cannot start at all -- Step 3 of
scripts/10 and scripts/12 fail immediately, after VQ extraction has already
spent the GPU time.

hydra-core 1.3.4 is the newest published release and still carries this, so
there is nothing to upgrade to.

The fix makes LazyCompletionHelp a real `str` subclass, built eagerly. That
gives up the deferred construction (one plugin discovery at parser build time,
which is what the laziness avoided) in exchange for actually working. A
lighter shim that only adds __contains__ would get past _check_help, but
`--help` would then crash inside _split_lines, which does a regex sub over the
value and needs a genuine str.

No-op on Python < 3.14, where _check_help does not exist and the lazy object is
fine. Idempotent -- safe to re-run.

Usage:
    python scripts/patch_hydra_py314_help.py
"""
import argparse
import importlib.util
import os
import sys

MARKER = "# [khmer-voice-clone patch] py3.14 argparse help validation"

OLD = '''    # defer building the completion help string until we actually need to render it
    class LazyCompletionHelp:
        def __repr__(self) -> str:
            return f"Install or Uninstall shell completion:\\n{_get_completion_help()}"
'''

NEW = f'''    {MARKER}
    # Python 3.14's ArgumentParser._check_help runs `'%' not in help_string` on
    # every help value. A lazy object that only defines __repr__ makes that
    # membership test raise TypeError, which argparse reports as the misleading
    # "badly formed help string" -- killing every hydra entrypoint at parser
    # construction. Build the string eagerly as a real str instead. See
    # scripts/patch_hydra_py314_help.py.
    class LazyCompletionHelp(str):
        def __new__(cls) -> "LazyCompletionHelp":
            return super().__new__(
                cls, f"Install or Uninstall shell completion:\\n{{_get_completion_help()}}"
            )
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="patch even on Python < 3.14 (normally a no-op there)")
    args = ap.parse_args()

    if sys.version_info < (3, 14) and not args.force:
        print(f"Python {sys.version_info.major}.{sys.version_info.minor} "
              "is unaffected (argparse gained _check_help in 3.14) -- nothing to do.")
        return

    spec = importlib.util.find_spec("hydra._internal.utils")
    if spec is None or not spec.origin:
        raise SystemExit("hydra-core is not installed in this environment.")
    target = spec.origin

    with open(target, encoding="utf-8") as f:
        src = f.read()

    if MARKER in src:
        print(f"already patched: {target}")
        return

    if OLD not in src:
        raise SystemExit(
            f"could not find LazyCompletionHelp in {target} -- hydra-core "
            f"{_hydra_version()} may already define it differently. Re-check "
            "this patch against the installed source."
        )

    with open(target, "w", encoding="utf-8") as f:
        f.write(src.replace(OLD, NEW, 1))
    print(f"patched: {target}")


def _hydra_version():
    try:
        import importlib.metadata as m
        return m.version("hydra-core")
    except Exception:
        return "?"


if __name__ == "__main__":
    main()
