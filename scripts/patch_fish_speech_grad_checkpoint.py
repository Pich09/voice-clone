#!/usr/bin/env python3
"""
Patch vendor/fish-speech's DualARTransformer to use NON-reentrant activation
checkpointing (`use_reentrant=False`) instead of the reentrant variant.

Why: fish-speech hardcodes `checkpoint(layer, ..., use_reentrant=True)` in both
the slow and fast transformer stacks (llama.py, BaseTransformer.forward and
DualARTransformer.forward). Reentrant checkpointing runs its recomputed
backward inside a *nested* autograd engine call, which fires DDP's
gradient-ready hooks a second time for every parameter in the recomputed
block. On a single GPU nothing notices. Under DDP (multi-GPU) it breaks two
different ways, both of which we hit on Kaggle's 2xT4:

  1. With find_unused_parameters=false (PyTorch's default), the reducer's
     bucket accounting goes out of sync between ranks and the allreduce
     deadlocks -- no error, no crash, just a session that sits silent
     forever (we burned ~90 minutes on exactly this).
  2. With find_unused_parameters=true, the double hook firing is detected
     and hard-fails instead:
         RuntimeError: Expected to mark a variable ready only once.
         Parameter at index 41 with name
         model.fast_layers.3.feed_forward.w2.lora_B has been marked as
         ready twice.

PyTorch's own error text suggests `_set_static_graph()` as *a* workaround.
Non-reentrant checkpointing is the better one: it uses saved-tensor hooks
rather than a nested backward, so each parameter's hook fires exactly once
per iteration and DDP needs no special-casing at all. It is also the mode
PyTorch now recommends generally (reentrant is the legacy path), and what
HF PEFT sets for LoRA training for this same reason.

Memory/compute behaviour is equivalent -- activations are still recomputed
rather than stored; only the mechanism for re-entering backward changes.

Run this once after cloning/updating vendor/fish-speech and before training.
Idempotent -- safe to re-run.

Usage:
    python scripts/patch_fish_speech_grad_checkpoint.py [--fish-dir vendor/fish-speech]
"""
import argparse
import os

MARKER = "# [khmer-voice-clone patch] non-reentrant activation checkpointing"

OLD_SLOW = (
    "                x = checkpoint(layer, x, freqs_cis, mask, use_reentrant=True)"
)
NEW_SLOW = (
    "                x = checkpoint(  " + MARKER + "\n"
    "                    layer, x, freqs_cis, mask, use_reentrant=False\n"
    "                )"
)

OLD_FAST = (
    "                x = checkpoint(layer, x, fast_freqs_cis, fast_mask, use_reentrant=True)"
)
NEW_FAST = (
    "                x = checkpoint(  " + MARKER + "\n"
    "                    layer, x, fast_freqs_cis, fast_mask, use_reentrant=False\n"
    "                )"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fish-dir", default="vendor/fish-speech")
    args = parser.parse_args()

    path = os.path.join(
        args.fish_dir, "fish_speech", "models", "text2semantic", "llama.py"
    )
    if not os.path.isfile(path):
        raise SystemExit(f"{path} not found -- clone vendor/fish-speech first.")

    with open(path, encoding="utf-8") as f:
        src = f.read()

    if MARKER in src:
        print(f"{path} already patched, skipping.")
        return

    missing = [name for name, old in (("slow", OLD_SLOW), ("fast", OLD_FAST))
               if old not in src]
    if missing:
        raise SystemExit(
            f"Could not find the expected {'/'.join(missing)} transformer "
            "checkpoint call(s) to patch -- vendor/fish-speech's llama.py may "
            "have changed upstream. Check scripts/"
            "patch_fish_speech_grad_checkpoint.py and update it. Training "
            "would deadlock or crash under DDP without this patch."
        )

    src = src.replace(OLD_SLOW, NEW_SLOW, 1)
    src = src.replace(OLD_FAST, NEW_FAST, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(src)

    print(f"Patched {path} (activation checkpointing -> use_reentrant=False).")


if __name__ == "__main__":
    main()
