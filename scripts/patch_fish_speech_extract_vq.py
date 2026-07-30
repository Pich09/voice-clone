#!/usr/bin/env python3
"""
Patch vendor/fish-speech's tools/vqgan/extract_vq.py for portability across
machines that are not the 24GB+ CUDA box upstream assumes.

Two independent fixes, each guarded by its own marker so they apply (and
re-apply) separately. Idempotent -- safe to re-run after every clone/update.

1. soundfile audio-I/O fallback
   torchaudio >= 2.9 delegates ALL audio loading to torchcodec, which needs
   the *shared* FFmpeg libraries (libavcodec.so, libavformat.so, ...). A host
   with only a static ffmpeg binary, or without the ffmpeg -dev packages,
   raises "torchaudio version X requires torchcodec for audio I/O" for every
   single file. That is bad here in a specific, silent way -- extract_vq's
   read loop is

       try:
           wav, sr = torchaudio.load(str(file), backend=backend)
       except Exception as e:
           logger.error(f"Error reading {file}: {e}")
           continue

   so it catches the failure per file, logs it, and keeps going: the process
   exits 0 having written ZERO .npy sidecars. build_dataset then emits zero
   protos and training dies later with an opaque ZeroDivisionError, far from
   the real cause. soundfile is already a hard dependency of this repo and
   reads these WAVs fine, so prefer torchaudio and fall back to it.

2. load the codec checkpoint via CPU
   get_model() does

       state_dict = torch.load(checkpoint_path, map_location=device)   # cuda
       ...
       model.to(device)

   which puts the ~1.87GB codec on the GPU twice at once -- once as the
   loaded state dict, once as the model being moved -- for a peak of ~3.7GB
   before a single audio file is read. On a 4GB card that is the difference
   between working and not; measured on an RTX 3050 Laptop it fails, and
   under WSL2 it surfaces as `RuntimeError: CUDA driver error: device not
   ready` rather than a clean CUDA OOM, which makes it look like a driver
   problem instead of a memory one. Loading to CPU first halves peak VRAM and
   changes nothing else: model.to(device) still puts the weights on the GPU.

Usage:
    python scripts/patch_fish_speech_extract_vq.py [--fish-dir vendor/fish-speech]
"""
import argparse
import os

IO_MARKER = "# [khmer-voice-clone patch] soundfile audio-io fallback"
MEM_MARKER = "# [khmer-voice-clone patch] load codec via CPU to halve peak VRAM"

IO_FUNC = f'''{IO_MARKER}
import numpy as _np
import soundfile as _sf


def _load_audio_any(path):
    """torchaudio.load(), falling back to soundfile.

    See scripts/patch_fish_speech_extract_vq.py for why. Returns the same
    (Tensor[channels, samples], sample_rate) shape torchaudio.load does.
    """
    try:
        return torchaudio.load(str(path), backend=backend)
    except Exception:
        data, sr = _sf.read(str(path), always_2d=False, dtype="float32")
        if data.ndim > 1:
            data = data.T          # (samples, ch) -> (ch, samples)
        else:
            data = data[None, :]   # mono -> (1, samples)
        return torch.from_numpy(_np.ascontiguousarray(data)), sr


'''

IO_OLD = """            wav, sr = torchaudio.load(
                str(file), backend=backend
            )  # Need to install libsox-dev"""
IO_NEW = """            wav, sr = _load_audio_any(file)"""
IO_ANCHOR = 'RANK = int(os.environ.get("SLURM_PROCID", 0))'

MEM_OLD = """    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
    )"""
MEM_NEW = f"""    {MEM_MARKER}
    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
    )"""


def apply(src, marker, checks, transform, label):
    """Apply one patch if its marker is absent. Returns (new_src, message)."""
    if marker in src:
        return src, f"  [skip] {label}: already applied"
    for needle, why in checks:
        if needle not in src:
            raise SystemExit(
                f"{label}: expected code not found ({why}). fish-speech's "
                "extract_vq.py has changed upstream -- re-check this patch "
                "against the new source before running it again."
            )
    return transform(src), f"  [ok]   {label}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fish-dir", default="vendor/fish-speech")
    args = ap.parse_args()

    target = os.path.join(args.fish_dir, "tools", "vqgan", "extract_vq.py")
    if not os.path.isfile(target):
        raise SystemExit(f"not found: {target}")

    with open(target, encoding="utf-8") as f:
        src = f.read()
    original = src
    msgs = []

    src, m = apply(
        src, IO_MARKER,
        [(IO_ANCHOR, "the RANK assignment used as the insertion point"),
         (IO_OLD, "the torchaudio.load call to replace")],
        lambda s: s.replace(IO_ANCHOR, IO_FUNC + IO_ANCHOR, 1).replace(IO_OLD, IO_NEW, 1),
        "soundfile audio-io fallback",
    )
    msgs.append(m)

    src, m = apply(
        src, MEM_MARKER,
        [(MEM_OLD, "the torch.load(map_location=device) call to replace")],
        lambda s: s.replace(MEM_OLD, MEM_NEW, 1),
        "load codec via CPU",
    )
    msgs.append(m)

    if src != original:
        with open(target, "w", encoding="utf-8") as f:
            f.write(src)
    print(f"extract_vq patches for {target}:")
    for m in msgs:
        print(m)


if __name__ == "__main__":
    main()
