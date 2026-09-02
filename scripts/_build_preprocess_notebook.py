#!/usr/bin/env python3
"""One-off generator for kaggle/preprocess_khmer_vq_v2.ipynb. Run manually after
editing this file; not part of the pipeline. Keeps the notebook's JSON out of
a hand-edited diff -- easier to review as Python source than as raw ipynb.

This notebook exists ONLY for Kaggle's 20GB /kaggle/working cap, which can't
hold the full ~22GB processed dataset at once. It works through the dataset
shard-by-shard: pull one shard's raw audio, VQ-extract it, pack it into
protobufs, upload the (much smaller) result, delete the local raw audio, move
to the next shard -- until every shard (and the validation set) is done. Each
shard is uploaded and marked ready as soon as it finishes, so this is safe to
interrupt and re-run (Run All) across as many Kaggle sessions as it takes;
already-ready shards are skipped.

kaggle/train_khmer_base.ipynb (the OTHER notebook) refuses to start training
on Kaggle until every shard reports ready here -- see its Section 4. Colab
doesn't need any of this (its local disk holds the full dataset), so it never
touches this notebook at all.
"""
import json
import os

def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}

cells = []

cells.append(md("""\
# Khmer Base Model — VQ Preprocessing (Kaggle)

Prepares the dataset for training **on Kaggle only**: pulls one data shard at
a time, VQ-extracts it, packs it into protobuf shards, uploads the result,
and moves on -- because Kaggle's `/kaggle/working` is hard-capped at 20GB,
smaller than the full ~22GB processed dataset.

**You do not need this notebook on Colab or locally** -- `train_khmer_base.ipynb`
handles the full dataset directly there.

Run this **repeatedly** (Run All) until it reports every shard done -- Kaggle
GPU sessions are time-limited, and this notebook always resumes from whatever
was already uploaded, never redoing finished shards. Once it reports
everything ready, switch to `train_khmer_base.ipynb` to actually train.

**Setup**
1. Enable a GPU: Settings > Accelerator > GPU T4 x2.
2. Add an `HF_TOKEN` secret (Add-ons > Secrets) -- needed to pull/push data.
3. Run All. Repeat next session if it doesn't finish.
"""))

cells.append(md("## 0 - Configuration - edit this cell, then Run All"))

cells.append(code("""\
# ============================= CONFIG =============================
NOTEBOOK_REVISION = 3  # v2 notebook, independent of the original

GITHUB_URL   = "https://github.com/Pich09/voice-clone.git"
DATASET_PATH = ""
WORKDIR      = ""   # "" = auto-pick per environment (see Section 2)

# Must match train_khmer_base.ipynb's HF_DATA_REPO / DATA_CACHE_KEY exactly --
# this notebook prepares data that one reads back by the same key.
HF_DATA_REPO   = "Panhapich/khmer-tts-processed"
DATA_CACHE_KEY = "khmer_base_v1"
# =================================================================
print('Config loaded.')
"""))

cells.append(md("## 1 - Environment & GPU check"))

cells.append(code("""\
import subprocess, sys, os

# This notebook only ever runs on Kaggle -- it exists purely to work around
# Kaggle's 20GB /kaggle/working cap (see the intro cell). Hardcoded rather
# than auto-detected: Kaggle's own kernel images are apparently built on
# Google's Colab base image lineage, so COLAB_RELEASE_TAG (normally a
# reliable Colab-only signal) can ALSO be set on a genuine Kaggle kernel --
# measured directly on a Kaggle notebook, not a guess -- which made
# environment auto-detection actively wrong here. No ambiguity to resolve
# once this notebook is Kaggle-only by definition.
IN_KAGGLE, IN_COLAB, IN_LOCAL = True, False, False
ENV_NAME = 'Kaggle'
print('Environment:', ENV_NAME, '(hardcoded -- this notebook is Kaggle-only)')

import torch
print('Python :', sys.version.split()[0])
print('Torch  :', torch.__version__)
print('CUDA   :', torch.cuda.is_available())

if torch.cuda.is_available():
    ACCELERATOR = 'gpu'
    print('GPU    :', torch.cuda.get_device_name(0), f'({torch.cuda.device_count()} device(s))')
    print(subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.free','--format=csv'],
                         capture_output=True, text=True).stdout)
else:
    ACCELERATOR = 'cpu'
    print('!! No GPU detected -- VQ extraction needs one (extract_vq.py is CUDA-only). '
          'Enable the GPU accelerator in this platform\\'s settings.')

VRAM_GB = (torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
           if torch.cuda.is_available() else 0.0)

# Same per-GPU extract_vq sizing as train_khmer_base.ipynb's Section 1 -- see
# that notebook for the full tier rationale (extract_vq.py round-robins
# workers across CUDA_VISIBLE_DEVICES, so sizing scales with GPU count).
N_GPUS = max(1, torch.cuda.device_count()) if ACCELERATOR == 'gpu' else 1
if ACCELERATOR == 'cpu':
    EXTRACT_WORKERS, EXTRACT_BATCH_SIZE = 1, 4
elif VRAM_GB < 6:
    EXTRACT_WORKERS, EXTRACT_BATCH_SIZE = 1 * N_GPUS, 4
elif VRAM_GB < 12:
    EXTRACT_WORKERS, EXTRACT_BATCH_SIZE = 1 * N_GPUS, 8
elif VRAM_GB < 20:
    # 2 workers/GPU measured OOMing on an actual 14.56GB Kaggle T4 (2 workers
    # sharing one GPU landed at ~14GB combined, no headroom left for a
    # batch's normal size fluctuation) -- 1 worker/GPU gives each the whole
    # card instead, slower but reliable.
    EXTRACT_WORKERS, EXTRACT_BATCH_SIZE = 1 * N_GPUS, 8
else:
    EXTRACT_WORKERS, EXTRACT_BATCH_SIZE = 2 * N_GPUS, 16
print(f'extract : {EXTRACT_WORKERS} worker(s) across {N_GPUS} GPU(s), '
      f'batch {EXTRACT_BATCH_SIZE}')
"""))

cells.append(md("## 2 - Get the code"))

cells.append(code("""\
import os, shutil, subprocess, sys

if not WORKDIR:
    if IN_KAGGLE:
        WORKDIR = '/kaggle/working/khmer-voice-clone'
    elif IN_COLAB:
        WORKDIR = '/content/khmer-voice-clone'
    else:
        WORKDIR = os.getcwd()
        _probe = WORKDIR
        while True:
            if os.path.isdir(os.path.join(_probe, 'khmer_tts')):
                WORKDIR = _probe
                break
            _parent = os.path.dirname(_probe)
            if _parent == _probe:
                break
            _probe = _parent
        if WORKDIR != os.getcwd():
            print('Found the repo root above the notebook cwd:', WORKDIR)

REPO_MARKER = os.path.join(WORKDIR, 'khmer_tts')
if os.path.isdir(REPO_MARKER):
    print('WORKDIR already has the repo, reusing:', WORKDIR)
    if os.path.isdir(os.path.join(WORKDIR, '.git')):
        result = subprocess.run(['git', '-C', WORKDIR, 'pull'],
                                 capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f'git pull failed in {WORKDIR} (exit code {result.returncode}):\\n'
                f'{result.stdout}\\n{result.stderr}\\n\\n'
                'This WORKDIR has local changes or history that conflict with '
                'the latest GitHub state. Easiest fix: back up anything you '
                'care about under this path, delete the WORKDIR folder, and '
                're-run this cell for a clean clone.'
            )
        print(result.stdout.strip() or 'Already up to date.')
elif DATASET_PATH and os.path.isdir(DATASET_PATH):
    shutil.copytree(DATASET_PATH, WORKDIR, dirs_exist_ok=True)
    print('Copied repo from dataset ->', WORKDIR)
elif GITHUB_URL:
    _clone = subprocess.run(['git', 'clone', '--depth', '1', GITHUB_URL, WORKDIR],
                             capture_output=True, text=True)
    if _clone.returncode != 0:
        _err = (_clone.stderr or '') + (_clone.stdout or '')
        _hint = ''
        if 'already exists and is not an empty directory' in _err:
            _listing = ''
            try:
                _listing = ', '.join(sorted(os.listdir(WORKDIR))[:12]) or '(empty)'
            except OSError:
                _listing = '(unreadable)'
            _hint = (
                f'\\n\\n{WORKDIR} already exists but does not contain khmer_tts/, '
                f'so it is not a usable checkout. It currently holds: {_listing}\\n'
                'Fix: MOVE it aside rather than deleting it, then re-run this cell --\\n'
                f'    import shutil; shutil.move({WORKDIR!r}, {WORKDIR + ".bak"!r})'
            )
        elif 'could not resolve host' in _err.lower() or 'network' in _err.lower():
            _hint = ('\\n\\nThis looks like no network access. On Kaggle, turn '
                     'Internet ON in the notebook Settings panel and re-run.')
        raise RuntimeError(f'git clone failed (exit code {_clone.returncode}):\\n{_err}{_hint}')
    print('Cloned', GITHUB_URL, '->', WORKDIR)
elif IN_LOCAL:
    raise SystemExit(
        f'WORKDIR ({WORKDIR}) does not look like the repo root (no khmer_tts/ found). '
        'Run this notebook from inside the repo, or set GITHUB_URL/DATASET_PATH.'
    )
else:
    raise SystemExit('Set GITHUB_URL or add the repo as a Dataset and set DATASET_PATH.')

os.chdir(WORKDIR)
print('cwd =', os.getcwd())
_commit = subprocess.run(['git', 'log', '--oneline', '-1'],
                          capture_output=True, text=True).stdout.strip()
print('Running commit:', _commit or '(not a git repo)')

try:
    import re as _re
    _nb_path = os.path.join(WORKDIR, 'kaggle', 'preprocess_khmer_vq_v2.ipynb')
    with open(_nb_path, encoding='utf-8') as _f:
        _m = _re.search(r'NOTEBOOK_REVISION\\s*=\\s*(\\d+)', _f.read())
    _repo_rev = int(_m.group(1)) if _m else None
except Exception as _e:
    _repo_rev = None
    print('(could not read the repo notebook revision:', _e, ')')

if _repo_rev is not None and _repo_rev > NOTEBOOK_REVISION:
    raise RuntimeError(
        f'STALE NOTEBOOK: this tab is revision {NOTEBOOK_REVISION}, but the repo '
        f'is at revision {_repo_rev} (commit {_commit}).\\n\\n'
        'Close this tab, re-open kaggle/preprocess_khmer_vq_v2.ipynb fresh from GitHub, '
        'and Run All again. Nothing on disk needs deleting.'
    )
elif _repo_rev is not None:
    print(f'Notebook revision {NOTEBOOK_REVISION} matches the repo -- tab is current.')

FISH_DIR = os.path.join(WORKDIR, 'vendor', 'fish-speech')
if not os.path.isdir(os.path.join(FISH_DIR, 'fish_speech')):
    shutil.rmtree(FISH_DIR, ignore_errors=True)
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/fishaudio/fish-speech', FISH_DIR], check=True)
print('Fish Speech at:', FISH_DIR)

for _patch_name in ('patch_fish_speech_tokenizer.py',
                     'patch_fish_speech_extract_vq.py',
                     'patch_fish_speech_dataloader.py'):
    _patch_script = os.path.join(WORKDIR, 'scripts', _patch_name)
    if not os.path.isfile(_patch_script):
        if _patch_name == 'patch_fish_speech_tokenizer.py':
            raise RuntimeError(
                f'{_patch_script} does not exist -- this tab is running stale code. '
                'Close it, re-open kaggle/preprocess_khmer_vq_v2.ipynb fresh from GitHub, '
                'and Run All again.'
            )
        continue
    _r = subprocess.run([sys.executable, _patch_script], capture_output=True, text=True)
    if _r.returncode != 0:
        raise RuntimeError(f'{_patch_name} failed (exit {_r.returncode}):\\n{_r.stdout}\\n{_r.stderr}')
    print(_r.stdout.strip())
"""))

cells.append(md("""\
## 3 - Install dependencies

Same verified flow as `train_khmer_base.ipynb` (Section 3) -- this notebook
only needs the VQ-extraction/protobuf-build half of it (no `lightning`
trainer run happens here, but installing the same full set keeps this in
sync with zero risk of a subtly different environment between the two
notebooks).
"""))

cells.append(code("""\
import importlib.util, os, re, subprocess, sys

_torch_ok = False
if importlib.util.find_spec('torch') is not None:
    import torch as _torch_probe
    _torch_ok = _torch_probe.cuda.is_available()
    del _torch_probe

def pip_install(args):
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + args,
                             capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f'pip install failed (exit code {result.returncode}) for: {args}\\n\\n'
            f'--- stdout (tail) ---\\n{result.stdout[-3000:]}\\n'
            f'--- stderr (tail) ---\\n{result.stderr[-3000:]}'
        )

_reqs = [l for l in open('requirements.txt')
         if not l.strip().lower().startswith('torchaudio')]
open('/tmp/requirements_no_torchaudio.txt', 'w').writelines(_reqs)

if _torch_ok:
    print('Working CUDA torch already present -- keeping it.')
else:
    pip_install(['torch', 'torchaudio'])
pip_install(['-r', '/tmp/requirements_no_torchaudio.txt'])

import torch
_torch_v = torch.__version__
_m = re.match(r'([0-9.]+)(?:\\+(\\w+))?', _torch_v)
_base_ver, _cuda_tag = _m.group(1), _m.group(2)
try:
    import torchaudio as _ta
    _ta_matches = _ta.__version__.split('+')[0] == _base_ver
except Exception:
    _ta_matches = False

if _ta_matches:
    print(f'torch {_torch_v} / torchaudio {_ta.__version__} already match')
else:
    _ta_args = ['--no-deps', '--force-reinstall', f'torchaudio=={_base_ver}']
    if _cuda_tag:
        _ta_args += ['--index-url', f'https://download.pytorch.org/whl/{_cuda_tag}']
    try:
        pip_install(_ta_args)
        print(f'torchaudio pinned to match torch {_torch_v}')
    except RuntimeError as _e:
        print(f'!! could not pin torchaudio=={_base_ver} -- continuing with '
              'whatever is installed.')
        print(_e)

try:
    _ta_major_minor = tuple(int(x) for x in _ta.__version__.split('+')[0].split('.')[:2])
except Exception:
    _ta_major_minor = (0, 0)
if _ta_major_minor >= (2, 9):
    try:
        pip_install(['torchcodec'])
        import torchcodec  # noqa: F401
        from torchcodec.decoders import AudioDecoder  # noqa: F401
        print('torchcodec available for torchaudio audio I/O')
    except Exception:
        print('!! torchcodec unavailable -- audio I/O falls back to soundfile, which is fine.')

pip_install(['huggingface_hub'])

subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'hf_xet'],
               capture_output=True, text=True)

pip_install(['--no-deps', '-e', 'vendor/fish-speech'])

_protobuf_pin = 'protobuf==4.25.5' if sys.version_info < (3, 14) else 'protobuf==6.33.6'
_required = [
    'hydra-core', 'loguru', 'natsort', 'einops', 'rich', 'lightning',
    'tensorboard', 'loralib', 'pyrootutils', 'einx[torch]', 'zstandard',
    'ormsgpack', 'tiktoken', 'cachetools', 'safetensors', 'kui',
    'transformers==4.56.1', _protobuf_pin,
]
_failed = []
for _pkg in _required:
    try:
        pip_install([_pkg])
    except RuntimeError as _e:
        _failed.append(_pkg)
        print(_e)

try:
    pip_install(['--no-deps', 'descript-audio-codec==1.0.0', 'descript-audiotools==0.7.2'])
except RuntimeError as _e:
    _failed.append('descript-audio-codec/audiotools')
    print(_e)
for _pkg in ['argbind', 'julius', 'pyloudnorm', 'ffmpy', 'flatten-dict',
             'markdown2', 'randomname', 'pystoi', 'torch-stoi',
             'importlib-resources', 'matplotlib']:
    try:
        pip_install([_pkg])
    except RuntimeError as _e:
        _failed.append(_pkg)
        print(_e)

if _failed:
    raise RuntimeError(
        f'These required packages failed to install: {_failed}\\n'
        'See the pip output above for each one.'
    )

# See train_khmer_base.ipynb Section 3 for why wandb gets uninstalled here.
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q', 'wandb'],
               capture_output=True)

_hydra_patch = os.path.join(WORKDIR, 'scripts', 'patch_hydra_py314_help.py')
if os.path.isfile(_hydra_patch):
    _hp = subprocess.run([sys.executable, _hydra_patch], capture_output=True, text=True)
    print(_hp.stdout.strip() or _hp.stderr.strip())
    if _hp.returncode != 0:
        raise RuntimeError(f'patch_hydra_py314_help.py failed:\\n{_hp.stdout}\\n{_hp.stderr}')

print('Dependencies installed.')
"""))

cells.append(md("## 4 - HF token and the base checkpoint"))

cells.append(code("""\
import os

HF_TOKEN = ''
if IN_KAGGLE:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret('HF_TOKEN')
        print('HF_TOKEN loaded from Kaggle Secrets.')
    except Exception as e:
        print('No HF_TOKEN in Kaggle Secrets:', e)
elif IN_COLAB:
    try:
        from google.colab import userdata
        HF_TOKEN = userdata.get('HF_TOKEN') or ''
        if HF_TOKEN:
            print('HF_TOKEN loaded from Colab Secrets.')
    except Exception as e:
        print('No HF_TOKEN in Colab Secrets:', e)

if not HF_TOKEN:
    HF_TOKEN = os.environ.get('HF_TOKEN', '')
    if HF_TOKEN:
        print('HF_TOKEN loaded from local environment.')

if not HF_TOKEN:
    try:
        from huggingface_hub import get_token
        HF_TOKEN = get_token() or ''
        if HF_TOKEN:
            print('HF_TOKEN loaded from cached `huggingface-cli login`.')
    except Exception:
        pass

if not HF_TOKEN:
    try:
        import getpass, warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error', getpass.GetPassWarning)
            HF_TOKEN = getpass.getpass(
                'Hugging Face token (input hidden -- press Enter to skip): ').strip()
        if HF_TOKEN:
            print('HF_TOKEN read from the prompt (not saved into this notebook).')
    except Exception as _e:
        print(f'Skipping the token prompt ({type(_e).__name__}: no interactive input available).')

if HF_TOKEN:
    os.environ['HF_TOKEN'] = HF_TOKEN
    os.environ['HUGGING_FACE_HUB_TOKEN'] = HF_TOKEN
    from huggingface_hub import login
    login(token=HF_TOKEN)
else:
    raise SystemExit(
        'No HF_TOKEN found. This notebook needs one to pull/push data shards -- '
        'add it as a secret and re-run.'
    )
"""))

cells.append(code("""\
# Base checkpoint -- extract_vq only needs codec.pth from it.
from huggingface_hub import snapshot_download
CKPT_DIR = os.path.join(WORKDIR, 'checkpoints', 'openaudio-s1-mini')
os.makedirs(CKPT_DIR, exist_ok=True)
snapshot_download(repo_id='fishaudio/openaudio-s1-mini', local_dir=CKPT_DIR,
                  token=os.environ.get('HF_TOKEN'))
print('Base checkpoint at:', CKPT_DIR)
print(os.listdir(CKPT_DIR))
"""))

cells.append(md("""\
## 5 - Preprocess every shard (resumable)

Works through each data shard: pull its raw audio, VQ-extract, pack into
protobufs, upload the result, delete local files, move on. Already-ready
shards (from a prior session) are skipped. Safe to just Run All again if a
session ends partway through.
"""))

cells.append(code("""\
import glob, os, shutil, subprocess

from khmer_tts.collab import (
    get_shard_count, get_ready_shards, is_val_ready,
    download_and_restore_shard, download_and_restore_val,
    pack_and_upload_ready_shard, pack_and_upload_ready_val,
)

_tok = os.environ.get('HF_TOKEN')

n_shards = get_shard_count(HF_DATA_REPO, DATA_CACHE_KEY, token=_tok)
if not n_shards:
    raise SystemExit(
        f'No SHARDED data found at {HF_DATA_REPO} under key {DATA_CACHE_KEY!r}. '
        'Run scripts/reshard_processed_data.py once (from a machine that already '
        'has the full processed data under data/fish/khmer_base) to build shards '
        'before running this notebook.'
    )

os.environ['EXTRACT_WORKERS'] = str(EXTRACT_WORKERS)
os.environ['EXTRACT_BATCH_SIZE'] = str(EXTRACT_BATCH_SIZE)

_log_path = 'outputs/logs/10a_preprocess.log'
os.makedirs('outputs/logs', exist_ok=True)

def run_10a(dataset_dir, proto_dir):
    '''Run scripts/10a on one dir; raise with the log tail on failure. Same
    stdbuf/tee treatment as train_khmer_base.ipynb Section 6 -- see there for
    why both are needed (buffering + Colab/Kaggle output truncation).'''
    os.environ['DATASET_DIR'] = dataset_dir
    os.environ['PROTO_DIR'] = proto_dir
    result = subprocess.run(
        ['bash', '-c',
         'set -o pipefail; stdbuf -oL -eL bash scripts/10a_extract_vq_and_build_protos.sh 2>&1 '
         f'| tee -a {_log_path}'],
        env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    if result.returncode != 0:
        _tail = ''
        if os.path.exists(_log_path):
            with open(_log_path, encoding='utf-8', errors='replace') as _f:
                _tail = _f.read()[-6000:]
        raise RuntimeError(
            f'10a_extract_vq_and_build_protos.sh failed (exit code {result.returncode}) '
            f'for DATASET_DIR={dataset_dir!r}.\\n\\n--- tail of {_log_path} ---\\n{_tail}'
        )

done = get_ready_shards(HF_DATA_REPO, DATA_CACHE_KEY, token=_tok)
print(f'{len(done)}/{n_shards} shard(s) already VQ-ready: {sorted(done)}')

# reshard_processed_data.py packs each shard's members using their ORIGINAL
# path (data/fish/khmer_base/<speaker>/<file>), not a per-shard path -- so
# download_and_restore_shard/download_and_restore_val always extract into
# these SAME two fixed directories no matter which shard/val was downloaded.
# Must be cleared before each download (every shard lands in the same place)
# and after each shard's processing (Kaggle's 20GB cap can't carry more than
# one shard's raw audio + protos at a time alongside the checkpoint).
RAW_SHARD_DIR = 'data/fish/khmer_base'
RAW_VAL_DIR = 'data/fish/khmer_base_val'

for i in range(n_shards):
    if i in done:
        print(f'-- shard {i}/{n_shards}: already ready, skipping --')
        continue
    print(f'-- shard {i}/{n_shards}: preprocessing --')
    proto_dir = f'data/fish/khmer_shards/shard{i}_protos'
    shutil.rmtree(RAW_SHARD_DIR, ignore_errors=True)
    shutil.rmtree(proto_dir, ignore_errors=True)
    ok = download_and_restore_shard(HF_DATA_REPO, DATA_CACHE_KEY, i, n_shards,
                                    WORKDIR, token=_tok)
    if not ok or not os.path.isdir(RAW_SHARD_DIR):
        raise SystemExit(f'Failed to download shard {i}/{n_shards} raw audio '
                         f'(expected it to land at {RAW_SHARD_DIR!r}).')
    run_10a(RAW_SHARD_DIR, proto_dir)
    pack_and_upload_ready_shard(HF_DATA_REPO, DATA_CACHE_KEY, i, n_shards,
                                WORKDIR, proto_dir, token=_tok, private=False)
    print(f'-- shard {i}/{n_shards}: uploaded and marked ready --')
    shutil.rmtree(RAW_SHARD_DIR, ignore_errors=True)
    shutil.rmtree(proto_dir, ignore_errors=True)

if is_val_ready(HF_DATA_REPO, DATA_CACHE_KEY, token=_tok):
    print('Validation set already VQ-ready, skipping.')
else:
    print('-- validation set: preprocessing --')
    val_proto_dir = 'data/fish/khmer_val_protos'
    shutil.rmtree(RAW_VAL_DIR, ignore_errors=True)
    shutil.rmtree(val_proto_dir, ignore_errors=True)
    ok = download_and_restore_val(HF_DATA_REPO, DATA_CACHE_KEY, WORKDIR, token=_tok)
    if not ok or not os.path.isdir(RAW_VAL_DIR):
        raise SystemExit(f'Failed to download the validation set raw audio '
                         f'(expected it to land at {RAW_VAL_DIR!r}).')
    run_10a(RAW_VAL_DIR, val_proto_dir)
    pack_and_upload_ready_val(HF_DATA_REPO, DATA_CACHE_KEY, WORKDIR, val_proto_dir,
                              token=_tok, private=False)
    print('-- validation set: uploaded and marked ready --')
    shutil.rmtree(RAW_VAL_DIR, ignore_errors=True)
    shutil.rmtree(val_proto_dir, ignore_errors=True)

done = get_ready_shards(HF_DATA_REPO, DATA_CACHE_KEY, token=_tok)
val_ready = is_val_ready(HF_DATA_REPO, DATA_CACHE_KEY, token=_tok)
if len(done) >= n_shards and val_ready:
    print(f'\\nAll {n_shards} shards + validation set are VQ-ready. '
          'You can now run train_khmer_base.ipynb on Kaggle.')
else:
    print(f'\\n{len(done)}/{n_shards} shards ready, val ready={val_ready}. '
          'Run this notebook again (new session if this one is out of GPU time) '
          'to continue.')
"""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = os.path.join(os.path.dirname(__file__), "..", "kaggle", "preprocess_khmer_vq_v2.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)
print("Wrote", os.path.abspath(out_path))
