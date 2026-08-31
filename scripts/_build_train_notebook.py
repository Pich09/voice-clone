#!/usr/bin/env python3
"""One-off generator for kaggle/train_khmer_base.ipynb. Run manually after
editing this file; not part of the pipeline. Keeps the notebook's JSON out of
a hand-edited diff -- easier to review as Python source than as raw ipynb.
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
# Khmer Base Model — Training Runner

Trains Stage 1 (the Khmer base TTS model) from the **already-preprocessed**
dataset cached on Hugging Face (`Panhapich/khmer-tts-processed`) — this
notebook does not run the data pipeline (scripts 01-09) itself.

Runs unmodified on **Kaggle**, **Google Colab**, or a **local machine** — the
environment is auto-detected in Section 1.

Designed to be run **across many separate sessions**: each run pulls the
`latest/` checkpoint from your Hugging Face model repo (`Panhapich/Tuna-TTS`),
trains `STAGE1_STEPS` more steps, then overwrites `latest/` with the result
(and `best/` too, if this session's validation loss is the new best) —
tagged in `metadata.json` with cumulative step count and validation loss.
Only the latest and best checkpoints are ever kept (not full history), to
minimize storage. Nothing is lost between sessions; just re-run this
notebook (Kaggle/Colab quotas reset regularly) to keep going.

**Setup**
1. Enable a GPU: Kaggle -> Settings > Accelerator > GPU T4 x2; Colab ->
   Runtime > Change runtime type > GPU.
2. Add an `HF_TOKEN` secret (Kaggle: Add-ons > Secrets; Colab: the key icon
   sidebar) — needed to push checkpoints to your `Panhapich/Tuna-TTS` repo.
3. Run All.
"""))

cells.append(md("## 0 - Configuration - edit this cell, then Run All"))

cells.append(code("""\
# ============================= CONFIG =============================
# Bumped whenever this notebook changes; Section 2 refuses to continue if a
# stale already-open tab is behind the repo's copy (see khmer_tts_kaggle.ipynb
# for why this matters -- opening a notebook from GitHub is a one-time
# snapshot, re-running cells never re-fetches it).
NOTEBOOK_REVISION = 1

# --- Where your repo code comes from ---
# Leave all three blank to run in-place (local machine, already inside the
# repo). On Kaggle/Colab, set GITHUB_URL (or DATASET_PATH for a Kaggle
# Dataset upload / no-internet session).
GITHUB_URL   = "https://github.com/Pich09/voice-clone.git"
DATASET_PATH = ""
WORKDIR      = ""   # "" = auto-pick per environment (see Section 2)

# --- Preprocessed training data (already uploaded, NOT rebuilt here) ---
# Produced by running scripts/01-09 once and packing the result with
# khmer_tts.collab.data_cache.pack_and_upload() -- see that module's
# docstring. This notebook only downloads and extracts it.
HF_DATA_REPO  = "Panhapich/khmer-tts-processed"
DATA_CACHE_KEY = "khmer_base_v1"

# --- Checkpoint store: resume/publish across sessions via your HF repo ---
# Keeps only two checkpoints in the repo -- `latest/` (always overwritten) and
# `best/` (overwritten only when a session's val_loss improves on it) -- plus
# metadata.json, instead of a full per-session history. See
# khmer_tts/collab/checkpoint_store.py.
BASE_CKPT_REPO   = "fishaudio/openaudio-s1-mini"  # free pretrained Fish Speech checkpoint
HF_CKPT_REPO     = "Panhapich/Tuna-TTS"           # your repo -- created automatically if missing
TRAINER_ID       = "panhapich"                    # just needs to be non-empty

# --- Run size ---
STAGE1_STEPS = 2000   # additional steps to train THIS session (target ~20000 total)
# =================================================================
print('Config loaded.')
"""))

cells.append(md("## 1 - Environment & GPU check"))

cells.append(code("""\
import subprocess, sys, os

# Colab is checked FIRST via its own markers, treated as authoritative -- a
# bare /kaggle directory is not a reliable Kaggle signal (kagglehub or a
# stray mkdir can leave one on a Colab VM too). See khmer_tts_kaggle.ipynb's
# Section 1 for the full reasoning; kept identical here.
IN_COLAB = bool(os.environ.get('COLAB_RELEASE_TAG')) or os.path.exists('/var/colab/hostname')
if IN_COLAB:
    IN_KAGGLE = False
else:
    IN_KAGGLE = ('KAGGLE_KERNEL_RUN_TYPE' in os.environ
                 or 'KAGGLE_URL_BASE' in os.environ
                 or os.path.isdir('/kaggle/input'))
IN_LOCAL = not IN_KAGGLE and not IN_COLAB
ENV_NAME = 'Colab' if IN_COLAB else ('Kaggle' if IN_KAGGLE else 'Local')
print('Environment:', ENV_NAME)

import torch
print('Python :', sys.version.split()[0])
print('Torch  :', torch.__version__)
print('CUDA   :', torch.cuda.is_available())

HAS_TPU = bool(os.environ.get('TPU_NAME') or os.environ.get('XRT_TPU_CONFIG')
               or os.environ.get('COLAB_TPU_ADDR'))

if torch.cuda.is_available():
    ACCELERATOR = 'gpu'
    print('GPU    :', torch.cuda.get_device_name(0), f'({torch.cuda.device_count()} device(s))')
    print(subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.free','--format=csv'],
                         capture_output=True, text=True).stdout)
elif HAS_TPU:
    ACCELERATOR = 'tpu'
    print('TPU detected, no CUDA GPU. UNSUPPORTED / EXPERIMENTAL -- see '
          'khmer_tts_kaggle.ipynb Section 1 for why this may still fail inside '
          "Fish Speech's own model code. Proceeding anyway.")
else:
    ACCELERATOR = 'cpu'
    print('!! No GPU or TPU detected -- training will be impractically slow. '
          'Enable the GPU/TPU accelerator in this platform\\'s settings.')

print('trainer.accelerator ->', ACCELERATOR)

# Per-GPU memory knobs sized from actual VRAM (not a hardcoded Kaggle T4) --
# see khmer_tts_kaggle.ipynb Section 1 for the full tier rationale.
if ACCELERATOR == 'gpu' and torch.cuda.is_available():
    VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
else:
    VRAM_GB = 0.0

if ACCELERATOR == 'cpu':
    TRAIN_BATCH_SIZE, TRAIN_MAX_LENGTH = 1, 512
elif not VRAM_GB:
    TRAIN_BATCH_SIZE, TRAIN_MAX_LENGTH = 4, 4096       # TPU
elif VRAM_GB < 6:
    TRAIN_BATCH_SIZE, TRAIN_MAX_LENGTH = 1, 512
elif VRAM_GB < 12:
    TRAIN_BATCH_SIZE, TRAIN_MAX_LENGTH = 1, 1024
elif VRAM_GB < 20:
    TRAIN_BATCH_SIZE, TRAIN_MAX_LENGTH = 2, 2048
else:
    TRAIN_BATCH_SIZE, TRAIN_MAX_LENGTH = 4, 4096
GRAD_ACCUM = max(1, 4 // TRAIN_BATCH_SIZE)

# extract_vq has its own memory profile (codec + padded raw waveforms) --
# see khmer_tts_kaggle.ipynb Section 1 for measured sizing.
if ACCELERATOR == 'cpu':
    EXTRACT_WORKERS, EXTRACT_BATCH_SIZE = 1, 4
elif not VRAM_GB:
    EXTRACT_WORKERS, EXTRACT_BATCH_SIZE = 1, 8
elif VRAM_GB < 6:
    EXTRACT_WORKERS, EXTRACT_BATCH_SIZE = 1, 4
elif VRAM_GB < 12:
    EXTRACT_WORKERS, EXTRACT_BATCH_SIZE = 1, 8
else:
    EXTRACT_WORKERS, EXTRACT_BATCH_SIZE = 4, 16
print(f'extract : {EXTRACT_WORKERS} worker(s), batch {EXTRACT_BATCH_SIZE}')

# Python 3.14+ switched Linux's multiprocessing default from fork to
# forkserver, which cannot pickle fish-speech's tiktoken-backed tokenizer --
# see khmer_tts_kaggle.ipynb Section 1. num_workers=0 is correct everywhere,
# just slower, so use it whenever the start method isn't fork.
import multiprocessing as _mp
TRAIN_NUM_WORKERS = 4 if _mp.get_start_method(allow_none=False) == 'fork' else 0
if TRAIN_NUM_WORKERS == 0:
    print('workers: multiprocessing start method is not fork -> data.num_workers=0')

print(f'sizing : batch {TRAIN_BATCH_SIZE}, max_length {TRAIN_MAX_LENGTH}, '
      f'grad_accum {GRAD_ACCUM}' + (f', VRAM {VRAM_GB:.1f}GB' if VRAM_GB else ''))


def run_step(args, **kwargs):
    \"\"\"Run a pipeline script with THIS interpreter, raising loudly on failure
    instead of the shell-magic `!python` pattern (which swallows a nonzero
    exit code and lets the notebook march on into a confusing later failure).
    \"\"\"
    cmd = [sys.executable] + [str(a) for a in args]
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(
            f'{args[0]} failed (exit code {result.returncode}) -- see the '
            'output above for the real error.'
        )
    return result
"""))

cells.append(code("""\
import shutil as _shutil

def disk_free_gb(path='.'):
    return _shutil.disk_usage(path).free / 1e9

def report_disk(label=''):
    total, used, free = _shutil.disk_usage('.')
    print(f'[disk{" - " + label if label else ""}] used={used/1e9:.1f}GB '
          f'free={free/1e9:.1f}GB (of {total/1e9:.1f}GB)')

report_disk('session start')
"""))

cells.append(md("## 2 - Get the code (your repo + Fish Speech engine)"))

cells.append(code("""\
import os, shutil, subprocess, sys

# 2a. Resolve WORKDIR for the detected environment if left blank.
if not WORKDIR:
    if IN_KAGGLE:
        WORKDIR = '/kaggle/working/khmer-voice-clone'
    elif IN_COLAB:
        # Deliberately NOT Drive. This notebook pulls the preprocessed
        # dataset (~22GB+) and the checkpoint fresh from Hugging Face every
        # session and publishes results back there too -- nothing here needs
        # to survive a disconnect, which is the only reason to pay Drive's
        # cost. And that cost is real: free-tier Drive's 15GB total quota is
        # smaller than just the dataset download alone, so mounting Drive as
        # WORKDIR fills it immediately and every download/training step
        # after that fails with "No space left on device". Colab's local
        # /content disk is far larger (70GB+) and exactly matches this
        # notebook's own "nothing persists, everything round-trips through
        # HF" design -- use it instead.
        WORKDIR = '/content/khmer-voice-clone'
    else:
        # Local: walk UP from cwd looking for the repo root, so this works
        # whether the notebook's cwd is the repo root or kaggle/ (Jupyter
        # front-ends disagree on this -- VS Code defaults to the notebook's
        # own folder).
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

# 2b. Bring in your repo -> WORKDIR
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

# Stale-tab guard, same idea as khmer_tts_kaggle.ipynb -- this notebook's own
# revision, checked against the just-pulled repo copy of ITSELF.
try:
    import re as _re
    _nb_path = os.path.join(WORKDIR, 'kaggle', 'train_khmer_base.ipynb')
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
        'Close this tab, re-open kaggle/train_khmer_base.ipynb fresh from GitHub, '
        'and Run All again. Nothing on disk needs deleting.'
    )
elif _repo_rev is not None:
    print(f'Notebook revision {NOTEBOOK_REVISION} matches the repo -- tab is current.')

# 2c. Clone the Fish Speech engine into vendor/
FISH_DIR = os.path.join(WORKDIR, 'vendor', 'fish-speech')
if not os.path.isdir(os.path.join(FISH_DIR, 'fish_speech')):
    shutil.rmtree(FISH_DIR, ignore_errors=True)
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/fishaudio/fish-speech', FISH_DIR], check=True)
print('Fish Speech at:', FISH_DIR)

# 2d-f. Idempotent upstream patches -- see each script's own docstring for
# the exact bug it works around. Identical to khmer_tts_kaggle.ipynb Section 2.
for _patch_name in ('patch_fish_speech_tokenizer.py',
                     'patch_fish_speech_extract_vq.py',
                     'patch_fish_speech_dataloader.py'):
    _patch_script = os.path.join(WORKDIR, 'scripts', _patch_name)
    if not os.path.isfile(_patch_script):
        if _patch_name == 'patch_fish_speech_tokenizer.py':
            raise RuntimeError(
                f'{_patch_script} does not exist -- this tab is running stale code. '
                'Close it, re-open kaggle/train_khmer_base.ipynb fresh from GitHub, '
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

Same verified flow as `khmer_tts_kaggle.ipynb` (Section 3) -- see that
notebook if you need the detailed reasoning for any one step.
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

_hydra_patch = os.path.join(WORKDIR, 'scripts', 'patch_hydra_py314_help.py')
if os.path.isfile(_hydra_patch):
    _hp = subprocess.run([sys.executable, _hydra_patch], capture_output=True, text=True)
    print(_hp.stdout.strip() or _hp.stderr.strip())
    if _hp.returncode != 0:
        raise RuntimeError(f'patch_hydra_py314_help.py failed:\\n{_hp.stdout}\\n{_hp.stderr}')

print('Dependencies installed.')
"""))

cells.append(md("## 4 - HF token, base checkpoint, and the preprocessed dataset"))

cells.append(code("""\
import os

# HF token, tried in order: Kaggle Secrets -> Colab Secrets -> local env var ->
# cached `huggingface-cli login` credential -> interactive prompt (getpass,
# never input() -- input() would echo the token into the saved .ipynb).
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
        'No HF_TOKEN found. This notebook needs one to pull the preprocessed '
        'dataset (if private) and to push checkpoints to your HF_CKPT_REPO -- '
        'add it as a secret and re-run.'
    )
"""))

cells.append(code("""\
# Base pretrained Fish Speech checkpoint (frozen -- codec.pth always comes
# from here even when resuming training from a later checkpoint).
from huggingface_hub import snapshot_download
CKPT_DIR = os.path.join(WORKDIR, 'checkpoints', 'openaudio-s1-mini')
os.makedirs(CKPT_DIR, exist_ok=True)
snapshot_download(repo_id=BASE_CKPT_REPO, local_dir=CKPT_DIR, token=os.environ.get('HF_TOKEN'))
print('Base checkpoint at:', CKPT_DIR)
print(os.listdir(CKPT_DIR))
"""))

cells.append(code("""\
# Pull the preprocessed training data (built once locally by running
# scripts/01-09, then uploaded with khmer_tts.collab.data_cache.pack_and_upload
# -- see that module's docstring). This notebook does NOT rebuild it.
from khmer_tts.collab import download_and_restore

report_disk('before data download')
ok = download_and_restore(HF_DATA_REPO, DATA_CACHE_KEY, WORKDIR, token=os.environ.get('HF_TOKEN'))
if not ok:
    raise SystemExit(
        f'No preprocessed data found at {HF_DATA_REPO} under key {DATA_CACHE_KEY!r}. '
        'Run the data pipeline (scripts/01-09) somewhere first, then '
        'khmer_tts.collab.data_cache.pack_and_upload(HF_DATA_REPO, DATA_CACHE_KEY, ...) '
        'to publish it -- this notebook only trains, it does not preprocess.'
    )
print('Preprocessed data restored:')
for _p in ('data/fish/khmer_base', 'data/fish/khmer_base_val',
           'data/manifests/ddd_train.jsonl', 'data/manifests/ddd_valid.jsonl'):
    print(' ', _p, '- present' if os.path.exists(_p) else '- MISSING')
report_disk('after data download')
"""))

cells.append(md("""\
## 5 - Pull the most-trained checkpoint

Every session resumes from wherever the LAST session (yours, from any
device) left off -- the `latest/` checkpoint in `Panhapich/Tuna-TTS`. The
very first run finds nothing and starts from the base pretrained checkpoint
instead. (Resuming always uses `latest`, never `best` -- resuming from
`best` can stall progress indefinitely: one session whose val_loss ticks up
from ordinary noise would mean every later session restarts from the same
older checkpoint and cumulative steps stop growing. `best/` is kept purely
as a separate, safe-to-ship reference.)
"""))

cells.append(code("""\
from khmer_tts.collab import CheckpointStore

store = CheckpointStore(HF_CKPT_REPO, token=os.environ.get('HF_TOKEN'))
store.ensure_repo()

ok, lk = store.acquire_lock(TRAINER_ID, ttl_sec=6 * 3600)
if not ok:
    raise SystemExit(
        f\"Repo is locked by {lk.get('owner')} until it expires -- if that's an \"
        \"earlier session of yours that crashed without releasing it, just wait \"
        \"for it to expire (6h) or acquire it again from that session.\"
    )

meta = store.load_metadata()
print('best so far  :', f\"step {meta.get('best_step')} val_loss={meta.get('best_val_loss')}\"
      if meta.get('best_step') is not None else '(none yet)')
print('latest so far:', f\"step {meta.get('latest_step')} val_loss={meta.get('latest_val_loss')}\"
      if meta.get('latest_step') is not None else '(none yet)')

RESUME_DIR = 'checkpoints/_resume'
FOUND = store.pull_latest(RESUME_DIR)
RESUME_CKPT = RESUME_DIR if FOUND else None
# Cumulative step count continues from whatever we actually resumed FROM.
RESUME_STEP = int(meta.get('latest_step') or 0) if FOUND else 0
print('resuming from:', RESUME_CKPT or '(none yet -- starting from the base checkpoint)',
      f'at step {RESUME_STEP}')
"""))

cells.append(md("## 6 - Train"))

cells.append(code("""\
import os, subprocess
# scripts/10 reads these from the environment -- no editing the script in
# place (in-place edits dirty the git tree and break Section 2's `git pull`
# the next time this WORKDIR is reused).
os.environ['STAGE1_MAX_STEPS'] = str(RESUME_STEP + STAGE1_STEPS)
os.environ.setdefault('TRAIN_ACCELERATOR', ACCELERATOR)
if ACCELERATOR == 'gpu':
    os.environ.setdefault('TRAIN_BATCH_SIZE', str(TRAIN_BATCH_SIZE))
    os.environ.setdefault('TRAIN_MAX_LENGTH', str(TRAIN_MAX_LENGTH))
    os.environ.setdefault('GRAD_ACCUM', str(GRAD_ACCUM))
os.environ.setdefault('TRAIN_NUM_WORKERS', str(TRAIN_NUM_WORKERS))
os.environ.setdefault('EXTRACT_WORKERS', str(EXTRACT_WORKERS))
os.environ.setdefault('EXTRACT_BATCH_SIZE', str(EXTRACT_BATCH_SIZE))
if RESUME_CKPT:
    # Resume from the pulled merged checkpoint (model.pth + config.json +
    # tokenizer). Only PRETRAINED_CKPT moves -- codec.pth still comes from
    # the base checkpoint dir, since merged checkpoints don't ship a codec.
    os.environ['PRETRAINED_CKPT'] = RESUME_CKPT
else:
    os.environ.pop('PRETRAINED_CKPT', None)

# NOTE: STAGE1_MAX_STEPS above is the CUMULATIVE target step count fish-speech's
# trainer.max_steps stops at, not "how many steps this session runs" -- Lightning
# resumes its own internal step counter from the checkpoint being fine-tuned
# from is NOT tracked here (LoRA fine-tuning restarts the optimizer state each
# session), so this notebook treats every session as training STAGE1_STEPS
# fresh steps on top of the resumed weights, and moves the cumulative counter
# (used for step numbering / registry metadata) forward by that same amount.
report_disk('before training')
# Tee to a log file (in addition to streaming into this cell) -- Colab
# collapses/truncates very long cell output, which can hide the actual error
# above a generic "exit code 1".
_log_path = 'outputs/logs/10_train.log'
os.makedirs('outputs/logs', exist_ok=True)
result = subprocess.run(
    ['bash', '-c', 'set -o pipefail; bash scripts/10_train_fish_khmer_base.sh 2>&1 '
                    f'| tee {_log_path}'])
if result.returncode != 0:
    # Read the log file back and put its tail directly INTO the exception --
    # don't make the user go dig for it in a separate cell, since Colab's
    # collapsed-output UI is exactly what hides it in the first place.
    if os.path.exists(_log_path):
        with open(_log_path, encoding='utf-8', errors='replace') as _f:
            _tail = _f.read()[-6000:]
    else:
        _tail = f'(no log file was created at {_log_path} -- the script ' \\
                'failed before it produced any output at all)'
    raise RuntimeError(
        f'10_train_fish_khmer_base.sh failed (exit code {result.returncode}).\\n\\n'
        f'--- tail of {_log_path} ---\\n{_tail}'
    )

# scripts/10's own Step 4 already merged the LoRA delta into
# models/khmer_base/merged -- the only checkpoint anything downstream reads.
# The source wav/lab files, VQ token sidecars, and packed protobuf shards are
# dead weight after this point (re-downloaded fresh from HF_DATA_REPO next
# session). results/khmer_base/csv/ is kept -- Section 7 reads val_loss from it.
for _p in ('data/fish/khmer_base', 'data/fish/khmer_base_protos',
           'data/fish/khmer_base_val', 'data/fish/khmer_base_val_protos',
           'results/khmer_base/checkpoints'):
    if os.path.isdir(_p):
        shutil.rmtree(_p)
report_disk('after training cleanup')
"""))

cells.append(md("""\
## 7 - Publish the checkpoint

Overwrites `latest/` in `Panhapich/Tuna-TTS` with what this session trained
(and `best/` too, if this session's val_loss is the new best), and updates
`metadata.json` at the repo root. Only these two checkpoints + the metadata
file are ever stored -- no per-session history -- to keep the repo small.
"""))

cells.append(code("""\
from khmer_tts.collab import read_val_loss

new_step = RESUME_STEP + STAGE1_STEPS
vloss = read_val_loss('results/khmer_base') or read_val_loss('models/khmer_base')
print('publishing: step', new_step, 'val_loss', vloss)

# Publish the MERGED checkpoint (model.pth + config.json + tokenizer) --
# models/khmer_base itself is just hydra's run dir. store.publish() overwrites
# latest/ unconditionally and best/ only if vloss improves on the recorded
# best, then rewrites metadata.json to match.
metadata = store.publish('models/khmer_base/merged', step=new_step, val_loss=vloss,
                         trainer_id=TRAINER_ID)
store.release_lock(TRAINER_ID)
print('metadata:', metadata)
print(f'https://huggingface.co/{HF_CKPT_REPO}')
"""))

cells.append(md("""\
## 8 - Sanity check (optional)

Synthesizes the 10 Khmer test sentences with the checkpoint just trained,
plays the first few inline, and uploads the batch to your HF repo under
`eval/step_XXXXXXX/` so you can compare sessions over time.
"""))

cells.append(code("""\
_tag = f'step_{new_step:07d}'
_eval_dir = f'outputs/eval/khmer_base_{_tag}'
run_step(['scripts/13_generate_eval_samples.py', '--model_dir', 'models/khmer_base/merged',
          '--speaker', 'default', '--sentences', 'eval/test_sentences_km.txt',
          '--out_dir', _eval_dir])

import IPython.display as ipd, glob
for w in sorted(glob.glob(_eval_dir + '/*.wav'))[:3]:
    print(w); ipd.display(ipd.Audio(w))

store.api.upload_folder(repo_id=HF_CKPT_REPO, repo_type='model',
                        folder_path=_eval_dir, path_in_repo=f'eval/{_tag}',
                        commit_message=f'eval samples {_tag}')
print(f'Uploaded eval samples to https://huggingface.co/{HF_CKPT_REPO}/tree/main/eval/{_tag}')
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

out_path = os.path.join(os.path.dirname(__file__), "..", "kaggle", "train_khmer_base.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)
print("Wrote", os.path.abspath(out_path))
