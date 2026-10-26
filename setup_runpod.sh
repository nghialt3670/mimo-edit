#!/usr/bin/env bash
# One-time RunPod environment setup for the mixed-prompt evaluation project.
# Does NOT run the evaluation and does NOT download large model weights.
# Usage: bash setup_runpod.sh [project_dir]
set -Eeuo pipefail

PROJECT_DIR="${1:-/workspace/mixed-prompt}"
PYTHON_REQUEST="3.13.11"
TORCH_VERSION="2.11.0"
TORCHVISION_VERSION="0.26.0"
TORCHAUDIO_VERSION="2.11.0"

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\n\033[1;33mWARNING: %s\033[0m\n' "$*" >&2; }
die()  { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }
trap 'warn "Setup stopped on line $LINENO. Existing .out/, repository data, and model caches were not deleted."' ERR

[[ "$(uname -s)" == "Linux" ]] || die "Run this script INSIDE your RunPod Linux pod, not on macOS."
[[ -f "$PROJECT_DIR/pyproject.toml" ]] || die "Missing $PROJECT_DIR/pyproject.toml. Pass the path to your cloned repo: bash setup_runpod.sh /workspace/mixed-prompt"
command -v python3 >/dev/null || die 'System python3 is required.'
command -v curl >/dev/null || die 'curl is required.'
command -v nvidia-smi >/dev/null || die 'nvidia-smi unavailable. Start a GPU pod before running setup.'

log "Checking GPU and disk storage"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || die 'GPU driver is not available.'
CUDA_DRIVER_CAP="$(nvidia-smi | sed -nE 's/.*CUDA Version: ([0-9]+\.[0-9]+).*/\1/p' | head -n1)"
if [[ -n "$CUDA_DRIVER_CAP" ]]; then
    printf 'Driver-reported max CUDA version: %s\n' "$CUDA_DRIVER_CAP"
    python3 - "$CUDA_DRIVER_CAP" <<'PY'
import sys
cap = tuple(map(int, sys.argv[1].split('.')))
if cap < (12, 8):
    raise SystemExit('ERROR: This RunPod host driver does not support CUDA 12.8. Use another pod/host with a newer NVIDIA driver.')
PY
else
    warn 'Could not determine driver-supported CUDA version; GPU test later will verify compatibility.'
fi

df -h "$PROJECT_DIR" /workspace
FREE_GB="$(df -Pk /workspace | awk 'NR==2 {print int($4/1048576)}')"
if (( FREE_GB < 15 )); then
    die "Only ~${FREE_GB} GiB free in /workspace. Increase volume storage (100-150 GB recommended) before installing large dependencies."
elif (( FREE_GB < 80 )); then
    warn "Only ~${FREE_GB} GiB free in /workspace. Setup may fit, but the three Hugging Face model caches and results may not. Aim for 100-150 GB."
fi

log 'Configuring persistent storage-backed caches'
mkdir -p /workspace/.cache/huggingface/{hub,xet} /workspace/.cache/uv /workspace/.cache/uv-python \
         /workspace/.cache/torch /workspace/tmp
umask 077
ENV_FILE="$PROJECT_DIR/.runpod_env.sh"
cat > "$ENV_FILE" <<'EOF'
#!/usr/bin/env bash
# Source in a new shell: source /workspace/mixed-prompt/.runpod_env.sh
export PATH="$HOME/.local/bin:$PATH"
export XDG_CACHE_HOME=/workspace/.cache
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export TORCH_HOME=/workspace/.cache/torch
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_PYTHON_INSTALL_DIR=/workspace/.cache/uv-python
export TMPDIR=/workspace/tmp
EOF
# Make the env file path independent of the project folder.
chmod 600 "$ENV_FILE"
if ! grep -Fq "source \"$ENV_FILE\"" "$HOME/.bashrc" 2>/dev/null; then
    printf '\n# mixed-prompt RunPod environment\n[ -f "%s" ] && source "%s"\n' "$ENV_FILE" "$ENV_FILE" >> "$HOME/.bashrc"
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

# Preserve a previously cached Hugging Face login token, if any.
if [[ ! -f "$HF_HOME/token" && -f "$HOME/.cache/huggingface/token" ]]; then
    cp "$HOME/.cache/huggingface/token" "$HF_HOME/token"
    chmod 600 "$HF_HOME/token"
fi

log 'Installing modern uv (bypassing the outdated system copy)'
curl -LsSf https://astral.sh/uv/install.sh | sh
UV_BIN="$HOME/.local/bin/uv"
[[ -x "$UV_BIN" ]] || die "Cannot find installed uv at $UV_BIN"
"$UV_BIN" --version

log 'Installing Python 3.13.11 (or a fixed 3.13 patch version)'
if "$UV_BIN" python install "$PYTHON_REQUEST"; then
    PYTHON_SPEC="$PYTHON_REQUEST"
else
    warn 'Exact Python 3.13.11 unavailable; trying the newest available Python 3.13 patch.'
    "$UV_BIN" python install 3.13
    PYTHON_SPEC="3.13"
fi
MANAGED_PYTHON="$($UV_BIN python find "$PYTHON_SPEC")"
"$MANAGED_PYTHON" - <<'PY'
import sys
print('Managed Python:', sys.version.split()[0])
if sys.version_info[:3] < (3, 13, 9) or sys.version_info[:2] != (3, 13):
    raise SystemExit('ERROR: This interpreter is not a suitable Python 3.13.9+ build.')
PY

cd "$PROJECT_DIR"
for name in pyproject.toml uv.lock .python-version; do
    if [[ -f "$name" && ! -f "$name.before-runpod" ]]; then
        cp -p "$name" "$name.before-runpod"
    fi
done

log 'Pinning compatible PyTorch/CUDA wheels in pyproject.toml'
python3 - "$PROJECT_DIR/pyproject.toml" <<'PY'
import pathlib, re, sys, tomllib
path = pathlib.Path(sys.argv[1])
s = path.read_text()
original = tomllib.loads(s)
if 'project' not in original or 'dependencies' not in original['project']:
    raise SystemExit('Unsupported pyproject.toml: no [project].dependencies')
# uv locks for the declared Python range. Restrict to the installed 3.13 line.
s, py_count = re.subn(r'(?m)^requires-python\s*=\s*"[^"]+"\s*$',
                       'requires-python = ">=3.13,<3.14"', s)
if py_count != 1:
    raise SystemExit('Cannot safely update requires-python; edit manually.')
versions = {'torch': '2.11.0', 'torchvision': '0.26.0', 'torchaudio': '2.11.0'}
# Replace constraints in both core and optional dependencies. This project
# uses one dependency per line, which keeps the rewrite narrowly scoped.
for pkg, version in versions.items():
    pattern = rf'"{pkg}(?:[<>=!~][^"\n]*)?"'
    s, count = re.subn(pattern, f'"{pkg}=={version}"', s)
    if count == 0:
        raise SystemExit(f'Could not find a "{pkg}..." dependency; edit pyproject.toml manually.')

header = '[tool.uv.sources]'
if header not in s:
    s = s.rstrip() + '\n\n' + header + '\n'
lines = s.splitlines(keepends=True)
start = next(i for i, line in enumerate(lines) if line.strip() == header)
end = next((i for i in range(start+1, len(lines)) if lines[i].lstrip().startswith('[')), len(lines))
section = lines[start+1:end]
cleaned = []
i = 0
while i < len(section):
    line = section[i]
    if re.match(r'^\s*(torch|torchvision|torchaudio)\s*=', line):
        # Remove one existing inline/table/array source value, including a
        # multiline array of markers, without removing unrelated sources.
        value = line.split('=', 1)[1]
        depth = value.count('[') + value.count('{') - value.count(']') - value.count('}')
        i += 1
        while depth > 0 and i < len(section):
            value = section[i]
            depth += value.count('[') + value.count('{') - value.count(']') - value.count('}')
            i += 1
        if depth != 0:
            raise SystemExit('Unsupported multiline torch source entry in pyproject.toml')
        continue
    cleaned.append(line)
    i += 1
new_sources = ''.join(
    f'{pkg} = [{{ index = "pytorch-cu128", marker = "sys_platform == \'linux\'" }}]\n'
    for pkg in versions
)
if cleaned and not cleaned[-1].endswith('\n'):
    cleaned.append('\n')
lines[start+1:end] = cleaned + [new_sources]
s = ''.join(lines)

# Add the named explicit index only once.
parsed = tomllib.loads(s)
indexes = parsed.get('tool', {}).get('uv', {}).get('index', [])
matching = [v for v in indexes if v.get('name') == 'pytorch-cu128']
if matching:
    if any(v.get('url') != 'https://download.pytorch.org/whl/cu128' or not v.get('explicit', False) for v in matching):
        raise SystemExit('Existing pytorch-cu128 index has incompatible settings; edit manually.')
else:
    s = s.rstrip() + '''\n\n[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
'''
# Validate before writing any changes.
parsed = tomllib.loads(s)
for pkg in versions:
    assert pkg in parsed['tool']['uv']['sources'], pkg
path.write_text(s)
print('Pinned Python 3.13 and torch 2.11.0 / torchvision 0.26.0 / torchaudio 2.11.0 (Linux: cu128).')
print('Original files backed up to *.before-runpod')
PY

log 'Recreating isolated Python environment (old .venv is disposable)'
"$UV_BIN" python pin "$PYTHON_SPEC"
# Only remove .venv (installed packages), NOT dataset, repo, .out, or model cache.
if [[ -d "$PROJECT_DIR/.venv" ]]; then
    rm -rf "$PROJECT_DIR/.venv"
fi

log 'Resolving project dependencies and installing from uv.lock'
# The lockfile is intentionally regenerated because the torch pins changed.
"$UV_BIN" lock
"$UV_BIN" sync --python "$PYTHON_SPEC" --locked

log 'Testing PyTorch, CUDA initialization, and an actual GPU tensor'
env -u LD_LIBRARY_PATH "$PROJECT_DIR/.venv/bin/python" - <<'PY'
import sys, torch, torchvision, torchaudio
print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__)
print('Torchvision:', torchvision.__version__)
print('Torchaudio:', torchaudio.__version__)
print('CUDA build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('ERROR: CUDA is still unavailable; check nvidia-smi and container GPU configuration.')
print('GPU:', torch.cuda.get_device_name(0))
x = torch.randn((512, 512), device='cuda')
y = x @ x
print('GPU computation successful:', y.device, y.shape)
PY

log 'Checking existing Hugging Face credentials and gated SAM3 access'
# This downloads only the tiny config.json if authenticated, not model weights.
# An absent/restricted HF_TOKEN is reported but does not abort CUDA setup.
env -u LD_LIBRARY_PATH "$PROJECT_DIR/.venv/bin/python" - "$PROJECT_DIR" <<'PY'
from pathlib import Path
import os, sys
from dotenv import dotenv_values
from huggingface_hub import hf_hub_download
project = Path(sys.argv[1]); home = Path(os.environ['HF_HOME'])
configured = dotenv_values(project / '.env') if (project / '.env').exists() else {}
token = os.environ.get('HF_TOKEN') or configured.get('HF_TOKEN')
if not token and (home / 'token').exists():
    token = (home / 'token').read_text().strip()
if not token:
    print('HF_TOKEN missing. Add HF_TOKEN=hf_... to /workspace/mixed-prompt/.env or log in using `hf auth login`.')
else:
    path = home / 'token'
    path.write_text(token.strip() + '\n')
    path.chmod(0o600)
    try:
        config = hf_hub_download('facebook/sam3', 'config.json', token=token)
        print('SAM3 gated access verified:', config)
    except Exception as exc:
        print(f'SAM3 access NOT verified ({type(exc).__name__}). Confirm approved access to facebook/sam3 and token permissions.')
PY

log 'Checking (but not modifying) existing annotation input'
"$PROJECT_DIR/.venv/bin/python" - "$PROJECT_DIR" <<'ANNOTATION_PY'
from pathlib import Path
import json, sys
root = Path(sys.argv[1])
rel = Path('annotation/compbench/run_20261007_000000_000000/compbench_local_multiobject_mixed.jsonl')
for prefix in ('out', '.out'):
    path = root / prefix / rel
    if path.exists():
        count = 0
        with path.open(encoding='utf-8') as f:
            for count, line in enumerate(f, 1):
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    raise SystemExit(f'Invalid JSONL at {path}:{count}: {e}')
        print(f'Annotation valid ({count} records): {path}')
        break
else:
    print('Annotation not found under out/ or .out/ — copy it before starting evaluation if needed.')
ANNOTATION_PY

log 'Setup complete — NO evaluation has been started'
printf '\nNext, in your RunPod terminal:\n'
printf '  source %q\n' "$ENV_FILE"
printf '  cd %q\n' "$PROJECT_DIR"
printf '  env -u LD_LIBRARY_PATH %q run --no-sync src/scripts/compbench_inference_mixed.py\n' "$UV_BIN"
printf '\nModel downloads will use: %s\n' "$HF_HOME"
printf 'Your existing annotations and previous results were not modified.\n'
