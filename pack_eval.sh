
START_DIR="$(pwd -P)"
OUT_DIR="${EVAL_OUT_DIR:-$START_DIR/.out}"
ARCHIVE="${EVAL_ARCHIVE:-$START_DIR/results.tar.gz}"
case "$OUT_DIR" in /*) ;; *) OUT_DIR="$START_DIR/$OUT_DIR" ;; esac
case "$ARCHIVE" in /*) ;; *) ARCHIVE="$START_DIR/$ARCHIVE" ;; esac
mkdir -p "$OUT_DIR" "$(dirname "$ARCHIVE")"

WORK_DIR="$(mktemp -d "$START_DIR/.eval_runtime.XXXXXXXX")"
finish() {
  local status=$?
  trap - EXIT
  printf '%s\n' "$status" > "$OUT_DIR/exit_code.txt"
  tar -czf "$ARCHIVE" -C "$(dirname "$OUT_DIR")" "$(basename "$OUT_DIR")" || true
  if [[ "${EVAL_KEEP_WORKDIR:-0}" != "1" ]]; then
    rm -rf "$WORK_DIR"
  else
    echo "Preserved extracted project: $WORK_DIR"
  fi
  echo "Output folder: $OUT_DIR"
  echo "Result archive: $ARCHIVE"
  exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Extract data following the exact marker, without external archive utilities.
PAYLOAD_LINE="$(awk '/^__EVAL_PAYLOAD_BELOW__$/ {print NR+1; exit}' "$0")"
if [[ -z "$PAYLOAD_LINE" ]]; then
  echo "ERROR: Bundle payload marker missing." >&2
  exit 3
fi
tail -n "+$PAYLOAD_LINE" "$0" | tar -xz -C "$WORK_DIR"

# The bundle contains exactly one file under .out: the input annotation.
# Move a copy into the persistent output directory BEFORE replacing .out with
# a symlink, so the evaluation sees the same relative input path as locally.
ANNOTATION_REL=".out/annotation/compbench/run_20261007_000000_000000/compbench_local_multiobject_mixed.jsonl"
INPUT_ANNOTATION="$WORK_DIR/$ANNOTATION_REL"
OUTPUT_ANNOTATION="$OUT_DIR/${ANNOTATION_REL#.out/}"
if [[ ! -f "$INPUT_ANNOTATION" ]]; then
  echo "ERROR: Bundle does not contain required annotation: $ANNOTATION_REL" >&2
  exit 6
fi
mkdir -p "$(dirname "$OUTPUT_ANNOTATION")"
cp "$INPUT_ANNOTATION" "$OUTPUT_ANNOTATION"
rm -rf "$WORK_DIR/.out"
ln -s "$OUT_DIR" "$WORK_DIR/.out"
export HF_HOME="${HF_HOME:-$START_DIR/.hf-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$START_DIR/.uv-cache}"
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME" "$UV_CACHE_DIR"

cd "$WORK_DIR"
if [[ -f .env ]]; then
  # Requires a shell-compatible .env file (KEY=value).
  set -a
  # shellcheck source=/dev/null
  source .env
  set +a
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "$OUT_DIR/gpu.txt" 2>&1 || true
elif [[ "${EVAL_REQUIRE_CUDA:-1}" == "1" ]]; then
  echo "ERROR: nvidia-smi not available. Run this on a CUDA GPU host." >&2
  exit 4
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl and uv are both unavailable." >&2
    exit 5
  fi
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# --frozen: use the project's existing lockfile (Linux-specific wheels selected).
# Tee logs to persistent disk. CUDA verification prevents silent CPU execution.
(
  uv sync --frozen
  .venv/bin/python -m pip --version >/dev/null 2>&1 || true
  .venv/bin/python - <<'PY'
import sys
print('Python:', sys.version)
try:
    import torch
    print('PyTorch:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
    elif __import__('os').environ.get('EVAL_REQUIRE_CUDA', '1') == '1':
        raise RuntimeError('PyTorch cannot access CUDA. Check uv.lock / PyTorch CUDA wheel.')
except ImportError:
    if __import__('os').environ.get('EVAL_REQUIRE_CUDA', '1') == '1':
        raise
PY
  .venv/bin/python -m pip freeze 2>/dev/null > "$OUT_DIR/packages.txt" || uv pip freeze --python .venv/bin/python > "$OUT_DIR/packages.txt"
) 2>&1 | tee "$OUT_DIR/setup.log"

# Actual CLI is provided by the caller because the project screenshot does not
# reveal which of the scripts under src/scripts/ starts the benchmark.
export PATH="$WORK_DIR/.venv/bin:$PATH"
if [[ "$1" == "-m" ]]; then
  shift
  .venv/bin/python -m "$@" 2>&1 | tee "$OUT_DIR/evaluation.log"
else
  .venv/bin/python "$@" 2>&1 | tee "$OUT_DIR/evaluation.log"
fi
exit 0
__EVAL_PAYLOAD_BELOW__
RUNNER

cat "$TMP_DIR/project.tar.gz" >> "$RUN_FILE"
chmod +x "$RUN_FILE"
echo "Included annotation: $ANNOTATION_REL (only file from .out)"
echo "Created: $RUN_FILE"
ls -lh "$RUN_FILE"
echo "Credentials inside bundle: yes, if .env / .viescore.key are present in the project."
