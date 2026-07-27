#!/usr/bin/env bash
set -Eeuo pipefail

echo "Python Version"
python -V

echo "Checking GPU access..."

python - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside Docker. Check the NVIDIA driver and Docker GPU configuration.")

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
print(f"CUDA test tensor: {torch.arange(4, device='cuda')}")
PY

# A new run has no checkpoint yet. Avoid giving the trainer a nonexistent
# load path in that case.
if [[ -n "${LOAD_PATH:-}" && ! -f "$LOAD_PATH" ]]; then
    echo "No existing checkpoint found at $LOAD_PATH."
    echo "Starting a new training run."
    unset LOAD_PATH
else
    if [[ -n "${LOAD_PATH:-}" ]]; then
        echo "Resuming from $LOAD_PATH"
    fi
fi

cd /app/src

exec python -u trainer.py "$@"