# Shared setup for every Slurm launcher in scripts/. Source it right after the
# #SBATCH header:
#
#   source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/common.sh"
#
# Submit jobs from the repository root (Slurm sets SLURM_SUBMIT_DIR to it).
#
# It fails fast on errors, moves into the repository, selects one Python
# environment, and provides:
#   check_gpu            print the allocation; stop if PyTorch sees no GPU
#   check_devices CFG    stop if CFG's setup.hardware.devices differs from the
#                        number of GPU tasks Slurm allocated (a mismatch would
#                        otherwise hang distributed training)
#   run ARGS...          run a Python entry point (through srun inside a job,
#                        so DDP gets one task per GPU)
#
# Hardware in a launcher's #SBATCH header is only a default; override it on the
# command line, before the script path, e.g.
#   sbatch -p a40 --gres=gpu:4 --ntasks-per-node=4 <launcher> <config>

set -euo pipefail

# Repository root = the parent of this scripts/ folder; Python from the active
# environment unless TRAIN_PYTHON points elsewhere.
SSOG_ROOT="${SSOG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${TRAIN_PYTHON:-python}"
cd "$SSOG_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1

check_gpu() {
  echo "Job ${SLURM_JOB_ID:-<local>} on ${SLURM_JOB_NODELIST:-$(hostname)}; GPUs: ${CUDA_VISIBLE_DEVICES:-<not set>}"
  "$PYTHON" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("ERROR: PyTorch cannot see a CUDA GPU; stopping the job.")
names = ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
print(f"PyTorch {torch.__version__} (CUDA {torch.version.cuda}); {torch.cuda.device_count()} GPU(s): {names}", flush=True)
PY
}

check_devices() {
  "$PYTHON" - "$1" <<'PY'
import os
import sys
from hydra import compose, initialize
from omegaconf import OmegaConf

config = sys.argv[1]
with initialize(version_base=None, config_path="configs/classification"):
    cfg = compose(config_name=config)
devices = OmegaConf.select(cfg, "setup.hardware.devices", default=1)
tasks = int(os.environ.get("SLURM_NTASKS_PER_NODE") or os.environ.get("SLURM_NTASKS") or 1)
if isinstance(devices, int) and devices != tasks:
    sys.exit(
        f"ERROR: {config} trains on {devices} GPU(s) but this job has {tasks} task(s).\n"
        f"Resubmit with: sbatch --gres=gpu:{devices} --ntasks-per-node={devices} <launcher> {config}"
    )
print(f"Config {config}: {devices} device(s), {tasks} Slurm task(s).", flush=True)
PY
}

run() {
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    srun "$PYTHON" "$@"
  else
    "$PYTHON" "$@"
  fi
}
