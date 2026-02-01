#!/bin/bash
#SBATCH --job-name=pg-10B224-50ep
#SBATCH -A marlowe-m000132
#SBATCH -p preempt
#SBATCH -N 1
#SBATCH -G 8
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=11:59:00
#SBATCH --array=0-1
#SBATCH --output=/projects/m000132/CTRATE_Paligemma_Training/logs/10B224-%A_%a.out
#SBATCH --error=/projects/m000132/CTRATE_Paligemma_Training/logs/10B224-%A_%a.err

set -euo pipefail

PROJ=/projects/m000132/CTRATE_Paligemma_Training
ENV=/projects/m000132/chexanatomy_conda
HF_CACHE=/projects/m000132/hf_cache

TRAIN_PY="$PROJ/cheXanatomy/training/train_paligemma.py"
BASE_YAML="$PROJ/cheXanatomy/config.yaml"

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
export PATH="$ENV/bin:$PATH"
hash -r

export HF_HOME="$HF_CACHE"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2
export WANDB_ENTITY="sergiosgatidis"
export WANDB_PROJECT="cheXanatomy"
export WANDB_START_METHOD=thread

if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
  FREEZE=true
else
  FREEZE=false
fi

RUN_NAME="10B_224_freeze${FREEZE}_50ep"
TMP_YAML="/tmp/${RUN_NAME}.yaml"

python - <<PY
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("${BASE_YAML}").read_text())

cfg["model"]["model_size"] = 10
cfg["model"]["input_image_size"] = 224

cfg["optimization"]["freeze_vision"] = ("${FREEZE}".lower() == "true")

cfg["training"]["num_train_epochs"] = 50
cfg["training"]["auto_find_batch_size"] = False
cfg["training"]["per_device_train_batch_size"] = 12
cfg["training"]["per_device_eval_batch_size"] = 12

cfg["logging"]["run_name"] = "${RUN_NAME}"

outdir = cfg["paths"]["model_output_dir"].rstrip("/")
cfg["paths"]["model_output_dir"] = f"{outdir}/${RUN_NAME}"

Path("${TMP_YAML}").write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

torchrun --standalone --nproc_per_node=8 "$TRAIN_PY" --config "$TMP_YAML"
