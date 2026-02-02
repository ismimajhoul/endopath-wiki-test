#!/bin/bash
#SBATCH --job-name=llm_gyneco
#SBATCH --partition=debug      # partition GPU
#SBATCH --gres=gpu:1           # alloue 1 GPU (V100-32GB)
#SBATCH --cpus-per-task=1
#SBATCH --mem=30g
#SBATCH --time=04:00:00
#SBATCH --output=experiments/llm_ollama/logs/slurm_%j.out


set -euo pipefail


echo "GPUs attribués par SLURM: ${SLURM_JOB_GPUS:-none}"
FIRST_GPU=$(echo "${SLURM_JOB_GPUS:-0}" | cut -d',' -f1)
export CUDA_VISIBLE_DEVICES=$FIRST_GPU
echo "GPU utilisé: $FIRST_GPU"


CFG="experiments/llm_ollama/configs/llm.yaml"
LOGDIR="experiments/llm_ollama/logs"
mkdir -p "$LOGDIR"


MODEL=$(sed -n 's/^[[:space:]]*model:[[:space:]]*"\(.*\)".*/\1/p' "$CFG" | head -n1)
INPUT_PATH=$(sed -n 's/^[[:space:]]*input_path:[[:space:]]*"\(.*\)".*/\1/p' "$CFG" | head -n1)


STAMP=$(date +%Y%m%d_%H%M%S)
BASEIN=$(basename "${INPUT_PATH:-input}" 2>/dev/null || echo input)
RUNNAME="${STAMP}__model-${MODEL//:/_}__input-${BASEIN%.*}"


# Assurer que le modèle est dispo
ollama pull "$MODEL" || true


# Lancer l’inférence
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 \
python3 src/run_ollama_infer.py --config "$CFG" 2>&1 | tee "$LOGDIR/${RUNNAME}.run.log"
