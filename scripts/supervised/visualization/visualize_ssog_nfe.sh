#!/bin/bash
#SBATCH --job-name=ssog_nfe_gif
#SBATCH --output=logs/cifar100/ssog_nfe_gif_%j.out
#SBATCH --error=logs/cifar100/ssog_nfe_gif_%j.err
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --partition=l40s
#
# Attention GIFs across ODE steps.
#
#   sbatch scripts/supervised/visualization/visualize_ssog_nfe.sh <model-config> <checkpoint> [output-dir]

source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/common.sh"
MODEL_CONFIG="${1:?Usage: sbatch scripts/supervised/visualization/visualize_ssog_nfe.sh <model-config> <checkpoint> [output-dir]}"
CHECKPOINT="${2:?Missing checkpoint path}"
OUTPUT_DIR="${3:-artifacts/ssog_nfe_gifs}"
check_gpu
run -m assets.visualizations.visualize_ssog_nfe --model-config "$MODEL_CONFIG" --checkpoint "$CHECKPOINT" --indices 0 1 2 3 --query 7 7 \
  --output-dir "$OUTPUT_DIR"
