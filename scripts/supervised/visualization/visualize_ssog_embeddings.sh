#!/bin/bash
#SBATCH --job-name=ssog_embeddings
#SBATCH --output=logs/cifar100/ssog_embeddings_%j.out
#SBATCH --error=logs/cifar100/ssog_embeddings_%j.err
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --partition=l40s
#
# t-SNE of SSoG embeddings.
#
#   sbatch scripts/supervised/visualization/visualize_ssog_embeddings.sh <model-config> <checkpoint> [output-dir]

source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/common.sh"
MODEL_CONFIG="${1:?Usage: sbatch scripts/supervised/visualization/visualize_ssog_embeddings.sh <model-config> <checkpoint> [output-dir]}"
CHECKPOINT="${2:?Missing checkpoint path}"
OUTPUT_DIR="${3:-artifacts/ssog_linear_probe_embeddings}"
check_gpu
run -m assets.visualizations.visualize_ssog_embeddings --model-config "$MODEL_CONFIG" --checkpoint "$CHECKPOINT" --output-dir "$OUTPUT_DIR" \
  --samples-per-class 25 --batch-size 128 --perplexity 30 --tsne-iterations 1500
