#!/bin/bash
#SBATCH --job-name=eval_patch_size
#SBATCH --output=logs/evaluation/patch_size_%j.out
#SBATCH --error=logs/evaluation/patch_size_%j.err
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:00:00
#SBATCH --partition=l40s
#
# Patch-size robustness evaluation of a trained model.
#
#   sbatch scripts/evaluation/patch_size/eval_patch_size.sh evaluation/patch_size/cifar100/looped.yaml

source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/common.sh"
CONFIG="${1:?Usage: sbatch scripts/evaluation/patch_size/eval_patch_size.sh evaluation/patch_size/<dataset>/<model>.yaml}"
check_gpu
run eval_patch_size.py -c "$CONFIG"
