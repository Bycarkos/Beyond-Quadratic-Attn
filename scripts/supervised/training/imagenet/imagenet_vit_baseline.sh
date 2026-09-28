#!/bin/bash
#SBATCH --job-name=imagenet_vit_b16
#SBATCH --output=logs/imagenet/vit_baseline_%j.out
#SBATCH --error=logs/imagenet/vit_baseline_%j.err
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:00:00
#SBATCH --partition=rtx3090
#
# Dense-attention ViT-B/16 baseline on ImageNet (ssog/baselines/imagenet/vit_b16*).
#
#   sbatch scripts/supervised/training/imagenet/imagenet_vit_baseline.sh ssog/baselines/imagenet/vit_b16.yaml

source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/common.sh"
CONFIG="${1:?Usage: sbatch scripts/supervised/training/imagenet/imagenet_vit_baseline.sh <config under configs/classification>}"
check_gpu
check_devices "$CONFIG"
run main_classification_vit_baseline.py -c "$CONFIG"
