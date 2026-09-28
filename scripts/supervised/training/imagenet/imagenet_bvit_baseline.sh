#!/bin/bash
#SBATCH --job-name=imagenet_bvit_b16
#SBATCH --output=logs/imagenet/bvit_baseline_%j.out
#SBATCH --error=logs/imagenet/bvit_baseline_%j.err
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:00:00
#SBATCH --partition=rtx6000
#
# Dense-attention looped ViT (bViT) baseline on ImageNet (ssog/baselines/imagenet/bvit_*).
# Defaults to one GPU; for the 4-GPU resume config:
#   sbatch -p a40 --gres=gpu:4 --ntasks-per-node=4 <this script> ssog/baselines/imagenet/bvit_b16_steps12_resume_4gpu.yaml
#
#   sbatch scripts/supervised/training/imagenet/imagenet_bvit_baseline.sh ssog/baselines/imagenet/bvit_b16_steps12.yaml

source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/common.sh"
CONFIG="${1:?Usage: sbatch scripts/supervised/training/imagenet/imagenet_bvit_baseline.sh <config under configs/classification>}"
check_gpu
check_devices "$CONFIG"
run main_classification_bvit_baseline.py -c "$CONFIG"
