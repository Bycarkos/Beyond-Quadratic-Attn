#!/bin/bash
#SBATCH --job-name=cifar100_bvit_baseline
#SBATCH --output=logs/cifar100/bvit_baseline_%j.out
#SBATCH --error=logs/cifar100/bvit_baseline_%j.err
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:00:00
#SBATCH --partition=l40s
#
# Dense-attention looped ViT (bViT) baseline on CIFAR-100 (ssog/baselines/cifar100/bvit_*).
#
#   sbatch scripts/supervised/training/cifar100/cifar100_bvit_baseline.sh ssog/baselines/cifar100/bvit_dim768_h12_steps12.yaml

source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/common.sh"
CONFIG="${1:?Usage: sbatch scripts/supervised/training/cifar100/cifar100_bvit_baseline.sh <config under configs/classification>}"
check_gpu
check_devices "$CONFIG"
run main_classification_bvit_baseline.py -c "$CONFIG"
