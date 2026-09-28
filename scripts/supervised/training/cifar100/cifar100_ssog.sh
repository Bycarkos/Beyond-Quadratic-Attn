#!/bin/bash
#SBATCH --job-name=cifar100_ssog
#SBATCH --output=logs/cifar100/ssog_%j.out
#SBATCH --error=logs/cifar100/ssog_%j.err
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:00:00
#SBATCH --partition=l40s
#
# Continuous-depth SSoG on CIFAR-100: from-scratch runs (ssog/cifar100/),
# augmentation (ssog/augmentation/cifar100/) and ImageNet transfer
# (ssog/cifar100_transfer/imagenet_*.yaml).
#
#   sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh ssog/cifar100/fixed_r4_nfe12.yaml

source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/common.sh"
CONFIG="${1:?Usage: sbatch scripts/supervised/training/cifar100/cifar100_ssog.sh <config under configs/classification>}"
check_gpu
check_devices "$CONFIG"
run main_classification_ssog.py -c "$CONFIG"
