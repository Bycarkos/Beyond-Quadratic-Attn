#!/bin/bash
#SBATCH --job-name=cifar100_dense_odevit
#SBATCH --output=logs/cifar100/dense_odevit_%j.out
#SBATCH --error=logs/cifar100/dense_odevit_%j.err
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:00:00
#SBATCH --partition=l40s
#
# Dense-attention ODE-ViT baseline on CIFAR-100 (ssog/baselines/cifar100/dense_odevit_*).
#
#   sbatch scripts/supervised/training/cifar100/cifar100_dense_odevit_baseline.sh ssog/baselines/cifar100/dense_odevit_dim768_h12_nfe12.yaml

source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/common.sh"
CONFIG="${1:?Usage: sbatch scripts/supervised/training/cifar100/cifar100_dense_odevit_baseline.sh <config under configs/classification>}"
check_gpu
check_devices "$CONFIG"
run main_classification_odevit_baseline.py -c "$CONFIG"
