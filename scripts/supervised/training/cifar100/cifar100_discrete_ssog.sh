#!/bin/bash
#SBATCH --job-name=cifar100_discrete_ssog
#SBATCH --output=logs/cifar100/discrete_ssog_%j.out
#SBATCH --error=logs/cifar100/discrete_ssog_%j.err
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:00:00
#SBATCH --partition=l40s
#
# Discrete and looped SSoG on CIFAR-100 (same entry point; looped configs set
# recurrent: true): ssog/discrete/cifar100/, ssog/looped_ssog/cifar100/ and
# ssog/cifar100_transfer/{discrete,looped}/.
#
#   sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh ssog/looped_ssog/cifar100/fixed_dim768_h12_r4_depth12.yaml

source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/common.sh"
CONFIG="${1:?Usage: sbatch scripts/supervised/training/cifar100/cifar100_discrete_ssog.sh <config under configs/classification>}"
check_gpu
check_devices "$CONFIG"
run main_classification_discrete_ssog.py -c "$CONFIG"
