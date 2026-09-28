#!/bin/bash
#SBATCH --job-name=imagenet_discrete_ssog
#SBATCH --output=logs/imagenet/discrete_ssog_%j.out
#SBATCH --error=logs/imagenet/discrete_ssog_%j.err
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:00:00
#SBATCH --partition=a40
#
# Discrete and looped SSoG on ImageNet (ssog/discrete/imagenet/, ssog/looped_ssog/imagenet/).
# Defaults to 4-GPU DDP; for a one-GPU config:
#   sbatch -p l40s --gres=gpu:l40s:1 --ntasks-per-node=1 --mem=40G <this script> <config>
#
#   sbatch scripts/supervised/training/imagenet/imagenet_discrete_ssog.sh ssog/discrete/imagenet/fixed_dim768_h12_r4_depth12.yaml

source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/common.sh"
CONFIG="${1:?Usage: sbatch scripts/supervised/training/imagenet/imagenet_discrete_ssog.sh <config under configs/classification>}"
check_gpu
check_devices "$CONFIG"
run main_classification_discrete_ssog.py -c "$CONFIG"
