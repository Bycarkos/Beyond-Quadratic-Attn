#!/bin/bash
#SBATCH --job-name=imagenet_ssog
#SBATCH --output=logs/imagenet/ssog_%j.out
#SBATCH --error=logs/imagenet/ssog_%j.err
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:00:00
#SBATCH --partition=a40
#
# Continuous-depth SSoG on ImageNet (ssog/imagenet/, ssog/augmentation/imagenet/).
# Defaults to 4-GPU DDP; for configs with other hardware.devices, e.g. one GPU:
#   sbatch -p l40s --gres=gpu:l40s:1 --ntasks-per-node=1 --mem=40G <this script> <config>
#
#   sbatch scripts/supervised/training/imagenet/imagenet_ssog.sh ssog/imagenet/fixed_r4_nfe12.yaml

source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/common.sh"
CONFIG="${1:?Usage: sbatch scripts/supervised/training/imagenet/imagenet_ssog.sh <config under configs/classification>}"
check_gpu
check_devices "$CONFIG"
run main_classification_ssog.py -c "$CONFIG"
