#!/bin/bash
#SBATCH --job-name=ssog_apply_bottleneck_audit
#SBATCH --output=logs/cifar100/ssog_apply_bottleneck_%j.out
#SBATCH --error=logs/cifar100/ssog_apply_bottleneck_%j.err
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --partition=l40s
#
# Profiles the SSoG apply step (arguments are passed to the script).

source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/common.sh"
check_gpu
run -m assets.benchmarking.audit_ssog_apply_bottleneck "$@"
