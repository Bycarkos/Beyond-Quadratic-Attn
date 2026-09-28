#!/bin/bash
#SBATCH --job-name=euler_rollout_bench
#SBATCH --output=logs/cifar100/euler_rollout_%j.out
#SBATCH --error=logs/cifar100/euler_rollout_%j.err
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --partition=l40s
#
# Benchmarks the hand-written Euler rollout against torchdiffeq (arguments are passed on).

source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/common.sh"
check_gpu
run -m assets.benchmarking.benchmark_euler_rollout "$@"
