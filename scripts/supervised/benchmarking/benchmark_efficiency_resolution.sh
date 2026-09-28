#!/bin/bash
#SBATCH --job-name=ssog_efficiency
#SBATCH --output=logs/ssog/efficiency_%j.out
#SBATCH --error=logs/ssog/efficiency_%j.err
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --partition=l40s
#
# Throughput / peak-memory benchmark (configs/benchmarks/*.yaml; each model is best run
# alone, e.g. the *_isolated configs).
#
#   sbatch scripts/supervised/benchmarking/benchmark_efficiency_resolution.sh configs/benchmarks/efficiency_vit_ssog_isolated.yaml

source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/common.sh"
CONFIG="${1:-configs/benchmarks/efficiency_resolution.yaml}"
check_gpu
run -m assets.benchmarking.benchmark_efficiency_resolution --config "$CONFIG"
