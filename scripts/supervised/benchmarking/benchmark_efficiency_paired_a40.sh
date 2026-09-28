#!/bin/bash
#SBATCH --job-name=ssog_pair_a40
#SBATCH --output=logs/ssog/efficiency_paired_a40_%j.out
#SBATCH --error=logs/ssog/efficiency_paired_a40_%j.err
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --partition=a40
#
# Paired dense/SSoG benchmark in one process, followed by its plots. Superseded by
# the per-model *_isolated runs of benchmark_efficiency_resolution.sh.

source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/common.sh"
CONFIG="${1:-configs/benchmarks/efficiency_paired_a40.yaml}"
OUTPUT_DIR="${2:-experiments/SSOG/efficiency/paired_a40}"
mkdir -p "$OUTPUT_DIR"
check_gpu
run -m assets.benchmarking.benchmark_efficiency_resolution --config "$CONFIG" --output-dir "$OUTPUT_DIR"
run -m assets.plotting.plot_efficiency_paired --input "$OUTPUT_DIR/resolution_scaling.csv" --output-dir "$OUTPUT_DIR/plots"
