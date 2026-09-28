#!/bin/bash
#SBATCH --job-name=attention_sinks
#SBATCH --output=logs/evaluation/attention_sinks_%j.out
#SBATCH --error=logs/evaluation/attention_sinks_%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:00:00
#SBATCH --partition=a40
#
# Attention-sink analysis of all trained models, then its LaTeX table. Extra
# arguments are passed on, e.g. --only "Discrete/imagenet/MHA".

source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/common.sh"
check_gpu
run -m assets.analysis.analyze_attention_sinks --num-images 512 --batch-size 32 --device cuda "$@"
"$PYTHON" -m assets.tables.generate_attention_sink_table
