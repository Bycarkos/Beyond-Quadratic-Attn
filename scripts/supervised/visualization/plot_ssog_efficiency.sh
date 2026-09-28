#!/bin/bash
#SBATCH --job-name=ssog_eff_plots
#SBATCH --output=logs/ssog/efficiency_plots_%j.out
#SBATCH --error=logs/ssog/efficiency_plots_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --partition=a40
#
# Efficiency figures (CPU only; arguments are passed to assets/plotting/plot_efficiency_results.py).

source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/common.sh"
run -m assets.plotting.plot_efficiency_results "$@"
