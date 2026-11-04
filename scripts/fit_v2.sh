#!/bin/bash -l
#SBATCH --job-name sctb_fit_v2
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 64G
#SBATCH --time 12:00:00
#SBATCH --output /nfs/scistore24/robingrp/adepope/sc_tensor_burden/logs/fit_v2_%j.log
#
# Version 2 fit of the coupled model and baselines (scripts/fit_v2.py, SPEC_v2.md), after
# scripts/magma_genes.sh and scripts/prepare_v2.py. Public GWAS summary statistics and OneK1K moments only.
# Expected: about 3 h for the eta grid (as in version 1) plus 1-2 h for baselines and the two full fits.
set -euo pipefail
P=/nfs/scistore24/robingrp/adepope/sc_tensor_burden
PY=/nfs/scistore24/robingrp/adepope/sc_dosage_J/envs/scdosage/bin/python
cd $P
nvidia-smi -L
PYTHONNOUSERSITE=1 $PY -I scripts/fit_v2.py --device cuda
