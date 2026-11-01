#!/bin/bash -l
#SBATCH --job-name sctb_fit
#SBATCH --partition gpu
#SBATCH --gres gpu:A100:1
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 64G
#SBATCH --time 12:00:00
#SBATCH --output /nfs/scistore24/robingrp/adepope/sc_tensor_burden/logs/fit_real_%j.log
#
# Real-data fit of the coupled model and baselines (scripts/fit_real.py), after
# scripts/burden_splithalf.sh and scripts/prepare_real.py. Summary statistics and OneK1K moments only.
set -euo pipefail
P=/nfs/scistore24/robingrp/adepope/sc_tensor_burden
PY=/nfs/scistore24/robingrp/adepope/sc_dosage_J/envs/scdosage/bin/python
cd $P
nvidia-smi -L
PYTHONNOUSERSITE=1 $PY -I scripts/fit_real.py --device cuda --replicate
