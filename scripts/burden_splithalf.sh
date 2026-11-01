#!/bin/bash -l
#SBATCH --job-name lof_half
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 32G
#SBATCH --time 3:00:00
#SBATCH --output /nfs/scistore24/robingrp/adepope/sc_tensor_burden/logs/burden_splithalf_%j.log
#
# Split-half LoF burden summary statistics for sc_tensor_burden (SPEC.md §1). The user runs this;
# Claude never reads its inputs or the half ID lists, only the .glm.linear outputs.
# Adapted from rare_lof_AB/A/scripts/marginal_burden_alltraits.sh: same genotypes (geno/lof_train,
# HC-LoF burden columns, MAC >= 10 in the full training set), same phenotypes (2M rows of the
# gVAMP-EC all-traits runs.tsv), same model (linear, no covariates), 18 quantitative traits.
#
# The training individuals are split once at random (plink2 --thin-indiv 0.5, fixed seed) into half
# 1 and its complement, half 2; every trait uses the same split, so the halves share no people.
# Relatives across halves would correlate the two halves' errors a little; with a kinship file,
# a family-aware split would be better (not done here).
# Outputs: OUT/half{1,2}/TRAIT_lof.glm.linear (summary statistics only), OUT/half1.id (sample IDs,
# for the record; stays on the cluster, gitignored, not read by the analysis code).
# Burden columns with MAC < 10 within a half are still tested (plink2 has no per-run MAC filter
# here); the fitting code downweights them through their SE.
set -euo pipefail
A=/nfs/scistore24/robingrp/adepope/rare_lof_AB/A
RUNS=/nfs/scistore24/robingrp/adepope/gVAMP-EC_runs/alltraits_newdefault/scripts/runs.tsv
OUT=/nfs/scistore24/robingrp/adepope/sc_tensor_burden/data/burden_halves
SEED=20261101
BINARY=" BP CAD T2D "
THREADS=${SLURM_CPUS_PER_TASK:-16}
module purge; module load plink/20240526
mkdir -p $OUT/half1 $OUT/half2

# 1. the random half (written once; rerunning the script reuses it)
if [ ! -s $OUT/half1.id ]; then
    plink2 --bfile $A/geno/lof_train --thin-indiv 0.5 --seed $SEED --write-samples \
        --threads $THREADS --out $OUT/half1
fi
echo "half 1: $(($(wc -l < $OUT/half1.id) - 1)) samples (header excluded)"

# 2. marginal burden regressions per trait and half. Phenotypes: 2M rows of the gVAMP-EC runs.tsv,
# plus traits only listed in rare_lof_AB's own runs.tsv (HT; same file as its full-sample marginal)
{ awk -F'\t' '$2 == "2M" {print $1"\t"$4}' $RUNS
  awk -F'\t' 'NR == FNR {if ($2 == "2M") seen[$1] = 1; next} $2 == "lof" && !($1 in seen) {print $1"\t"$4}' \
      $RUNS $A/scripts/runs.tsv
} | while IFS=$'\t' read -r TRAIT PHEN; do
    [[ $BINARY == *" $TRAIT "* ]] && continue
    for H in 1 2; do
        O=$OUT/half$H
        if [ -s $O/${TRAIT}_lof.glm.linear ]; then echo "skip $TRAIT half $H (done)"; continue; fi
        [ $H = 1 ] && SEL="--keep $OUT/half1.id" || SEL="--remove $OUT/half1.id"
        echo "== $TRAIT half $H $(date)"
        plink2 --bfile $A/geno/lof_train --pheno $PHEN $SEL --glm allow-no-covars --threads $THREADS \
            --memory 30000 --out $O/${TRAIT}_lof
        f=$(ls $O/${TRAIT}_lof.*.glm.linear | head -1)
        mv "$f" $O/${TRAIT}_lof.glm.linear
    done
done
echo "DONE $(date)"
