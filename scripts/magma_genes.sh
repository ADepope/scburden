#!/bin/bash -l
#SBATCH --job-name sctb_magma
#SBATCH --array 1-29
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 48G
#SBATCH --time 24:00:00
#SBATCH --output /nfs/scistore24/robingrp/adepope/sc_tensor_burden/logs/magma_%A_%a.log
#
# MAGMA gene scores for SPEC_v2.md §1: one array task per row of scripts/gwas_traits.tsv
# (17 European discovery files, then 12 Biobank Japan files).
# 1. scripts/prepare_magma_input.py writes data/gwas/magma_in/<COHORT>_<TRAIT>.tsv (SNP, P, N).
# 2. SNP-to-gene annotation, once per reference panel (EUR or EAS; the first task to get the lock
#    makes it): NCBI37.3 genes, window 10 kb upstream and 1.5 kb downstream.
# 3. MAGMA v1.10 gene analysis, SNP-wise mean model, 1000 Genomes reference of the matching ancestry,
#    N per SNP (column N). Output: data/gwas/magma_out/<COHORT>_<TRAIT>.genes.out (ZSTAT = gene z).
# Runtime: EUR height took 4.0 h and 1.8 GB (job 1695575); MAGMA is single-threaded. Most of the time
# goes to simulated p-values for very significant genes, so dense, well-powered files may take longer.
# Public summary statistics and the public 1000 Genomes reference only.
# Rerun a single row with: sbatch --array <row> scripts/magma_genes.sh
set -euo pipefail
P=/nfs/scistore24/robingrp/adepope/sc_tensor_burden
PY=/nfs/scistore24/robingrp/adepope/sc_dosage_J/envs/scdosage/bin/python
G=$P/data/gwas
MAGMA=$G/magma/magma
T=${SLURM_ARRAY_TASK_ID:-$1}
cd $P

read -r COHORT TRAIT < <(awk -F'\t' -v t=$((T + 1)) 'NR == t {print $1, $2}' scripts/gwas_traits.tsv)
case $COHORT in EUR) REF=$G/ref_eur/g1000_eur ;; BBJ) REF=$G/ref_eas/g1000_eas ;; esac
ANC=$(basename $REF)
mkdir -p $G/magma_out
echo "task $T: $COHORT $TRAIT, reference $REF ($(date))"

PYTHONNOUSERSITE=1 $PY -I scripts/prepare_magma_input.py $T

ANNOT=$G/magma_out/annot_$ANC
(
    flock 9
    if [ ! -s $ANNOT.genes.annot ]; then
        $MAGMA --annotate window=10,1.5 --snp-loc $REF.bim --gene-loc $G/magma/NCBI37.3.gene.loc --out $ANNOT
    fi
) 9> $G/magma_out/.annot_$ANC.lock

$MAGMA --bfile $REF --gene-annot $ANNOT.genes.annot --gene-model snp-wise=mean \
    --pval $G/magma_in/${COHORT}_$TRAIT.tsv use=SNP,P ncol=N \
    --out $G/magma_out/${COHORT}_$TRAIT
echo "done $(date)"
