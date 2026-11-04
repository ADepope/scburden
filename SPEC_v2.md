# Specification, version 2: GWAS gene scores

Status: **frozen on 2026-11-02.** This is the pre-registration. It was committed after the summary
statistics were downloaded and checked, but before any GWAS gene score was computed. Later changes go
into a Deviations section (§5).

## Why a version 2

Version 1 ([SPEC.md](SPEC.md)) coupled the OneK1K tensor to UK Biobank LoF burden effects on 18
quantitative traits. Its pre-registered test failed: no method predicted held-out genes' burden
effects better than zero. The post-hoc diagnostics (`results/real/diagnostics.tsv`) showed why.
- **Per-gene burden effects are about 92% noise.** The mean of $`z_A z_B`$ across halves is 0.08.
- **Most traits don't match the tissue.** The traits with replicating signal (red-cell traits, HT,
  lipids) act mainly outside PBMCs.
- **Even an in-sample linear map** from the expression programs explained only 0.4% of burden-effect variance.

Version 2 keeps the model and the expression tensor. It changes the gene × trait matrix to one with
far more signal per gene and better tissue match:
- **Gene-level GWAS scores** (common variants) replace the LoF burden effects.
- **The traits are immune traits** (immune-cell counts and immune-mediated diseases), plus a few
  control traits that shouldn't depend on PBMCs.
- **Replication is in another ancestry**, Biobank Japan, which shares no people with the discovery data.

Everything uses public summary statistics. No individual-level data is read.

## Question

Do multicellular gene programs learned jointly from the OneK1K tensor and immune-trait GWAS gene
scores predict the GWAS signal of genes on held-out chromosomes, and do they replicate in Biobank
Japan? The comparison is against the same programs learned from expression alone and against
cell-type specificity features.

## 1. Data

### Expression

The same tensor as version 1: OneK1K, 981 donors × 16 cell types × 6,444 genes
(`data/real/tensor.npz`). Genes in the MHC region (chr6:25–34 Mb, hg19) are removed from every
analysis, because their GWAS signal is extreme and driven by LD.

### Traits and summary statistics

All sources below are public full summary statistics in the GWAS Catalog (open download, no
application). They were checked on 2026-11-02 against the Catalog REST API, and four accessions were
spot-checked independently. Some European studies include UK Biobank participants. These are
published consortium results; nothing from our own UK Biobank application is used.

- **Discovery:** European ancestry. All 17 traits enter the discovery fit and help shape the programs.
- **Replication:** Biobank Japan only (Sakaue et al., *Nat Genet* 2021; BBJ-only files, *not* the
  meta-analyses with UKB/FinnGen). Biobank Japan shares no people with any discovery study.
- **Primary score:** only the 12 traits with a well-powered Biobank Japan counterpart (9 immune + 3 control).

| Trait | Group | Discovery (EUR): study, GCST, N | UKB in discovery? | Replication (BBJ-only): GCST, N | In primary score? |
|---|---|---|---|---|---|
| Lymphocyte count | immune | Vuckovic 2020, GCST90002388, 408,112 | yes (UKB-only file) | GCST90018742, 95,717 | yes |
| Monocyte count | immune | Vuckovic 2020, GCST90002393, 408,112 | yes | GCST90018747, 95,119 | yes |
| Neutrophil count | immune | Vuckovic 2020, GCST90002398, 408,112 | yes | GCST90018748, 82,810 | yes |
| Eosinophil count | immune | Vuckovic 2020, GCST90002381, 408,112 | yes | GCST90018733, 93,063 | yes |
| Basophil count | immune | Vuckovic 2020, GCST90002379, 408,112 | yes | GCST90018726, 91,908 | yes |
| White-cell count | immune | Vuckovic 2020, GCST90002407, 408,112 | yes | GCST90018758, 154,355 | yes |
| Rheumatoid arthritis | immune | Ishigaki 2022 EUR, GCST90132223, 22,350 / 74,823 | probably not | GCST90018690, 5,348 / 173,268 | yes |
| Asthma | immune | Han 2020 EUR, GCST010042, 64,538 / 329,321 | yes | GCST90018575, 13,015 / 162,933 | yes |
| Atopic dermatitis | immune | Budu-Aggrey 2023, GCST90244787, 60,653 / 804,329 | yes | GCST90018564, 4,296 / 163,807 | yes |
| SLE | immune | Bentham 2015, GCST003156, 5,201 / 9,066 | no | (317 cases: underpowered) | no |
| Crohn's disease | immune | de Lange 2017, GCST004132, 12,194 / 28,072 | no | none in Biobank Japan | no |
| Ulcerative colitis | immune | de Lange 2017, GCST004133, 12,366 / 33,609 | no | (314 cases: underpowered) | no |
| Type 1 diabetes | immune | Chiou 2021, GCST90014023, 18,942 / 501,638 | yes | (1,219 cases: borderline) | no |
| Psoriasis | immune | Dand 2025, GCST90472771, 36,466 / 458,078 | yes | (206 cases: underpowered) | no |
| Height | control | Yengo 2022 EUR, GCST90245992, 1,597,374 | yes | GCST90018739, 165,056 | yes |
| LDL cholesterol | control | Graham 2021 EUR, GCST90239658, 1,320,016 | yes | GCST90018741, 72,866 | yes |
| MCV | control | Vuckovic 2020, GCST90002392, 408,112 | yes | GCST90018746, 129,832 | yes |

Notes:
- **Control traits.** Their biology is mostly outside PBMCs: height acts in the growth plate, LDL in
  the liver and MCV in erythroid precursors. They test whether any gain from coupling is specific
  to immune traits.
- **Ancestry mixing must be avoided.** Chen 2020 all-ancestry blood-count files and Ishigaki
  2022's trans-ancestry RA files include Biobank Japan, so only the European files listed are used.
- **Build.** Gene scores use the rsIDs in each file, matched to the reference panel, so differences
  in genome build (Chiou 2021 is GRCh38) don't matter. Files without rsIDs are matched by position
  after liftover to GRCh37.
- **Sample size.** MAGMA is given the per-SNP N where the file has it, otherwise the total N
  (cases + controls for diseases). The files with per-SNP N are height and LDL (`n`), psoriasis
  (`cum_eff_sample_size`) and European asthma (`N`).
- **Files as downloaded.** All 29 files match their published md5 and pass `gzip -t`
  (`scripts/download_gwas.sh`). The SNP identifier used is, in order: `hm_rsid` or `rsid` (GWAS
  Catalog harmonised files), then `variant_id` if it is an rsID. This fallback matters for SLE,
  where some rows failed harmonisation but keep their rsID in `variant_id`.
- **Two files are not harmonised.** European asthma (`HanY_prePMID_asthma_UKBB.txt.gz`) has no
  rsIDs, so its SNPs (`chr:pos_alleles`, GRCh37) are matched to the reference panel by position and
  alleles. European atopic dermatitis (`GCST90244787_buildGRCh37.tsv`) has rsIDs but no N column.
- **Not included.** Graves' disease (no European source was checked). The larger 2026 T1D and
  psoriasis GWAS were not needed, because neither trait is in the primary score.
- **Height** uses the European file that the Catalog lists with about 1.37M SNPs (HapMap3-like).
  This is sparse but enough for MAGMA gene scores.
- **A sensitivity analysis without UK Biobank** would need FinnGen R12 for the diseases, but no
  non-UKB European blood-count GWAS of comparable size exists. It is therefore not planned.

### Gene scores

- **Method:** MAGMA v1.10, SNP-wise mean model.
- **Gene windows:** NCBI37.3 gene locations, with a window of 10 kb upstream and 1.5 kb downstream.
- **LD reference:** 1000 Genomes Phase 3. EUR for the discovery traits: the local LDSC plink files
  (`annotations/raw/1000G_plink`), with `.bed` and `.fam` extracted from the local archive. EAS
  for replication, downloaded from the MAGMA site.
- **Output:** a gene z-score $`z_{gt}`$, which is $`\mathcal N(0,1)`$ under the null.
- **Gene matching:** genes are matched to the tensor by Entrez ID, via Ensembl.
- **Covariates:** gene length, number of SNPs and number of MAGMA parameters (all on the log
  scale) are regressed out of $`z`$, using the training chromosomes only. This follows PoPS.

## 2. Model

The model is the same as version 1 (SPEC.md §2). The burden block now takes the residual gene
scores, $`\hat\beta_{gt} = z_{gt}`$, with $`S_g = I`$, because z-scores have unit sampling variance.
$`R`$ is estimated from genes with $`\max_t z_{gt} < 2`$ (MAGMA z-scores are one-sided).
$`\Theta`$ stays signed, although we expect it to be mostly positive.

**Known misspecification.** MAGMA scores of nearby genes are correlated through LD, but the model
treats genes as independent. The joint fit will therefore be overconfident. This is handled by
using the refit route as the primary prediction (as in version 1) and by holding out whole
chromosomes.

## 3. Prediction and selection

- **Folds:** 5 groups of chromosomes, balanced by gene count. Genes on a held-out chromosome group
  never contribute to a fit that predicts them.
- **Routes:** joint and refit predictions, as in version 1. Refit is primary.
- **Choosing η:** grid $`\{1, 3, 10, 30, 100\}`$, selected by the held-out gain on the discovery
  GWAS. The replication data are never used for any choice.
- **K:** $`K_{\max} = 20`$, with ARD.

## 4. Evaluation

### Primary

- **Score:** the Pearson correlation, per trait, between the out-of-fold predictions and the
  Biobank Japan gene z-scores of the same genes. The pooled score is the mean over the 12 traits
  marked "in primary score" in §1.
- **Why correlation and not R²:** the two ancestries differ in sample size and LD, so the z-scores
  are on different scales and R² would penalize calibration.
- **Comparison:** coupled (refit) minus each baseline.
- **Uncertainty:** a block bootstrap over contiguous ~5 Mb gene blocks, 1,000 draws.

### Baselines

| ID | Baseline | Note |
|---|---|---|
| B0 | Covariates only | gene length, number of SNPs, number of parameters |
| B1 | scITD-style | as in version 1 |
| B2 | Two-step | expression-only fit of the same model, then nested ridge |
| B3 | Random gene sets | as in version 1 |
| B5 | Cell-type specificity | per-cell-type mean expression and specificity (16 + 16 features), then nested ridge. **Key baseline:** it asks whether donor-to-donor co-variation adds anything beyond which cell types express a gene, which is what sc-linker and most GWAS–single-cell methods use. |
| B6 | Full PoPS (optional) | its published features, if the download is approved; secondary, as a reference point only |

### Success criteria (pre-registered)

1. Coupled minus B2 on the pooled Biobank Japan correlation is above 0, with a 95% CI that
   excludes 0.
2. Coupled minus B5 likewise: above 0, with a 95% CI that excludes 0.
3. **Specificity:** the gain from criterion 1, averaged over the 9 immune traits in the primary
   score, is larger than its average over the 3 control traits.

If criterion 1 holds but criterion 2 doesn't, the conclusion is that coupling helps, but the
helpful part is cell-type specificity, not donor-level co-variation.

### Secondary

- Within-EUR held-out-chromosome noise-corrected R² (the version 1 metric, with $`SE = 1`$).
- Program–trait replication. Fit the model again on the Biobank Japan scores, match programs by
  $`W`$ cosine, and correlate the program effects $`\Theta`$ between the two fits.
- Highlights for external validation follow the version 1 plan, but only programs whose trait links
  replicate in Biobank Japan qualify.

## 5. What is fixed now

This document: the traits and files, the gene-score pipeline, the model, the selection rules, the
metrics and the success criteria. Any later change goes into a Deviations section at the end of
this file, with its date and reason, as in version 1.

## 6. Downloads

Done on 2026-11-02, with approval. Everything is in `data/gwas/`, which is not in git.

| Item | Size |
|---|---|
| MAGMA v1.10 Linux binary and NCBI37.3 gene locations (`magma/`) | < 10 MB |
| 1000 Genomes EAS reference, MAGMA site (`ref_eas/`) | 0.46 GB zipped; 22.3M variants, 504 people |
| 1000 Genomes EUR reference, merged from the local LDSC plink files (`ref_eur/`) | 10.0M variants, 489 people |
| Discovery summary statistics, 17 traits (`EUR/`) | 18 GB |
| Biobank Japan-only summary statistics, 12 traits (`BBJ/`) | 3.9 GB |
| PoPS features (optional, B6) | not downloaded; needs separate approval |

## 7. Compute

Each MAGMA run (per trait) takes minutes on a CPU, so the whole set fits in one SLURM array job of
~30 tasks. The model fits need the same GPU job as version 1: about 3 h per η grid on an A10, so
expect ~4–6 h with replication. Every job is submitted only after explicit approval.

## Deviations

2026-11-04, after the MAGMA gene scores were computed and before any model fit:
- **Gene z-scores are capped at 6.1094** ($`p = 5\times10^{-10}`$), for every trait and both
  cohorts. When MAGMA's numerical integration fails, it falls back to simulation and reports
  $`p = 5\times10^{-10}`$ if no simulated statistic beats the observed one (MAGMA v1.10 manual,
  SNP-wise mean model). Those genes have $`z \ge 6.11`$ with no upper bound, while genes where
  integration worked reach $`z = 14.9`$, so scores above 6.11 are not comparable. Floored genes:
  1,544 of 18,018 for European height, 79–490 for the other well-powered European traits, 1–128
  in Biobank Japan. After the cap, 4,225 discovery and 368 replication gene–trait pairs (of the
  6,320 analysed genes) sit at the cap.
- **Note on §1.** The European asthma file does have rsIDs for some SNPs; the others are
  `chr:pos_alleles`. As specified, all its SNPs are matched to the reference panel by position and
  alleles, so the method is unchanged; only the description in §1 was inaccurate.

Implementation details that the spec left open (not deviations):
- Gene matching: tensor symbol → Ensembl ID from the OneK1K gene table → Entrez ID from HGNC. Of
  the 6,444 tensor genes, 6,434 are in NCBI37.3, 24 of these have no MAGMA score (no SNPs in the
  reference panel) and 90 overlap the MHC, leaving 6,320 genes.
- Chromosome folds: greedy assignment by gene count, largest chromosome first. Groups:
  {1, 5, 14, 18}, {6, 15, 16, 19}, {2, 7, 9, 21, 22}, {4, 8, 11, 12}, {3, 10, 13, 17, 20}, with
  1,228–1,302 genes each.
- $`R`$ (discovery) rests on only 220 genes with $`\max_t z < 2`$ across all 17 traits, because the
  tensor genes are expressed in immune cells and the European studies are large. Its largest entry
  is neutrophil count with white-cell count (0.76); the median $`|r|`$ is 0.06. $`R`$ (Biobank Japan)
  uses 2,066 genes.
