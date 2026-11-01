# Novelty check (2026-11-01)

**Question.** Has anyone coupled a donor × cell type × gene tensor decomposition of a population
single-cell atlas, or another multicellular-program method, to genetic association statistics
(GWAS, eQTL or rare-variant burden) in one model, for example through shared gene loadings?

**Verdict: no close prior work found. Building goes ahead.** Expression programs and genetic
statistics have been linked many times, but always in one of three other ways:
- the genetics enters through the donor mode, by testing donor scores against genotypes;
- the programs are fitted first and tested against GWAS afterwards (enrichment or regression);
- the summary statistics are factorized on their own, with no expression in the model.

I found no model where gene loadings are shared between a multicellular expression tensor and a
gene × trait matrix of burden (or GWAS) effects, fitted jointly with the burden sampling error
modelled. The closest work is Ota et al. 2025. It is two-step, uses Perturb-seq knockouts in one
cell line rather than donor variation across cell types, and has no tensor (details below).

How the check was done: web search (extended), PubMed and bioRxiv searches, and metadata checks of
the key papers. Full texts could not be fetched from nature.com, biorxiv.org or the MIT thesis
server (blocked or rate-limited), so for those papers the verdict rests on abstracts, package
documentation and search snippets. Read the Ota et al. and LIVI full texts before writing the
paper's related-work section.

## Multicellular programs from donor × cell type × gene tensors (no genetics in the model)

| Work | What it does | Genetics? |
|---|---|---|
| scITD, Mitchel et al., *Nat Biotechnol* 2024 ([doi](https://doi.org/10.1038/s41587-024-02411-z); [bioRxiv 2022](https://www.biorxiv.org/content/10.1101/2022.02.16.480703v2); [code](https://github.com/kharchenkolab/scITD)) | Tucker decomposition of pseudobulk donor × cell type × gene; programs = donor scores + cell-type × gene loadings; lupus and COVID-19 cohorts | None in the published abstract. Donor scores are tested against clinical variables. **Main baseline.** |
| Federated scITD, Faes, van den Berg & Amir Haeri, arXiv 2606.24938 (June 2026) ([arXiv](https://arxiv.org/abs/2606.24938)) | Privacy-preserving federated version of the same decomposition (local subspaces merged by stacked SVD); lupus atlas | None |
| Multicellular factor analysis (MOFAcell), Ramirez Flores et al., *eLife* 2023 ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10718529/)) | Group factor analysis (MOFA) with one view per cell type | None |
| Washburn et al., *J Crohns Colitis* 2026 ([doi](https://doi.org/10.1093/ecco-jcc/jjaf225)) | Tensor decomposition of pediatric Crohn's scRNA-seq | GWAS used **separately**, to rank cell types. Not coupled. |
| Mitchel, PhD thesis, MIT 2025 ([DSpace](https://dspace.mit.edu/handle/1721.1/162335)) | Chapter 1 is scITD. A separate chapter covers a method for mechanisms of GWAS variants through cell-type or cell-state eQTLs. | A separate chapter, not linked to the tensor programs (based on the abstract) |

## Genetics linked to latent expression programs, but through donors or in two steps

| Work | Link to genetics | How it differs from this project |
|---|---|---|
| LIVI, Vagiaki … Stegle, bioRxiv 2026 ([doi](https://doi.org/10.64898/2026.02.04.703363)) | Structured sparse VAE on OneK1K single cells. Donor-specific latent factors are tested against genotypes (trans-eQTLs). Also benchmarks scITD on OneK1K. | Genetics enters through the **donor** mode, with common variants of the same donors. It uses no gene-level association statistics and no external biobank. |
| SDA, Hore et al., *Nat Genet* 2016 | Sparse tensor decomposition of individuals × tissues × genes (bulk), followed by GWAS of the individual scores | Same difference: donor mode, bulk tissue |
| Ota, Spence, Zeng, Dann, Milind, Marson & Pritchard, *Nature* 2025 ([doi](https://doi.org/10.1038/s41586-025-09866-3)) | cNMF programs from genome-wide Perturb-seq (K562) are linked to UK Biobank LoF burden and GWAS effects on blood traits, giving gene → program → trait graphs | Closest in spirit, but **two-step**. Programs come from knockout responses in one cell line, not from natural donor variation across cell types, and burden statistics do not shape the programs. Natural reference and possible extra baseline. |
| PoPS, Weeks et al., *Nat Genet* 2023; sc-linker, Jagadeesh et al., *Nat Genet* 2022 | Gene features or programs (from scRNA-seq) are regressed against gene-level GWAS signal, or tested by S-LDSC enrichment | Two-step. The features are fixed before the genetics is seen. |

## Factorizations of association statistics alone

| Work | Note |
|---|---|
| DeGAs, Tanigawa et al., *Nat Commun* 2019 (includes protein-truncating variants in UK Biobank) | Truncated SVD of a variant × trait z-score matrix. No expression. |
| GLEANR, bioRxiv 2024 ([link](https://www.biorxiv.org/content/10.1101/2024.11.12.623313v2.full)) | Sparse factorization of GWAS summary statistics that models sample sharing (correlated errors across traits), as we do with R. No expression. |
| SBAT, *AJHG* 2024 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11480795/)) | Joint testing of burden scores by non-negative least squares. A test, not a factorization. |

## Generic method

Coupled matrix–tensor factorization (CMTF; Acar et al. 2011 and later work) is a standard tool in
chemometrics and recommender systems. The new part here is not the CMTF idea. It is the
statistical model:
- the per-entry sampling variance of pseudobulk shares, which is known;
- burden standard errors with a cross-trait error correlation estimated from null genes;
- non-negative shared gene programs;
- a coupling weight chosen by cross-validation over held-out genes.

These are applied to a population atlas (OneK1K) paired with biobank LoF burden statistics from a
separate cohort.

## Risks for the novelty claim

- A preprint posted in the last few weeks could have been missed. Re-run a quick search before
  submitting.
- Ota et al. (2025) will be the reviewers' first comparison. The pitch has to be about natural
  donor-to-donor co-variation across many cell types in a population atlas, learned jointly with
  the burden statistics, as opposed to perturbation programs in one cell line linked afterwards.
