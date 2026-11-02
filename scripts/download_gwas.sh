#!/bin/bash
# Downloads the public GWAS summary statistics for SPEC_v2 (§1) and the MAGMA East Asian LD reference.
# Run from anywhere; files go to data/gwas/ (gitignored). Rerunning skips finished files and resumes
# partial ones (*.part). The summary statistics download 4 at a time (EBI serves ~1 MB/s per connection).
# Per study: the GWAS Catalog harmonised file (uniform columns, rsIDs) where one exists, otherwise
# the file as deposited; plus README / meta-yaml / md5sum for provenance.
# MAGMA itself and the NCBI37.3 gene locations were fetched by hand into data/gwas/magma/
# (https://cncr.nl/research/magma/, v1.10 Linux static build).
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/data/gwas
FTP=http://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics

get() {  # url dest
    [ -s "$2" ] && { echo "skip $2"; return; }
    curl -sS -L -C - --retry 5 --retry-delay 30 -m 14400 -o "$2.part" "$1" && mv "$2.part" "$2" && echo "ok $2" \
        || echo "FAIL $1"
}

# 1. MAGMA 1000 Genomes EAS reference (440 MB)
mkdir -p $OUT/ref_eas
get https://vu.data.surfsara.nl/index.php/s/dz6PYdKOi3xVqHn/download $OUT/ref_eas/g1000_eas.zip

# 2. summary statistics: accession, trait, cohort, file (relative to the accession folder)
one() {
    read -r G TRAIT COHORT FILE <<< "$1"
    case $G in
        GCST0*) n=$((10#${G#GCST})); lo=$(( (n - 1) / 1000 * 1000 + 1 )); R=$(printf "GCST%06d-GCST%06d" $lo $((lo + 999))) ;;
        *)      n=$((10#${G#GCST})); lo=$(( (n - 1) / 1000 * 1000 + 1 )); R=$(printf "GCST%08d-GCST%08d" $lo $((lo + 999))) ;;
    esac
    D=$OUT/$COHORT/${TRAIT}_$G
    mkdir -p $D
    get "$FTP/$R/$G/$FILE" "$D/$(basename $FILE)"
    for meta in README.txt readme.txt md5sum.txt "$FILE-meta.yaml"; do
        curl -sS -f -m 120 -o "$D/$(basename "$meta")" "$FTP/$R/$G/$meta" 2>/dev/null || rm -f "$D/$(basename "$meta")"
    done
    if [[ $FILE == harmonised/* ]]; then  # checksums of the harmonised files
        curl -sS -f -m 120 -o "$D/md5sum.harmonised.txt" "$FTP/$R/$G/harmonised/md5sum.txt" 2>/dev/null \
            || rm -f "$D/md5sum.harmonised.txt"
    fi
}
export -f one get
export OUT FTP
grep . <<'EOF' | xargs -d '\n' -P 4 -I{} bash -c 'one "$1"' _ {}
GCST90002388 LYMPH EUR harmonised/32888494-GCST90002388-EFO_0004587.h.tsv.gz
GCST90002393 MONO EUR harmonised/GCST90002393.h.tsv.gz
GCST90002398 NEUT EUR harmonised/32888494-GCST90002398-EFO_0004833.h.tsv.gz
GCST90002381 EOS EUR harmonised/GCST90002381.h.tsv.gz
GCST90002379 BASO EUR harmonised/32888494-GCST90002379-EFO_0005090.h.tsv.gz
GCST90002407 WBC EUR harmonised/32888494-GCST90002407-EFO_0004308.h.tsv.gz
GCST90002392 MCV EUR harmonised/32888494-GCST90002392-EFO_0004526.h.tsv.gz
GCST90132223 RA EUR harmonised/GCST90132223.h.tsv.gz
GCST010042 ASTHMA EUR HanY_prePMID_asthma_UKBB.txt.gz
GCST90244787 AD EUR GCST90244787_buildGRCh37.tsv
GCST003156 SLE EUR harmonised/26502338-GCST003156-EFO_0002690.h.tsv.gz
GCST004132 CD EUR harmonised/28067908-GCST004132-EFO_0000384.h.tsv.gz
GCST004133 UC EUR harmonised/28067908-GCST004133-EFO_0000729.h.tsv.gz
GCST90014023 T1D EUR harmonised/34012112-GCST90014023-EFO_0001359.h.tsv.gz
GCST90472771 PSO EUR harmonised/GCST90472771.h.tsv.gz
GCST90245992 HEIGHT EUR harmonised/GCST90245992.h.tsv.gz
GCST90239658 LDL EUR harmonised/GCST90239658.h.tsv.gz
GCST90018742 LYMPH BBJ harmonised/GCST90018742.h.tsv.gz
GCST90018747 MONO BBJ harmonised/GCST90018747.h.tsv.gz
GCST90018748 NEUT BBJ harmonised/GCST90018748.h.tsv.gz
GCST90018733 EOS BBJ harmonised/GCST90018733.h.tsv.gz
GCST90018726 BASO BBJ harmonised/GCST90018726.h.tsv.gz
GCST90018758 WBC BBJ harmonised/GCST90018758.h.tsv.gz
GCST90018690 RA BBJ harmonised/GCST90018690.h.tsv.gz
GCST90018575 ASTHMA BBJ harmonised/GCST90018575.h.tsv.gz
GCST90018564 AD BBJ harmonised/GCST90018564.h.tsv.gz
GCST90018739 HEIGHT BBJ harmonised/GCST90018739.h.tsv.gz
GCST90018741 LDL BBJ harmonised/GCST90018741.h.tsv.gz
GCST90018746 MCV BBJ harmonised/GCST90018746.h.tsv.gz
EOF
echo "DONE $(date)"
