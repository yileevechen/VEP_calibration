#!/bin/bash
# Universal wrapper to run best-fit and data generation for a given cluster
# Usage:
#   bash 00.wrapper_generate_simuData.sh <working_directory> <cluster_number>

# === Input arguments ===
wd=$1   # working directory (e.g., /path/to/cluster_simu_analysis_AM_10302025)
cln=$2  # cluster number
alpha=$3 # pre-estimated prior
dist=$4

if [ -z "$wd" ] || [ -z "$cln" ] || [ -z "$alpha" ] || [ -z "$dist" ]; then
    echo "Usage: bash 00.wrapper_generate_simuData.sh <working_directory> <cluster_number> <cluster_prior> <predictor type>"
    exit 1
fi

OUTDIR=${wd}/cluster${cln}
mkdir -p "$OUTDIR"

echo " True prior for cluster$cln is $alpha"

# === Activate environment ===
source activate calibrate_para 2>/dev/null || true
unset PYTHONPATH
source /sc/arion/work/cheny60/test-env/envs/calibrate_para/bin/activate

# === Run best fit ===
python ${wd}/00.get_best_fit.py "$cln" "yes" "$dist"

# === Extract parameters ===
pnr=$(grep pnr ${OUTDIR}/clust${cln}_${dist}_SimuInfo.txt | cut -d':' -f2)
nsamp=$(grep nsamp ${OUTDIR}/clust${cln}_${dist}_SimuInfo.txt | cut -d':' -f2)
method=$(grep method ${OUTDIR}/clust${cln}_${dist}_SimuInfo.txt | cut -d':' -f2)

if [ -z "$pnr" ] || [ -z "$nsamp" ] || [ -z "$method" ]; then
    echo " Missing simulation parameters in clust${cln}_${dist}_SimuInfo.txt"
    exit 1
fi

# === Run data generation (30 replicates, parallelized 6-way) ===
seq 1 30 | xargs -P 6 -I {} bash -c '
    python "$0"/00.data_generation.py \
        --dist "$1" \
        --seed "$2" \
        --clustn "$3" \
        --pnratio_calibrate "$4" \
        --pnratio_test "$4" \
        --alpha "$5" \
        --n_calibrate "$6" \
        --n_test 1000 \
        --outdir "$7" \
        --method "$8"
' "$wd" "$dist" {} "$cln" "$pnr" "$alpha" "$nsamp" "$OUTDIR" "$method"

echo "  Simulation data generation completed for cluster$cln"
