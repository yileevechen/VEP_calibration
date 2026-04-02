#!/bin/bash
#BSUB -P acc_hpcstaff
#BSUB -q premium
#BSUB -n 1
#BSUB -W 06:00
#BSUB -J calib_step00
#BSUB -e "logs/calib_step00.%J.err"
#BSUB -o "logs/calib_step00.%J.out"

source activate calibrate

python -m calib_pipeline.calib_step00 \
    --gene BRCA1 \
    --predictor REVEL \
    --prior 0.072 \
    --outdir /path/to/output \
    --labeled /data/BRCA1_REVEL_labeled.txt \
    --unlabeled /data/BRCA1_REVEL_unlabeled.txt
