#!/bin/bash
#BSUB -P acc_hpcstaff
#BSUB -q premium
#BSUB -n 1
#BSUB -W 01:00
#BSUB -J calib_step02
#BSUB -e "logs/calib_step02.%J.%I.err"
#BSUB -o "logs/calib_step02.%J.%I.out"

source activate calibrate_para

python -m calib_pipeline.calib_step02 \
    --gene BRCA1 \
    --predictor REVEL \
    --prior 0.072 \
    --outdir /path/to/output
