#!/bin/bash
#BSUB -P acc_hpcstaff
#BSUB -q premium
#BSUB -n 16
#BSUB -W 08:00
#BSUB -J calib_step03
#BSUB -e "logs/calib_step03.%J.err"
#BSUB -o "logs/calib_step03.%J.out"

source activate calibrate_para

python -m calib_pipeline.calib_step03 \
    --gene BRCA1 \
    --predictor REVEL \
    --prior 0.12 \
    --outdir /path/to/output
