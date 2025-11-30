#!/bin/bash
#BSUB -P acc_hpcstaff
#BSUB -q premium
#BSUB -n 1
#BSUB -W 06:00
#BSUB -J calib_step00[1]
#BSUB -e "logs/calib_step00.%J.%I.err"
#BSUB -o "logs/calib_step00.%J.%I.out"

ARRAY_IDX=${LSB_JOBINDEX}

source activate calibrate

python -m calib_pipeline.calib_step00 "$ARRAY_IDX"
