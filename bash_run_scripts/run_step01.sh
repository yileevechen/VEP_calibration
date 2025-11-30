#!/bin/bash
#BSUB -P acc_hpcstaff
#BSUB -q premium
#BSUB -n 10
#BSUB -W 06:00
#BSUB -J calib_step01[1]
#BSUB -e "logs/calib_step01.%J.%I.err"
#BSUB -o "logs/calib_step01.%J.%I.out"

ARRAY_IDX=${LSB_JOBINDEX}

source activate calibrate_para

python -m calib_pipeline.calib_step01 "$ARRAY_IDX"
