#!/bin/bash
#BSUB -P acc_hpcstaff
#BSUB -q premium
#BSUB -n 1
#BSUB -W 01:00
#BSUB -J calib_step02[1]
#BSUB -e "logs/calib_step02.%J.%I.err"
#BSUB -o "logs/calib_step02.%J.%I.out"

ARRAY_IDX=${LSB_JOBINDEX}

source activate calibrate_para

python -m calib_pipeline.calib_step02 "$ARRAY_IDX"
