#!/bin/bash
#BSUB -P acc_hpcstaff
#BSUB -q premium
#BSUB -n 16
#BSUB -W 08:00
#BSUB -J calib_step03[1]
#BSUB -e "logs/calib_step03.%J.%I.err"
#BSUB -o "logs/calib_step03.%J.%I.out"

ARRAY_IDX=${LSB_JOBINDEX}

source activate calibrate_para

python -m calib_pipeline.calib_step03 "$ARRAY_IDX"
