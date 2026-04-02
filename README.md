# `calib_pipeline`
### Gene-specific & Domain-aggregate Calibration Framework for Variant Effect Prediction  

---

# ⭐ Overview

`calib_pipeline` implements a four-stage calibration workflow for gene-specific and cluster-aware posterior probability estimation of variant pathogenicity scores (e.g., AlphaMissense, REVEL, MutPred2).

The pipeline:

- Generates simulation data  
- Runs local and other post-hoc calibration  
- Performs bootstrap-based uncertainty quantification  
- Produces final gene-specific posterior probabilities used for ACMG/AMP PP3/BP4 evidence code assignment.

Supported calibration models:

- Local posterior calibration (Pejaver et al. 2022)
- Platt / Weighted-Platt
- Isotonic / Smoothed-Isotonic
- Beta Calibration
- SplineCalib
- Beta mixture
- Truncated normal mixture
- Skew-normal mixture from MAVE calibration
- MonoPostNN (monotonic posterior neural network)

---

# ⭐ Pipeline Steps

---

## **Step 00 — Simulation Data Generation (`calib_step00.py`)**

Creates a `.pkl` simulation file per gene × predictor, including:

- Labeled + unlabeled data  
- True posterior  
- Predictor values  
- Calibrate/test partitions  
- Metadata and simulation parameters  

Run:

```bash
python -m calib_pipeline.calib_step00 \
    --gene BRCA1 \
    --predictor REVEL \
    --prior 0.072 \
    --outdir /path/to/output \
    --labeled /data/BRCA1_REVEL_labeled.txt \
    --unlabeled /data/BRCA1_REVEL_unlabeled.txt
```

## **Step 01 — Run Calibration (`calib_step01.py`)**

Runs all calibration methods (local + others) on the simulation data for the target gene:

- Computes calibrated posterior estimates for each simulated dataset
- Automatically skips methods whose output files already exist

Run:

```bash
python -m calib_pipeline.calib_step01 \
    --gene BRCA1 \
    --predictor REVEL \
    --alpha 0.072 \
    --base_dir /path/to/output \
    --seed ${SEED}
```

## **Step 02 — Compute Metrics (`calib_step02.py`)**

Evaluates the performance of each calibration model using the true posterior from Step 00.

Metrics include:
- Overall average misestimation
- PP3/BP4 misclassification rate

Output files (e.g., ave_misests_combined.csv) are later used by Step 03 to determine the best calibration model.

Run:

```bash
python -m calib_pipeline.calib_step02 \
    --gene BRCA1 \
    --predictor REVEL \
    --prior 0.072 \
    --outdir /path/to/output
```

## **Step 03 — Final Calibration (`calib_step03.py`)**

Uses Step 02 metrics to select the best-performing calibration method, then:
1. Runs 1000 bootstrap replicates
   - For each replicate:
     - Bootstrap the labeled data
     - Predict on out-of-bag labeled + unlabeled points
     - Generate posterior curves

2. Computes posterior percentile bands
   - Pathogenic (P-side) 5%
   - Benign (B-side) 5%

3. Produces final gene-level posterior files:
   - {gene}_{predictor}_oob_posterior_{best_calibration_model}.csv
   - {gene}_{predictor}_finalposterior_{best_calibration_model}.csv
   - Diagnostic plots for publication


Run:

```bash
python -m calib_pipeline.calib_step03 \
    --gene BRCA1 \
    --predictor REVEL \
    --prior 0.072 \
    --outdir /path/to/output
```

# ⭐ Environment
Install:
```bash
conda env create -f environment.yml
conda activate calibrate
```
