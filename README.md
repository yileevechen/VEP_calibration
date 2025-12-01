# `calib_pipeline`
### Single-Gene & Cluster-Aware Calibration Framework for Variant Effect Prediction  

---

# ⭐ Overview
flowchart TD

    A00[Step 00: Simulation Data Generation<br/>calib_step00.py<br/><br/>• Generate gene×predictor simulation data<br/>• Labeled + unlabeled sets<br/>• True posterior<br/>• Store .pkl files] 

    A01[Step 01: Run Calibration Models<br/>calib_step01.py<br/><br/>• Run all models:<br/>  - Local<br/>  - Platt / Weighted-Platt<br/>  - Isotonic / SmoothIso<br/>  - Beta / BetaMixture<br/>  - TruncNorm<br/>  - MixSkewNorm<br/>  - SplineCalib<br/>  - MonoPostNN<br/>• Output per-method calibrated posteriors]

    A02[Step 02: Metric Computation<br/>calib_step02.py<br/><br/>• Compare each method to true posterior<br/>• Compute metrics:<br/>  - Misestimation<br/>  - PP3/BP4 errors<br/>  - Percentile errors<br/>  - Combined ranks<br/>• Output *_combined.csv files]

    A03[Step 03: Final Calibration<br/>calib_step03.py<br/><br/>• Select best calibration method<br/>• 1000 bootstrap OOB posteriors<br/>• MonoPostNN OOB bootstrap<br/>• Compute P95/P50/B95/B50 envelopes<br/>• Generate final posterior curves & plots]

    A00 --> A01 --> A02 --> A03

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
python -m calib_pipeline.calib_step00 ARRAY_IDX
```

## **Step 01 — Run Calibration (`calib_step01.py`)**

Runs all calibration methods (local + others) on the simulation data for the target gene:

- Computes calibrated posterior estimates for each simulated dataset
- Automatically skips methods whose output files already exist

Run:

```bash
python -m calib_pipeline.calib_step01 ARRAY_IDX
```

## **Step 02 — Compute Metrics (`calib_step02.py`)**

Evaluates the performance of each calibration model using the true posterior from Step 00.

Metrics include:
- Overall average misestimation
- PP3/BP4 misclassification rate

Output files (e.g., ave_misests_combined.csv) are later used by Step 03 to determine the best calibration model.

Run:

```bash
python -m calib_pipeline.calib_step02 ARRAY_IDX
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
python -m calib_pipeline.calib_step03 ARRAY_IDX
```
