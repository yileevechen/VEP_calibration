#!/usr/bin/env python
import os
import sys
import pickle
import bisect
import numpy as np
import pandas as pd
import argparse

# ---------------------------------------------------------------------
# Ensure GitHub repo is available
# ---------------------------------------------------------------------
GITHUB_REPO_DIR = "/tmp/clingen-svi-comp_calibration_python"
REPO_URL = "https://github.com/pejaverlab/clingen-svi-comp_calibration_python"

if not os.path.exists(GITHUB_REPO_DIR):
    # Clone repo if not present
    print(f"[INFO] Cloning repo from {REPO_URL} to {GITHUB_REPO_DIR} ...")
    import subprocess
    subprocess.run(["git", "clone", REPO_URL, GITHUB_REPO_DIR], check=True)

if GITHUB_REPO_DIR not in sys.path:
    sys.path.append(GITHUB_REPO_DIR)

# Now import
from Tavtigian.tavtigianutils import get_tavtigian_c, get_tavtigian_thresholds, get_tavtigian_plr
from LocalCalibration.gaussiansmoothing import *
from Tavtigian.Tavtigian import LocalCalibrateThresholdComputation
from LocalCalibration.LocalCalibration import LocalCalibration


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _lininterpol(x, x1, x2, y1, y2):
    y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
    return y

def _get_prob_linear_interp(s, scores, posterior):
    ix = bisect.bisect_left(scores, s)
    if ix == 0:
        ans = _lininterpol(s, scores[0], scores[1], posterior[0], posterior[1])
    elif ix == len(scores):
        ans = _lininterpol(s, scores[-1], scores[-2], posterior[-1], posterior[-2])
    else:
        ans = _lininterpol(s, scores[ix - 1], scores[ix], posterior[ix - 1], posterior[ix])
    if ans < 0.0:
        return 0.0
    if ans > 1.0:
        return 1.0
    return ans

def _get_probs_linear_interp(scores, thresholds, posteriors):
    return np.array([_get_prob_linear_interp(e, thresholds, posteriors) for e in scores])

def _calibrate_model(scores, labels, pudata, alpha, win_frac, gnfrac):
    scores = np.asarray(scores).flatten()
    labels = np.asarray(labels).flatten()
    pudata = np.asarray(pudata).flatten()

    win_size = int(win_frac * len(scores))
    print(f"\n[Local] window size for train data size {len(scores)} and alpha {alpha} is {win_size}.\n")

    calib = LocalCalibration(alpha, False, win_size, gnfrac, False, True)
    thresh, posteriors_p = calib.fit(scores, labels, pudata, alpha)

    c = get_tavtigian_c(alpha)
    calib_boot = LocalCalibrateThresholdComputation(alpha, c, False, win_size, gnfrac, False, True)
    _, posteriors_p_bootstrap = calib_boot.get_both_bootstrapped_posteriors_parallel(
        scores, labels, pudata, 1000, alpha, thresh
    )

    all_pathogenic = np.row_stack((posteriors_p, posteriors_p_bootstrap))
    all_benign = 1 - np.flip(all_pathogenic, axis=1)

    pathogenic5 = np.percentile(all_pathogenic[1:], 5, axis=0)
    benign5 = np.percentile(all_benign[1:], 5, axis=0)
    pathogenic50 = np.percentile(all_pathogenic[1:], 50, axis=0)
    benign50 = np.percentile(all_benign[1:], 50, axis=0)

    thresh = list(thresh)
    pathogenic = list(np.flip(posteriors_p))
    pathogenic5 = list(np.flip(pathogenic5))
    pathogenic50 = list(np.flip(pathogenic50))
    benign5 = list(np.flip(benign5))
    benign50 = list(np.flip(benign50))

    return thresh, pathogenic, pathogenic5, benign5, pathogenic50, benign50


def _local_calibration(
    y_calibrate,
    y_calibrate_pred_prob,
    y_test_pred_prob,
    y_unlabeled_pred_prob,
    alpha,
    win_frac,
    gnfrac,
):
    thresh, local_calib, p5, b5, p50, b50 = _calibrate_model(
        y_calibrate_pred_prob, y_calibrate, y_unlabeled_pred_prob, alpha, win_frac, gnfrac
    )

    local_calibrated_prob_test = _get_probs_linear_interp(y_test_pred_prob, thresh, local_calib)
    local_calibrated_prob_test_p5 = _get_probs_linear_interp(y_test_pred_prob, thresh, p5)
    local_calibrated_prob_test_b5 = _get_probs_linear_interp(np.flip(y_test_pred_prob), thresh, b5)
    local_calibrated_prob_test_p50 = _get_probs_linear_interp(y_test_pred_prob, thresh, p50)
    local_calibrated_prob_test_b50 = _get_probs_linear_interp(np.flip(y_test_pred_prob), thresh, b50)

    return (
        local_calibrated_prob_test.flatten(),
        local_calibrated_prob_test_p5.flatten(),
        local_calibrated_prob_test_b5.flatten(),
        local_calibrated_prob_test_p50.flatten(),
        local_calibrated_prob_test_b50.flatten(),
    )


# ---------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------
def run_local_calibration(
    predictor,
    method,
    seed_raw,
    outdir_base,
    alpha,
    n_calibrate,
    n_test,
    gene,
    pnratio_calibrate,
    pnratio_test,
):
    seed_scaled = int(seed_raw) * 828
    tag = seed_scaled / 828.0

    simdir = os.path.join(outdir_base, f"{predictor}_{gene}_{method}_Ntrain{n_calibrate}")
    os.makedirs(simdir, exist_ok=True)

    base_fname = f"{predictor}_simu_{method}{tag}"
    out_csv = os.path.join(simdir, base_fname + "_calib_outputs.csv")
    out_p95 = os.path.join(simdir, base_fname + "_calib_outputs_P95.csv")
    out_b95 = os.path.join(simdir, base_fname + "_calib_outputs_B95.csv")
    out_p50 = os.path.join(simdir, base_fname + "_calib_outputs_P50.csv")
    out_b50 = os.path.join(simdir, base_fname + "_calib_outputs_B50.csv")

    if all(os.path.exists(f) for f in [out_csv, out_p95, out_b95]):
        print(f"[Local] Outputs exist for {gene}, {predictor}, seed {seed_raw}. Skipping.")
        return

    pkl_path = os.path.join(simdir, base_fname + ".pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Simulation file not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        simudat = pickle.load(f)

    y_calibrate_pred_prob = np.asarray(simudat["y_calibrate_pred_prob"]).flatten()
    y_calibrate = np.asarray(simudat["y_calibrate"]).flatten()
    y_test_pred_prob = np.asarray(simudat["y_test_pred_prob"]).flatten()
    y_unlabelled_pred_prob = np.asarray(simudat["y_unlabelled_pred_prob"]).flatten()
    true_posterior = np.asarray(simudat["true_posterior"]).flatten()

    local_mean = {}
    local_p5 = {}
    local_b5 = {}
    local_p50 = {}
    local_b50 = {}

    # Grid over (win_frac, gnfrac) — names must match what calib_step03 expects
    for win_frac in [0.1, 0.2, 0.3]:
        for gnfrac in [0.0, 0.03, 0.06]:
            key = f"local_{win_frac}_{gnfrac}"
            (
                mean_vals,
                p5_vals,
                b5_vals,
                p50_vals,
                b50_vals,
            ) = _local_calibration(
                y_calibrate,
                y_calibrate_pred_prob,
                y_test_pred_prob,
                y_unlabelled_pred_prob,
                alpha,
                win_frac,
                gnfrac,
            )
            local_mean[key] = mean_vals
            local_p5[key] = p5_vals
            local_b5[key] = b5_vals
            local_p50[key] = p50_vals
            local_b50[key] = b50_vals

    # P-side
    df_mean = pd.DataFrame(local_mean)
    df_mean["True"] = true_posterior
    df_mean.to_csv(out_csv, index=False)

    df_p5 = pd.DataFrame(local_p5)
    df_p5["True"] = true_posterior
    df_p5.to_csv(out_p95, index=False)

    # B-side (flipped)
    df_b5 = pd.DataFrame(local_b5)
    df_b5["True"] = 1 - np.flip(true_posterior)
    df_b5.to_csv(out_b95, index=False)

    df_p50 = pd.DataFrame(local_p50)
    df_p50["True"] = true_posterior
    df_p50.to_csv(out_p50, index=False)

    df_b50 = pd.DataFrame(local_b50)
    df_b50["True"] = 1 - np.flip(true_posterior)
    df_b50.to_csv(out_b50, index=False)

    print(f"[Local] Saved outputs for {gene}, {predictor}, seed {seed_raw} to {simdir}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def _build_argparser():
    parser = argparse.ArgumentParser(
        description="Run local calibration for a single gene/predictor/method/seed."
    )
    parser.add_argument("--seed", type=int, required=True, help="Raw seed index")
    parser.add_argument("--predictor", type=str, required=True, help="Predictor name (e.g. AM, REVEL, MP2)")
    parser.add_argument("--gene", type=str, required=True, help="Gene name")
    parser.add_argument("--pnratio_calibrate", type=float, required=True)
    parser.add_argument("--pnratio_test", type=float, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--n_calibrate", type=int, required=True)
    parser.add_argument("--n_test", type=int, required=True)
    parser.add_argument("--outdir", type=str, required=True, help="Base output directory")
    parser.add_argument("--method", type=str, required=True, help="Simulation method tag")
    return parser

def main():
    parser = _build_argparser()
    args = parser.parse_args()

    run_local_calibration(
        predictor=args.predictor,
        method=args.method,
        seed_raw=args.seed,
        outdir_base=args.outdir,
        alpha=args.alpha,
        n_calibrate=args.n_calibrate,
        n_test=args.n_test,
        gene=args.gene,
        pnratio_calibrate=args.pnratio_calibrate,
        pnratio_test=args.pnratio_test,
    )

if __name__ == "__main__":
    main()
