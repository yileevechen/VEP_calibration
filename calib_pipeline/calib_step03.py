import os
import re
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator
from scipy.stats import beta, truncnorm
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

import torch

# -------------------------------------------------------------------
# Imports for Tavtigian + LocalCalibration + MonoPostNN
# -------------------------------------------------------------------
TAV_BASE = (
    "/sc/arion/projects/pejaverlab/IGVF/users/cheny60/analysis/"
    "calibrationexp-main/calib_decision_tree/clingen-svi-comp_calibration_python-master"
)
if TAV_BASE not in sys.path:
    sys.path.append(TAV_BASE)

from Tavtigian.tavtigianutils import (
    get_tavtigian_c,
    get_tavtigian_thresholds,
    get_tavtigian_plr,
)
from calib_pipeline.local_calib import _get_prob_linear_interp
from Tavtigian.Tavtigian import LocalCalibrateThresholdComputation
from LocalCalibration.LocalCalibration import LocalCalibration

sys.path.append(
    "/sc/arion/projects/pejaverlab/IGVF/users/cheny60/analysis/calibrationexp-main/PosteriorCalibration-master"
)
from MonotonicPosterior.computePosterior_fast import computePosteriorFromEnsemble

import ml_insights as mli
from betacal import BetaCalibration

sys.path.append(
    "/sc/arion/projects/pejaverlab/IGVF/users/cheny60/analysis/calibrationexp-main/mave_calibration-master"
)
from mave_calibration.main import single_fit
from mave_calibration.skew_normal.density_utils import joint_densities

# global used by multiprocessing initializer
_BOOT_GLOBALS = {}

# how many cores to use for Pool
N_PROCS = 16  # set to match your BSUB -n


# -------------------------------------------------------------------
# Small helpers that do NOT depend on gene/dist
# -------------------------------------------------------------------
def rm_monoincrease_transform(posterior):
    """Fix minor non-monotonic wiggles for posterior curves."""
    def check_monotonic(lst):
        if all(x <= y for x, y in zip(lst, lst[1:])):
            return True
        if all(x >= y for x, y in zip(lst, lst[1:])):
            return True
        return False

    posterior = np.array(posterior, dtype=float)
    if check_monotonic(posterior):
        return posterior

    min_idx = np.argmin(posterior[0:100])
    posterior[:min_idx] = posterior[min_idx]

    max_idx = np.argmax(posterior)
    posterior[max_idx + 1:] = np.maximum.accumulate(posterior[max_idx + 1:])
    posterior[max_idx + 1:] = posterior[max_idx]
    return posterior


def rm_force_monodecreasing(lst):
    """Force a decreasing curve (for benign side)."""
    lst = np.array(lst, dtype=float)
    max_idx = np.argmax(lst[0:100])
    lst[:max_idx] = np.max(lst[0:100])
    for i in range(1, len(lst)):
        if lst[i] > lst[i - 1]:
            lst[i] = lst[i - 1]
    return lst


def intersect_lists(*lists):
    if not lists:
        return []
    return list(set(lists[0]).intersection(*lists[1:]))


def getMonoCalibNN(X, y, X_test, alpha, num_ensemble=15, epochs=300):
    """Fit MonoPost ensemble on real labeled data and predict on X_test grid."""
    modelAvg_func, modelMedian_func, modelRobustMean_func, Ensemble = computePosteriorFromEnsemble(
        X,
        y,
        alpha=alpha,
        test_size=0.0,
        num_ensemble=num_ensemble,
        epochs=epochs,
    )
    X_test = np.array(X_test, dtype=np.float32).reshape(-1, 1)
    X_torch = torch.tensor(X_test, dtype=torch.float32)
    postRobustAvg = modelRobustMean_func(X_torch)
    return postRobustAvg


def transform(data_prior=0.1, est_prior=0.1, posterior=None):
    """Reweight posterior to target prior (Tavtigian-compatible prior adjustment)."""
    posterior = np.array(posterior, dtype=float)
    posterior[posterior == 1] = 1 - 1e-10
    gamma_x = (
        (est_prior / (1 - est_prior))
        * ((1 - data_prior) / data_prior)
        * (posterior / (1 - posterior))
    )
    return gamma_x / (1 + gamma_x)


def getSplineCalibProbs(y_calibrate, y_calibrate_pred_nn_prob, y_test_pred_nn_prob):
    splinecalib = mli.SplineCalib()
    splinecalib.fit(y_calibrate_pred_nn_prob, y_calibrate)
    return splinecalib.predict(y_test_pred_nn_prob)


def getMixGaussCalib(y_calibrate, y_calibrate_pred_nn_prob, yall_pred, priors):
    """Skew-normal mixture (MixSkewNorm) calibration."""
    sample_names = [1, 0]
    sample_indicators = np.zeros(
        (y_calibrate_pred_nn_prob.shape[0], len(sample_names)), dtype=bool
    )
    for i, sample_name in enumerate(sample_names):
        sample_indicators[:, i] = y_calibrate.flatten() == sample_name

    bestFitres = single_fit(
        y_calibrate_pred_nn_prob,
        sample_indicators,
        max_iters=1000,
        n_inits=100,
        verbose=False,
    )

    class AttrDict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(f"'AttrDict' object has no attribute '{name}'")

        def __setattr__(self, name, value):
            self[name] = value

    bestFit = AttrDict(bestFitres)
    f_P = joint_densities(
        yall_pred, bestFit["component_params"], bestFit["weights"][0]
    ).sum(0)
    f_B = joint_densities(
        yall_pred, bestFit["component_params"], bestFit["weights"][1]
    ).sum(0)
    P = f_P / f_B
    posteriors = P * priors / ((P - 1) * priors + 1)
    return posteriors


def fit_truncnorm_mixture(classes, predictions, test_preds):
    def clean_data(data):
        data = np.array(data)
        data = data[~np.isnan(data)]
        data = data[(data > 0) & (data < 1)]
        return data

    def fit_truncnorm(data):
        def negative_log_likelihood(params):
            mu, sigma = params
            if sigma <= 0:
                return np.inf
            dist = truncnorm(
                (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma
            )
            return -np.sum(dist.logpdf(data))

        lower = 0
        upper = 1
        init_params = [np.mean(data), np.std(data)]
        bounds = [(lower, upper), (1e-3, None)]
        result = minimize(
            negative_log_likelihood, init_params, bounds=bounds, method="L-BFGS-B"
        )
        return {"mu": result.x[0], "sigma": result.x[1]}

    lower = 0
    upper = 1
    pos_sample = clean_data(predictions[classes == 1])
    neg_sample = clean_data(predictions[classes == 0])

    plp_truncnorm_params = fit_truncnorm(pos_sample)
    blb_truncnorm_params = fit_truncnorm(neg_sample)

    pos_dist = truncnorm(
        (lower - plp_truncnorm_params["mu"]) / plp_truncnorm_params["sigma"],
        (upper - plp_truncnorm_params["mu"]) / plp_truncnorm_params["sigma"],
        loc=plp_truncnorm_params["mu"],
        scale=plp_truncnorm_params["sigma"],
    )

    neg_dist = truncnorm(
        (lower - blb_truncnorm_params["mu"]) / blb_truncnorm_params["sigma"],
        (upper - blb_truncnorm_params["mu"]) / blb_truncnorm_params["sigma"],
        loc=blb_truncnorm_params["mu"],
        scale=blb_truncnorm_params["sigma"],
    )

    pos_probs = pos_dist.pdf(test_preds)
    neg_probs = neg_dist.pdf(test_preds)
    pos_neg_sum = pos_probs + neg_probs
    posteriors = np.divide(
        pos_probs, pos_neg_sum, out=np.zeros_like(pos_probs), where=pos_neg_sum != 0
    )
    return posteriors


def fit_beta_mixture(classes, predictions, test_preds):
    def clean_data(data):
        data = np.array(data)
        data = data[~np.isnan(data)]
        data = data[(data > 0) & (data < 1)]
        return data

    pos_sample = clean_data(predictions[classes == 1])
    neg_sample = clean_data(predictions[classes == 0])

    plp_beta_params = beta.fit(pos_sample, floc=0, fscale=1)
    blb_beta_params = beta.fit(neg_sample, floc=0, fscale=1)

    beta_params = {
        "positives": {"alpha": plp_beta_params[0], "beta": plp_beta_params[1]},
        "negatives": {"alpha": blb_beta_params[0], "beta": blb_beta_params[1]},
    }

    pos_pdf = beta(
        beta_params["positives"]["alpha"], beta_params["positives"]["beta"]
    ).pdf
    neg_pdf = beta(
        beta_params["negatives"]["alpha"], beta_params["negatives"]["beta"]
    ).pdf

    pos_probs = pos_pdf(test_preds)
    neg_probs = neg_pdf(test_preds)
    pos_neg_sum = pos_probs + neg_probs
    posteriors = pos_probs / pos_neg_sum
    print(f"!!!beta fitting paras: {beta_params}.\n")
    return posteriors


# -------------------------------------------------------------------
# Bootstrap helpers for non-MonoPost methods (top-level for multiprocessing)
# -------------------------------------------------------------------
def _to_numpy(x):
    """
    Safely convert either a torch.Tensor or a NumPy-like object to np.ndarray.
    Works whether computePosteriorFromEnsemble returns torch tensors or numpy arrays.
    """
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        # torch not available or x not a tensor
        pass

    return np.asarray(x)
    
def _bootstrap_initializer(X_lab_, y_lab_, X_unlab_, best_model_, alpha_, pnrat_):
    """Initializer for multiprocessing.Pool — stash shared data in globals."""
    global _BOOT_GLOBALS
    _BOOT_GLOBALS = {
        "X_lab": np.asarray(X_lab_, float),
        "y_lab": np.asarray(y_lab_, int),
        "X_unlab": np.asarray(X_unlab_, float),
        "best_model": best_model_,
        "alpha": float(alpha_),
        "pnrat": float(pnrat_),
    }


def _fit_and_predict_once(best_model, X_train, y_train, X_test, X_unlabel, alpha, pnrat):
    """
    Fit the chosen calibration model on (X_train, y_train),
    return calibrated posterior on X_test.
    """
    X_train = np.asarray(X_train, float)
    y_train = np.asarray(y_train, int)
    X_test = np.asarray(X_test, float)
    X_unlabel = np.asarray(X_unlabel, float)

    # strip suffixes if present
    best_model_core = re.sub(r"(_others|_local)$", "", best_model)

    if best_model_core == "MonoPostNN":
        # used only for one-time full curve, not for generic bootstrap anymore
        post = getMonoCalibNN(
            X_train.reshape(-1, 1),
            y_train,
            X_test,
            alpha=alpha,
            num_ensemble=15,
            epochs=300,
        )
        ##return post.detach().cpu().numpy().flatten()
        return _to_numpy(post).flatten()

    if best_model_core == "Beta":
        beta_reg = BetaCalibration(parameters="abm").fit(X_train, y_train)
        p = beta_reg.predict(X_test)
        return transform(data_prior=pnrat, est_prior=alpha, posterior=p)

    if best_model_core == "SplineCalib":
        p = getSplineCalibProbs(y_train, X_train, X_test)
        return transform(data_prior=pnrat, est_prior=alpha, posterior=p)

    if best_model_core == "Platt":
        logreg = LogisticRegression(
            C=99999999999, solver="lbfgs", class_weight={0: 0.5, 1: 0.5}
        )
        logreg.fit(X_train.reshape(-1, 1), y_train)
        p = logreg.predict_proba(X_test.reshape(-1, 1))[:, 1]
        return transform(data_prior=pnrat, est_prior=alpha, posterior=p)

    if best_model_core == "WeightedPlatt":
        w0 = (1 - alpha) / (2 * (1 - pnrat))
        w1 = alpha / (2 * pnrat)
        logreg = LogisticRegression(
            C=99999999999, solver="lbfgs", class_weight={0: w0, 1: w1}
        )
        logreg.fit(X_train.reshape(-1, 1), y_train)
        p = logreg.predict_proba(X_test.reshape(-1, 1))[:, 1]
        return p

    if best_model_core == "Isotonic":
        iso_reg = IsotonicRegression(out_of_bounds="clip").fit(X_train, y_train)
        p = iso_reg.predict(X_test)
        return transform(data_prior=pnrat, est_prior=alpha, posterior=p)

    if best_model_core == "SmoothIsotonic":
        predictions = X_train
        classes = y_train
        test_preds = X_test

        iso_reg = IsotonicRegression(out_of_bounds="clip")
        iso_reg.fit(predictions, classes)
        iso_preds_train = iso_reg.predict(predictions)

        sort_idx = np.argsort(predictions)
        x_sorted = predictions[sort_idx]
        y_sorted = iso_preds_train[sort_idx]

        num_bins = 15
        bins = np.linspace(0, 100, num_bins + 1)
        quantiles = np.percentile(x_sorted, bins)
        x_rep, y_rep = [], []
        for i in range(num_bins):
            mask = (x_sorted >= quantiles[i]) & (x_sorted < quantiles[i + 1])
            if np.sum(mask) > 0:
                x_rep.append(np.median(x_sorted[mask]))
                y_rep.append(np.median(y_sorted[mask]))
        pchip = PchipInterpolator(x_rep, y_rep)
        smopred = pchip(test_preds)
        smopred = np.clip(smopred, 1e-6, 1 - 1e-6)
        return transform(data_prior=pnrat, est_prior=alpha, posterior=smopred)

    if best_model_core == "MixSkewNorm":
        return getMixGaussCalib(
            y_calibrate=y_train,
            y_calibrate_pred_nn_prob=X_train,
            yall_pred=X_test,
            priors=alpha,
        )

    if best_model_core == "TruncNorm":
        trnorm_mix_post = fit_truncnorm_mixture(y_train, X_train, X_test)
        return transform(data_prior=pnrat, est_prior=alpha, posterior=trnorm_mix_post)

    if best_model_core == "BetaMixture":
        beta_mix_post = fit_beta_mixture(y_train, X_train, X_test)
        return transform(data_prior=pnrat, est_prior=alpha, posterior=beta_mix_post)

    if best_model_core == "Local":
        # winfrac / gnfrac are set later in run_final_calibration_for_gene
        raise RuntimeError("Local should not go through generic _fit_and_predict_once in bootstrap.")

    raise ValueError(f"Unsupported best_model for bootstrap: {best_model}")


def _run_single_bootstrap(seed):
    """
    One bootstrap replicate for NON-MonoPost models:
      - resample labeled (with replacement)
      - OOB labeled + all unlabeled as test set

    Returns:
      lab_post_full: length n_lab, NaN except at OOB indices
      unlab_post:    length n_unlab, posteriors for all unlabeled
    """
    g = _BOOT_GLOBALS
    X_lab = g["X_lab"]
    y_lab = g["y_lab"]
    X_unlab = g["X_unlab"]
    best_model = g["best_model"]
    alpha = g["alpha"]
    pnrat = g["pnrat"]

    rng = np.random.default_rng(seed*828)
    n = len(X_lab)

    # bootstrap indices with replacement
    sample_idx = rng.integers(0, n, size=n)
    unique_sample = np.unique(sample_idx)

    # out-of-bag mask
    oob_mask = np.ones(n, dtype=bool)
    oob_mask[unique_sample] = False

    if not np.any(oob_mask):
        # degenerate: everyone is in-sample
        return None

    X_train = X_lab[sample_idx]
    y_train = y_lab[sample_idx]

    lab_oob_idx = np.where(oob_mask)[0]
    X_test_lab = X_lab[lab_oob_idx]
    X_test_unlab = X_unlab
    X_test_all = np.concatenate([X_test_lab, X_test_unlab])

    post_all = _fit_and_predict_once(
        best_model=best_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test_all,
        X_unlabel=X_test_unlab,
        alpha=alpha,
        pnrat=pnrat,
    )

    n_lab_oob = len(X_test_lab)
    lab_post = post_all[:n_lab_oob]
    unlab_post = post_all[n_lab_oob:]

    lab_post_full = np.full(n, np.nan, dtype=float)
    lab_post_full[lab_oob_idx] = lab_post

    return lab_post_full, unlab_post


# -------------------------------------------------------------------
# SPECIAL: True OOB bootstrap for MonoPostNN
# -------------------------------------------------------------------
def _monopost_oob_bootstrap(X_lab, y_lab, X_unlab, alpha,
                            B=200, num_ensemble=5, epochs=300, seed=828):
    """
    Proper OOB bootstrap for MonoPostNN:

      For each bootstrap b:
        - draw a bootstrap sample of labeled data
        - train MonoPostNN ensemble on that sample
        - predict robust-mean posterior for all (labeled + unlabeled)
        - collect predictions for:
            * labeled variants that are OOB in this bootstrap
            * all unlabeled variants (they are always "OOB")

      Finally:
        - labeled: 5th percentile across bootstraps where each variant was OOB
        - unlabeled: 5th percentile across all bootstraps
    """
    X_lab = np.asarray(X_lab, float)
    y_lab = np.asarray(y_lab, int)
    X_unlab = np.asarray(X_unlab, float)

    n_lab = len(X_lab)
    n_unlab = len(X_unlab)

    #rng = np.random.default_rng(seed)

    # store predictions per labeled variant
    lab_preds_list = [[] for _ in range(n_lab)]
    # store predictions across bootstraps for unlabeled
    unlab_preds_all = []

    X_all = np.concatenate([X_lab, X_unlab]).astype(float)

    for b in range(B):
        # bootstrap indices with replacement from labeled
        rng = np.random.default_rng(seed * 5 * b)
        sample_idx = rng.integers(0, n_lab, size=n_lab)
        unique_sample = np.unique(sample_idx)

        # OOB mask
        oob_mask = np.ones(n_lab, dtype=bool)
        oob_mask[unique_sample] = False

        if not np.any(oob_mask):
            continue

        X_boot = X_lab[sample_idx]
        y_boot = y_lab[sample_idx]

        # train MonoPostNN ensemble on this bootstrap sample
        post_robust = getMonoCalibNN(
            X_boot.reshape(-1, 1),
            y_boot,
            X_all,
            alpha=alpha,
            num_ensemble=num_ensemble,
            epochs=epochs,
        )
        post_robust = _to_numpy(post_robust).flatten() ## post_robust.detach().cpu().numpy().flatten()

        # split back into labeled/unlabeled
        lab_post = post_robust[:n_lab]
        unlab_post = post_robust[n_lab:]

        # store labeled predictions only for OOB indices
        oob_idx = np.where(oob_mask)[0]
        for idx in oob_idx:
            lab_preds_list[idx].append(float(lab_post[idx]))

        # unlabeled are always "OOB"
        unlab_preds_all.append(unlab_post)

    # aggregate 5th percentile
    lab_p95 = np.full(n_lab, np.nan)
    for i in range(n_lab):
        arr = np.array(lab_preds_list[i], dtype=float)
        if arr.size > 0:
            lab_p95[i] = np.percentile(arr, 5)

    if len(unlab_preds_all) > 0:
        unlab_stack = np.vstack(unlab_preds_all)
        unlab_p95 = np.percentile(unlab_stack, 5, axis=0)
    else:
        unlab_p95 = np.full(n_unlab, np.nan)

    return lab_p95, unlab_p95


# -------------------------------------------------------------------
# Core function: can be called from main() using ARRAY_IDX logic
# -------------------------------------------------------------------
BASE_PATH = (
    "/sc/arion/projects/pejaverlab/IGVF/users/cheny60/analysis/"
    "calibrationexp-main/single_gene_calibration_pipeline"
)


def run_final_calibration_for_gene(gene, dist, alpha, pnrat, method):
    """
    Final calibration for a single gene/dist:
      * selects best model (including Local)
      * runs bootstrap OOB calibration
      * writes OOB p95 CSV
      * fits one-time full posterior curve
      * plots posteriors + Tavtigian thresholds
    """
    METRIC_DIR = f"{BASE_PATH}/{gene}/{dist}_{method}_calib_metric"

    # Path to labeled / unlabeled ClinVar + gnomAD score files
    labfn = f"{BASE_PATH}/{gene}/{gene}_{dist}_labeled.txt"
    labdat = pd.read_table(labfn, header=None)

    unlabfn = f"{BASE_PATH}/{gene}/{gene}_{dist}_unlabeled.txt"
    unlabdat = pd.read_table(unlabfn, header=None)

    # Log file for this gene/dist
    logfn = f"{BASE_PATH}/{gene}/{gene}_specific_calibration_output_sequential_oob_p95.txt"

    # Tavtigian posterior thresholds
    c = get_tavtigian_c(alpha)
    Post_p, Post_b = get_tavtigian_thresholds(c, alpha)

    with open(logfn, "a") as f:
        f.write("\n__________________________________\n")
        f.write(f"\n\n{gene} {dist} information:\n")
        f.write(f"alpha: {alpha}; c: {c}\n")
        f.write(f"Post_p: {Post_p}, Post_b: {Post_b}\n")

    # ------------------------
    # Best-method selection
    # ------------------------
    def get_best_calib():
        """
        Select best calibration method using the *_combined.csv metric files.

        Returns
        -------
        local_para : str
            Name of the best "local" method (column name containing 'local'),
            used later to derive winfrac / gnfrac.
        best_method_all : str
            Best overall method across all columns.
        """
        # === Load combined metric tables ===
        ave_calib = pd.read_csv(f"{METRIC_DIR}/ave_misests_combined.csv", index_col=0)

        pp3_p50 = pd.read_csv(f"{METRIC_DIR}/pp3_50ps_combined.csv", index_col=0)
        pp3_p75 = pd.read_csv(f"{METRIC_DIR}/pp3_75ps_combined.csv", index_col=0)
        pp3_p90 = pd.read_csv(f"{METRIC_DIR}/pp3_90ps_combined.csv", index_col=0)
        pp3_max = pd.read_csv(f"{METRIC_DIR}/pp3_maxs_combined.csv", index_col=0)
        pp3_frac = pd.read_csv(f"{METRIC_DIR}/pp3_fracs_combined.csv", index_col=0)

        bp4_p50 = pd.read_csv(f"{METRIC_DIR}/bp4_50ps_combined.csv", index_col=0)
        bp4_p75 = pd.read_csv(f"{METRIC_DIR}/bp4_75ps_combined.csv", index_col=0)
        bp4_p90 = pd.read_csv(f"{METRIC_DIR}/bp4_90ps_combined.csv", index_col=0)
        bp4_max = pd.read_csv(f"{METRIC_DIR}/bp4_maxs_combined.csv", index_col=0)
        bp4_frac = pd.read_csv(f"{METRIC_DIR}/bp4_fracs_combined.csv", index_col=0)

        # ------------------------------------------------------------------
        # 2a. Identify "local" methods from column names (e.g. 'local_win0.3_fn0.06')
        # ------------------------------------------------------------------
        all_methods = list(ave_calib.columns)
        local_methods = [m for m in all_methods if "local" in m.lower()]

        with open(logfn, "a") as f:
            f.write(f"All methods in metrics: {all_methods}\n")
            f.write(f"Detected local methods (name contains 'local'): {local_methods}\n")

        if len(local_methods) == 0:
            local_methods = all_methods[:]
            with open(logfn, "a") as f:
                f.write("WARNING: No methods with 'local' in name; using all methods as local.\n")

        ave_local = ave_calib[local_methods]

        # Rank by average mis-estimation: smaller is better
        ori_top3_local = (
            ave_local.rank(axis=1, method="min")
            .mean(axis=0)
            .sort_values()
            .head(3)
            .index
            .tolist()
        )
        with open(logfn, "a") as f:
            f.write(f"The ori_top3_methods for Local: {ori_top3_local}\n")

        # Quality filters for local methods using PP3/BP4 metrics
        p2_meth_loc = pp3_p75[local_methods].loc[
            :, pp3_p75[local_methods].quantile(0.75, axis=0) < 2
        ].columns.tolist()
        p4_meth_loc = pp3_max[local_methods].loc[
            :, pp3_max[local_methods].quantile(0.75, axis=0) < 3
        ].columns.tolist()

        b2_meth_loc = bp4_p75[local_methods].loc[
            :, bp4_p75[local_methods].quantile(0.75, axis=0) < 2
        ].columns.tolist()
        b4_meth_loc = bp4_max[local_methods].loc[
            :, bp4_max[local_methods].quantile(0.75, axis=0) < 3
        ].columns.tolist()

        print(f"Local: p2_meth: {p2_meth_loc}; p4_meth: {p4_meth_loc}")
        print(f"Local: b2_meth: {b2_meth_loc}; b4_meth: {b4_meth_loc}")

        local_qual_meths = intersect_lists(
            p2_meth_loc, p4_meth_loc, b2_meth_loc, b4_meth_loc
        )
        with open(logfn, "a") as f:
            f.write(f"The qual_meths for Local: {local_qual_meths}\n")

        local_final_qual_meths = intersect_lists(local_qual_meths, ori_top3_local)
        with open(logfn, "a") as f:
            f.write(f"The final_qual_meths for Local: {local_final_qual_meths}\n")

        # Ranking for LOCAL subset only
        pp3_ranks_local = pp3_frac[local_methods].rank(axis=1, method="min").mean(axis=0)
        bp4_ranks_local = bp4_frac[local_methods].rank(axis=1, method="min").mean(axis=0)
        combined_rank_local = (pp3_ranks_local + bp4_ranks_local) / 2

        if len(local_final_qual_meths) > 0:
            best_method_local = (
                combined_rank_local[local_final_qual_meths].sort_values().index[0]
            )
        else:
            best_method_local = (
                combined_rank_local[ori_top3_local].sort_values().index[0]
            )

        local_para = best_method_local
        with open(logfn, "a") as f:
            f.write(f"The best local para combo: {best_method_local}.\n")

        # ------------------------------------------------------------------
        # 2c. Pick best method across ALL methods using combined metrics
        # ------------------------------------------------------------------
        pp3_ranks_all = pp3_frac.rank(axis=1, method="min").mean(axis=0)
        bp4_ranks_all = bp4_frac.rank(axis=1, method="min").mean(axis=0)
        combined_rank_all = (pp3_ranks_all + bp4_ranks_all) / 2

        ori_top3_all = combined_rank_all.sort_values().head(3).index.tolist()
        with open(logfn, "a") as f:
            f.write(f"The ori_top3_methods for ALL methods: {ori_top3_all}\n")

        p2_meth_all = pp3_p75.loc[:, pp3_p75.quantile(0.75, axis=0) < 2].columns.tolist()
        p4_meth_all = pp3_max.loc[:, pp3_max.quantile(0.75, axis=0) < 3].columns.tolist()

        b2_meth_all = bp4_p75.loc[:, bp4_p75.quantile(0.75, axis=0) < 2].columns.tolist()
        b4_meth_all = bp4_max.loc[:, bp4_max.quantile(0.75, axis=0) < 3].columns.tolist()

        print(f"ALL: p2_meth: {p2_meth_all}; p4_meth: {p4_meth_all}")
        print(f"ALL: b2_meth: {b2_meth_all}; b4_meth: {b4_meth_all}")

        qual_meths_all = intersect_lists(
            p2_meth_all, p4_meth_all, b2_meth_all, b4_meth_all
        )
        with open(logfn, "a") as f:
            f.write(f"The qual_meths for ALL methods: {qual_meths_all}\n")

        final_qual_meths_all = intersect_lists(qual_meths_all, ori_top3_all)
        with open(logfn, "a") as f:
            f.write(f"The final_qual_meths for ALL methods: {final_qual_meths_all}\n")

        if len(final_qual_meths_all) > 0:
            best_method_all = (
                combined_rank_all[final_qual_meths_all].sort_values().index[0]
            )
        else:
            best_method_all = (
                combined_rank_all[ori_top3_all].sort_values().index[0]
            )

        with open(logfn, "a") as f:
            f.write(
                f"\n~~~~~The final best calibration method for {dist}: "
                f"{best_method_all}.~~~~~\n"
            )

        return local_para, best_method_all

    # select best method
    local_para, best_model = get_best_calib()

    # decode local parameter combo: e.g. "local_win0.3_fn0.06"
    win_match = re.search(r"win([0-9.]+)", local_para)
    fn_match = re.search(r"fn([0-9.]+)", local_para)
    if win_match:
        winfrac = float(win_match.group(1))
    else:
        winfrac = 0.3  # default / fallback
    if fn_match:
        gnfrac = float(fn_match.group(1))
    else:
        gnfrac = 0.06  # default / fallback

    # labeled/unlabeled arrays
    X_lab_full = labdat[0].values.astype(float)
    y_lab_full = labdat[1].values.astype(int)
    X_unlab_full = unlabdat[0].values.astype(float)
    X_all = np.concatenate([X_lab_full, X_unlab_full])
    X_all_sorted = np.sort(X_all)

    n_lab = len(X_lab_full)
    n_unlab = len(X_unlab_full)

    # core method name (without _others / local params)
    best_model_core = re.sub(r"(_others|_local.*)$", "", best_model)

    # ------------------------
    # Run bootstrap OOB
    # ------------------------
    if best_model_core == "MonoPostNN":
        # SPECIAL: MonoPostNN OOB bootstrap
        lab_p95, unlab_p95 = _monopost_oob_bootstrap(
            X_lab_full, y_lab_full, X_unlab_full, alpha,
            B=200, num_ensemble=5, epochs=300, seed=123
        )
        with open(logfn, "a") as f:
            f.write(
                "MonoPostNN OOB bootstrap (B=200, num_ensemble=5) completed.\n"
            )
    else:
        # generic bootstrap using multiprocessing
        B = 1000
        seeds = list(range(B))
        lab_rows = []
        unlab_rows = []

        with Pool(
            processes=N_PROCS,
            initializer=_bootstrap_initializer,
            initargs=(
                X_lab_full,
                y_lab_full,
                X_unlab_full,
                best_model,
                alpha,
                pnrat,
            ),
        ) as pool:
            start = time.time()
            results = pool.map(_run_single_bootstrap, seeds)
            print(f"[Bootstrap {best_model}] done in {time.time() - start:.2f} sec")

        for res in results:
            if res is None:
                continue
            lab_full, unlab_post = res
            lab_rows.append(lab_full)
            unlab_rows.append(unlab_post)

        if len(lab_rows) == 0:
            with open(logfn, "a") as f:
                f.write(
                    "WARNING: No valid bootstrap replicates; OOB p95 not computed.\n"
                )
            lab_p95 = np.full(n_lab, np.nan)
            unlab_p95 = np.full(n_unlab, np.nan)
        else:
            lab_stack = np.vstack(lab_rows)     # (B_eff, n_lab)
            unlab_stack = np.vstack(unlab_rows) # (B_eff, n_unlab)

            lab_p95 = np.nanpercentile(lab_stack, 5, axis=0)
            unlab_p95 = np.nanpercentile(unlab_stack, 5, axis=0)

        # write CSV for generic methods
        df_labeled = pd.DataFrame(
            {
                "type": "labeled",
                "index": np.arange(n_lab),
                "score": X_lab_full,
                "label": y_lab_full,
                "oob_post_p95": lab_p95,
            }
        )

        df_unlab = pd.DataFrame(
            {
                "type": "unlabeled",
                "index": np.arange(n_unlab),
                "score": X_unlab_full,
                "label": np.nan,
                "oob_post_p95": unlab_p95,
            }
        )

        oob_all = pd.concat([df_labeled, df_unlab], ignore_index=True)
        out_csv = f"{BASE_PATH}/{gene}/{gene}_{dist}_oob_posterior_{best_model}.csv"
        oob_all.to_csv(out_csv, index=False)

        with open(logfn, "a") as f:
            f.write(
                f"OOB bootstrap ({B} reps) completed for {best_model}.\n"
                f"OOB 95th-percentile posterior output: {out_csv}\n"
            )

    # (for plotting benign side)
    lab_b95 = 1.0 - lab_p95

    # ------------------------
    # One-time “full” calibration curve
    # ------------------------
    def _fit_once_full(best_model_name, X_lab_full_, y_lab_full_, X_unlab_full_):
        X_all_ = np.concatenate([X_lab_full_, X_unlab_full_])
        X_grid = np.sort(X_all_)
        model_core = re.sub(r"(_others|_local.*)$", "", best_model_name)

        if model_core == "Local":
            win_size = int(winfrac * len(X_lab_full_))
            calib = LocalCalibration(
                alpha, False, win_size, gnfrac, False, True
            )
            thresh, posterior_p = calib.fit(
                X_lab_full_, y_lab_full_, X_unlab_full_, alpha
            )
            full_post = _get_prob_linear_interp(
                X_grid, thresh, posterior_p
            )
            return X_grid, full_post
        elif model_core == "MonoPostNN":
            post = getMonoCalibNN(
                X_lab_full_.reshape(-1, 1),
                y_lab_full_,
                X_grid,
                alpha=alpha,
                num_ensemble=15,
                epochs=300,
            )
            full_post = _to_numpy(post).flatten()  ##post.detach().cpu().numpy().flatten()
            return X_grid, full_post
        else:
            full_post = _fit_and_predict_once(
                best_model_name,
                X_train=X_lab_full_,
                y_train=y_lab_full_,
                X_test=X_grid,
                X_unlabel=X_unlab_full_,
                alpha=alpha,
                pnrat=pnrat,
            )
            return X_grid, full_post

    X_grid, final_post_p = _fit_once_full(
        best_model, X_lab_full, y_lab_full, X_unlab_full
    )
    final_post_b = 1.0 - np.flip(final_post_p)

    # ------------------------
    # Plot posterior curves + thresholds
    # ------------------------
    colors = pd.DataFrame(plt.cm.tab20(np.linspace(0, 1, 11)))
    colors.index = [
        "Beta",
        "SplineCalib",
        "SmoothIsotonic",
        "Local",
        "MonoPostNN",
        "Isotonic",
        "WeightedPlatt",
        "Platt",
        "BetaMixture",
        "MixSkewNorm",
        "TruncNorm",
    ]

    # choose color key
    color_key = best_model_core if best_model_core in colors.index else "MonoPostNN"

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1])

    # --- Benign side ---
    ax11 = fig.add_subplot(gs[1, 0])

    ax11.plot(
        np.flip(X_grid),
        final_post_b,
        linewidth=2.0,
        color=colors.loc[color_key],
        label=best_model_core,
    )

    for lvl, ls in zip(
        Post_b[::-1], ["dotted", "dashed", "dashdot", (5, (10, 3)), "solid"]
    ):
        ax11.axhline(lvl, linestyle=ls, color="steelblue")

    ax11.set_xlabel(f"{dist} Score")
    ax11.set_ylabel("Posterior")
    ax11.set_ylim([min(0.975, Post_b[4] - 0.002), 1.001])
    ax11.legend(loc="best", fontsize=8)

    if dist == "AM":
        old_threshb = [0.169, 0.099, 0.07, np.nan]
    elif dist == "MP2":
        old_threshb = [0.391, 0.197, 0.031, 0.01]
    elif dist == "REVEL":
        old_threshb = [0.29, 0.183, 0.052, 0.016]
    else:
        old_threshb = [np.nan, np.nan, np.nan, np.nan]

    for i in range(len(old_threshb)):
        ax11.scatter(
            old_threshb[i],
            Post_b[4 - i],
            marker="s",
            s=100,
            color="green",
        )

    # intersections benign
    y_b = np.array(final_post_b)
    x_b = np.array(np.flip(X_grid))
    bintersections = []
    for y_horiz in Post_b:
        if y_horiz < np.min(y_b):
            bintersections.append(np.nan)
            continue
        x_candidates = []
        for i in range(len(y_b) - 1):
            if (y_b[i] < y_horiz <= y_b[i + 1]) or (
                y_b[i] > y_horiz >= y_b[i + 1]
            ):
                x_int = x_b[i] + (x_b[i + 1] - x_b[i]) * (
                    y_horiz - y_b[i]
                ) / (y_b[i + 1] - y_b[i])
                x_candidates.append(x_int)
        if x_candidates:
            if best_model_core == "SmoothIsotonic":
                bintersections.append(max(x_candidates))
            else:
                bintersections.append(min(x_candidates))
        else:
            bintersections.append(np.nan)
        print(f"y_horiz: {y_horiz}; bintersections: {bintersections}")

    for i in range(1, len(bintersections)):
        if bintersections[i] < bintersections[i - 1]:
            bintersections[i - 1] = np.nan

    bintersections.reverse()
    bintersections = [x for x in bintersections if not np.isnan(x)]

    # --- Pathogenic side ---
    ax21 = fig.add_subplot(gs[1, 1])

    ax21.plot(
        X_grid,
        final_post_p,
        linewidth=2.0,
        color=colors.loc[color_key],
        label=best_model_core,
    )

    ax21.axhline(Post_p[4], linestyle="dotted", color="r")
    ax21.axhline(Post_p[3], linestyle="dashed", color="r")
    ax21.axhline(Post_p[2], linestyle="dashdot", color="r")
    ax21.axhline(Post_p[1], linestyle=(5, (10, 3)), color="r")
    ax21.axhline(Post_p[0], linestyle="solid", color="r")
    ax21.set_xlabel(f"{dist} Score")
    ax21.set_ylabel("Posterior")
    ax21.legend(loc="best", fontsize=8)

    if dist == "AM":
        old_threshp = [0.792, 0.906, 0.972, 0.99]
    elif dist == "MP2":
        old_threshp = [0.737, 0.829, 0.895, 0.932]
    elif dist == "REVEL":
        old_threshp = [0.644, 0.773, 0.879, 0.932]
    else:
        old_threshp = [np.nan, np.nan, np.nan, np.nan]

    for i in range(len(old_threshp)):
        ax21.scatter(
            old_threshp[i],
            Post_p[4 - i],
            marker="s",
            s=100,
            color="green",
        )

    y_p = np.array(final_post_p)
    x_p = np.array(X_grid)

    pintersections = []
    for y_horiz in Post_p:
        if y_horiz > np.max(y_p):
            pintersections.append(np.nan)
            continue
        x_candidates = []
        for i in range(len(y_p) - 1):
            if (y_p[i] < y_horiz <= y_p[i + 1]) or (
                y_p[i] > y_horiz >= y_p[i + 1]
            ):
                x_int = x_p[i] + (x_p[i + 1] - x_p[i]) * (
                    y_horiz - y_p[i]
                ) / (y_p[i + 1] - y_p[i])
                x_candidates.append(x_int)
        if x_candidates:
            if best_model_core == "SplineCalib":
                pintersections.append(min(x_candidates))
            else:
                pintersections.append(max(x_candidates))
        else:
            pintersections.append(np.nan)
        print(f"y_horiz: {y_horiz}; pintersections: {pintersections}")

    for i in range(0, len(pintersections) - 1):
        if pintersections[i] < pintersections[i + 1]:
            pintersections[i] = np.nan

    pintersections.reverse()
    pintersections = [x for x in pintersections if not np.isnan(x)]

    # --- Histogram + vertical lines ---
    ax2 = fig.add_subplot(gs[0, :])
    sns.histplot(
        labdat[labdat[1] == 0][0],
        bins=50,
        color="blue",
        alpha=0.6,
        ax=ax2,
        label=f"BLB ({len(labdat[labdat[1] == 0][0])})",
    )
    sns.histplot(
        labdat[labdat[1] == 1][0],
        bins=50,
        color="red",
        alpha=0.6,
        ax=ax2,
        label=f"PLP ({len(labdat[labdat[1] == 1][0])})",
    )
    ax2.legend(loc="upper left", fontsize=8)
    ax3 = ax2.twinx()
    sns.histplot(
        unlabdat[0],
        bins=50,
        color="grey",
        alpha=0.3,
        ax=ax3,
        label=f"gnomAD ({len(unlabdat[0])})",
    )
    ax3.legend(loc="upper right", fontsize=8)
    ax2.set_xlabel(f"{gene} {dist} Score (Prior: {round(alpha,4)})")
    plt.tight_layout()

    linestyles = ["dotted", "dashed", "dashdot", (5, (10, 3)), "solid"]
    linewidths = [1.5, 1.5, 1.5, 1.5, 2.2]
    strens = ["Supporting", "Moderate", "+3", "Strong", "VeryStrong"]

    with open(logfn, "a") as f:
        f.write(f"The {dist} thresholds in BP4: {bintersections}.\n")
        f.write(
            f"The {dist} highest strength in BP4: "
            f"{strens[len(bintersections)-1]}.\n"
        )
        f.write(f"The {dist} thresholds in PP3: {pintersections}.\n")
        f.write(
            f"The {dist} highest strength in PP3: "
            f"{strens[len(pintersections)-1]}.\n"
        )
        f.write("\n\n__________________________________\n\n")

    old_threshp.append(np.nan)
    for i in range(len(pintersections)):
        ax2.axvline(
            pintersections[i],
            color="r",
            linestyle=linestyles[i],
            linewidth=linewidths[i],
            label=f"{strens[i]}: {round(pintersections[i], 3)}",
        )

    for i in range(len(old_threshp)):
        ax2.axvline(
            old_threshp[i],
            color="lightcoral",
            alpha=0.5,
            linestyle=linestyles[i],
            linewidth=linewidths[i],
        )

    old_threshb.append(np.nan)
    for i in range(len(bintersections)):
        ax2.axvline(
            bintersections[i],
            color="b",
            linestyle=linestyles[i],
            linewidth=linewidths[i],
            label=f"{strens[i]}: {round(bintersections[i], 3)}",
        )

    for i in range(len(old_threshb)):
        ax2.axvline(
            old_threshb[i],
            color="lightblue",
            alpha=0.5,
            linestyle=linestyles[i],
            linewidth=linewidths[i],
        )

    ax2.legend(bbox_to_anchor=(1.09, 1), loc="upper left")
    plt.tight_layout()
    out_png = f"{BASE_PATH}/{gene}/p95_oob{dist}_{method}_{gene}_final_calib.png"
    plt.savefig(out_png, dpi=200)
    plt.close()


# -------------------------------------------------------------------
# main(): ARRAY_IDX wrapper
# -------------------------------------------------------------------

GENE_DIST_LIST = f"{BASE_PATH}/00.gene_dist.list"


def _parse_gene_dist_from_index(idx: int):
    """
    Read the idx-th line (1-based) from 00.gene_dist.list.
    Each line contains:  <GENE> <DIST>
    Returns (gene, dist)
    """
    with open(GENE_DIST_LIST) as f:
        lines = f.read().strip().splitlines()

    if idx < 1 or idx > len(lines):
        raise IndexError(f"ARRAY_IDX={idx} out of range (1–{len(lines)})")

    parts = lines[idx - 1].strip().split()
    gene = parts[0]
    dist = parts[1]
    return gene, dist


def _read_median_prior_for_gene(gene: str) -> float:
    """
    Extract the median 'The prior is:' value from the bootstrap prior file.
    """
    fn = (
        "/sc/arion/projects/pejaverlab/IGVF/users/cheny60/analysis/"
        "calibrationexp-main/calib_decision_tree/inheritance_analysis/"
        f"prior_gnomad/{gene}/uniboot_rus_pca_notfiltmp2train/"
        "cluster_gnomad.res_btstrap.txt"
    )

    if not os.path.exists(fn):
        raise FileNotFoundError(f"Missing prior file: {fn}")

    vals = []
    with open(fn) as f:
        for line in f:
            m = re.search(r"The prior is:\s*([0-9.]+)", line)
            if m:
                try:
                    vals.append(float(m.group(1)))
                except ValueError:
                    continue

    if len(vals) == 0:
        raise ValueError(f"No valid 'The prior is:' entries found for gene {gene}")

    vals.sort()
    n = len(vals)
    return vals[n // 2] if n % 2 == 1 else (vals[n // 2 - 1] + vals[n // 2]) / 2


def _parse_simu_info(gene: str, dist: str):
    """
    Reads pnratio, nsamp and method from new{gene}_{dist}_SimuInfo.txt.
    """
    fn = f"{BASE_PATH}/{gene}/new{gene}_{dist}_SimuInfo.txt"

    if not os.path.exists(fn):
        raise FileNotFoundError(f"Missing simulation info: {fn}")

    pnrat = None
    nsamp = None
    method = None

    with open(fn) as f:
        for line in f:
            line = line.strip()
            if line.startswith("pnr"):
                pnrat = float(line.split(":")[1])
            elif line.startswith("nsamp"):
                nsamp = int(line.split(":")[1])
            elif line.startswith("method"):
                method = line.split(":")[1]

    if pnrat is None or nsamp is None or method is None:
        raise ValueError(f"Malformed SimuInfo file: {fn}")

    return pnrat, nsamp, method


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m calib_pipeline.calib_step03 ARRAY_IDX", file=sys.stderr)
        sys.exit(1)
    try:
        array_idx = int(sys.argv[1])
    except ValueError:
        print(f"ARRAY_IDX must be an integer, got {sys.argv[1]!r}", file=sys.stderr)
        sys.exit(1)

    gene, dist = _parse_gene_dist_from_index(array_idx)
    alpha = _read_median_prior_for_gene(gene)
    pnrat, nsamp, method = _parse_simu_info(gene, dist)

    print(
        f"[calib_step03] ARRAY_IDX={array_idx} -> gene={gene}, dist={dist}, "
        f"alpha={alpha}, pnrat={pnrat}, nsamp={nsamp}, method={method}"
    )

    run_final_calibration_for_gene(
        gene=gene,
        dist=dist,
        alpha=alpha,
        pnrat=pnrat,
        method=method,
    )


if __name__ == "__main__":
    main()
