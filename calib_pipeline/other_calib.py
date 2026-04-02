import sys
import os
import pickle
import json
import jsonpickle
import time
from multiprocessing import Pool
import numpy as np

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

import pandas as pd
from scipy.stats import beta, truncnorm
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

import ml_insights as mli
import torch

# --- External dependencies (user-provided or auto-fallback to local clones) ---
EXTERNAL_BASE = os.environ.get("CALIB_EXTERNAL_BASE", None)

def _safe_append(path):
    if path and os.path.exists(path) and path not in sys.path:
        sys.path.append(path)

# Expected structure (user can override via env var):
#   $CALIB_EXTERNAL_BASE/
#       simulation_strategy/
#       mave_calibration/
#       PosteriorCalibration/

if EXTERNAL_BASE:
    _safe_append(os.path.join(EXTERNAL_BASE, "simulation_strategy"))
    _safe_append(os.path.join(EXTERNAL_BASE, "mave_calibration"))
    _safe_append(os.path.join(EXTERNAL_BASE, "PosteriorCalibration"))

# Fallback: allow imports if user installed via pip or manually cloned into PYTHONPATH
try:
    from parser import getParser  # from simulation_strategy
except ImportError:
    raise ImportError(
        "Cannot import 'parser'. Please clone or install simulation_strategy repo "
        "(https://github.com/shajain/GaussianMixDataGenerator) and set CALIB_EXTERNAL_BASE."
    )

try:
    from mave_calibration.main import single_fit
    from mave_calibration.skew_normal.density_utils import joint_densities
except ImportError:
    raise ImportError(
        "Cannot import mave_calibration. Please install from "
        "https://github.com/Dzeiberg/mave_calibration"
    )

try:
    from MonotonicPosterior.computePosterior_fast import computePosteriorFromEnsemble
except ImportError:
    raise ImportError(
        "Cannot import PosteriorCalibration. Please install from "
        "https://github.com/shajain/PosteriorCalibration"
    )


# =====================================================================
# Globals for multiprocessing
# =====================================================================

y_calibrate_global = None
y_calibrate_pred_prob_global = None
y_test_pred_prob_global = None
alpha_global = None

global outdir


# =====================================================================
# Utilities for multiprocessing / bootstrapping
# =====================================================================

def initialize_data(y_calibrate, y_calibrate_pred_prob, y_test_pred_prob, alpha):
    """Initializer for Pool: store arrays in globals."""
    global y_calibrate_global, y_calibrate_pred_prob_global, y_test_pred_prob_global, alpha_global
    y_calibrate_global = y_calibrate
    y_calibrate_pred_prob_global = y_calibrate_pred_prob
    y_test_pred_prob_global = y_test_pred_prob
    alpha_global = alpha


def make_sample_indicators(y):
    """Make 2-column boolean indicator matrix for labels 1 (P) and 0 (B)."""
    sample_names = [1, 0]
    indicators = np.zeros((len(y), len(sample_names)), dtype=bool)
    for i, sample_name in enumerate(sample_names):
        indicators[:, i] = y.flatten() == sample_name
    return indicators


def select_oob(y, y_pred, seed):
    """
    Bootstrap (balanced P/B) sample from calibration set given seed.
    Returns y_sample, y_pred_sample.
    """
    y = np.array(y)
    y_pred = np.array(y_pred)

    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    boot_pos_idx = rng.choice(pos_idx, size=len(pos_idx), replace=True)
    boot_neg_idx = rng.choice(neg_idx, size=len(neg_idx), replace=True)
    idx = np.concatenate([boot_pos_idx, boot_neg_idx])

    y_sample = y[idx]
    y_pred_sample = y_pred[idx]
    return y_sample, y_pred_sample


def transform(data_prior=0.1, est_prior=0.1, posterior=None):
    """
    Adjust posterior probabilities when data_prior != est_prior.
    posterior: 1D numpy array or list.
    """
    if posterior is None:
        return None

    posterior = np.array(posterior, dtype=float)

    # if all NaNs, keep as is
    if np.isnan(posterior).all():
        return posterior

    # if any NaN, we still try to transform elementwise, leaving NaNs
    mask_valid = ~np.isnan(posterior)
    p = posterior.copy()
    p[p == 1] = 1 - 1e-10

    gamma_x = np.empty_like(p)
    gamma_x[:] = np.nan

    num = (est_prior / (1 - est_prior)) * ((1 - data_prior) / data_prior)

    gamma_x[mask_valid] = num * (p[mask_valid] / (1 - p[mask_valid]))
    out = np.empty_like(p)
    out[:] = np.nan
    out[mask_valid] = gamma_x[mask_valid] / (1 + gamma_x[mask_valid])
    return out


# =====================================================================
# Core calibration methods (single fit on full data)
# =====================================================================

def getMixGaussCalib(observations, sample_indicators, yall_pred, priors):
    """Mixture of skew-normal Gaussians from mave_calibration."""
    bestFitres = single_fit(
        observations.flatten(), sample_indicators, max_iters=1000, n_inits=100, verbose=False
    )
    f_P = joint_densities(
        yall_pred, bestFitres["component_params"], bestFitres["weights"][0]
    ).sum(0)
    f_B = joint_densities(
        yall_pred, bestFitres["component_params"], bestFitres["weights"][1]
    ).sum(0)
    P = f_P / f_B
    posteriors = P * priors / ((P - 1) * priors + 1)
    return posteriors


def getMonoCalibNN(X, y, X_test, alpha, save=False):
    """
    Train MonoPost ensemble once and return robust mean predictions for X_test.
    """
    if save:
        np.save("xtrain.npy", X)
        np.save("ytrain.npy", y)
        np.save("xtest.npy", X_test)

    modelAvg_func, modelMedian_func, modelRobustMean_func, Ensemble = computePosteriorFromEnsemble(
        X, y, alpha=alpha, test_size=0.0, num_ensemble=10, epochs=300
    )
    X_test = np.array(X_test.reshape(len(X_test), 1))
    device = next(Ensemble[0].parameters()).device
    X_torch = torch.tensor(X_test, dtype=torch.float).to(device)
    postRobustAvg = modelRobustMean_func(X_torch).detach().cpu().numpy()
    return postRobustAvg.flatten()


def run_monopost_ensemble(X_train, y_train, X_test, alpha, save=False):
    """
    Run MonoPost calibration using the PosteriorCalibration ensemble.
    Returns:
      - monopost_point: robust-mean point estimate on X_test (1D array)
      - monopost_ensemble: all ensemble member predictions on X_test (2D array: n_models x n_samples)
    """
    if save:
        np.save("xtrain.npy", X_train)
        np.save("ytrain.npy", y_train)
        np.save("xtest.npy", X_test)

    # Train ensemble via PosteriorCalibration
    modelAvg_func, modelMedian_func, modelRobustMean_func, Ensemble = computePosteriorFromEnsemble(
        X_train,
        y_train,
        alpha=alpha,
        test_size=0.0,
        num_ensemble=10,
        epochs=300,
    )

    # Make sure X_test is shape (n_samples, 1)
    X_test = np.array(X_test).reshape(-1, 1)

    # Try to build a torch tensor if models are torch-based, otherwise fall back to numpy
    try:
        import torch

        device = next(Ensemble[0].parameters()).device
        X_torch = torch.tensor(X_test, dtype=torch.float32).to(device)

        # Ensemble member outputs
        postEsts = []
        for model in Ensemble:
            out = model(X_torch)
            out_np = _to_numpy(out)
            postEsts.append(out_np.flatten())

        # Robust mean posterior
        out_robust = modelRobustMean_func(X_torch)
        monopost_point = _to_numpy(out_robust).flatten()

    except Exception:
        # Fallback path if Ensemble / models are not torch-based
        postEsts = []
        for model in Ensemble:
            out = model(X_test)
            postEsts.append(_to_numpy(out).flatten())

        out_robust = modelRobustMean_func(X_test)
        monopost_point = _to_numpy(out_robust).flatten()

    monopost_ensemble = np.vstack(postEsts)
    return monopost_point, monopost_ensemble



def getPlattCalibratedProbs(y_calibrate, y_calibrate_pred_nn_prob, y_test_pred_nn_prob):
    logreg = LogisticRegression(
        C=99999999999, solver="lbfgs", class_weight={0: 0.5, 1: 0.5}
    )
    logreg.fit(
        y_calibrate_pred_nn_prob.reshape(-1, 1), y_calibrate.reshape(len(y_calibrate),)
    )
    platt_calibrated_prob_test = [
        e[1] for e in logreg.predict_proba(y_test_pred_nn_prob.reshape(-1, 1))
    ]
    return np.asarray(platt_calibrated_prob_test)


def getWeightedPlattCalibratedProbs(
    y_calibrate, y_calibrate_pred_nn_prob, y_test_pred_nn_prob, w0, w1
):
    logreg = LogisticRegression(
        C=99999999999, solver="lbfgs", class_weight={0: w0, 1: w1}
    )
    logreg.fit(
        y_calibrate_pred_nn_prob.reshape(-1, 1), y_calibrate.reshape(len(y_calibrate),)
    )
    platt_calibrated_prob_test = [
        e[1] for e in logreg.predict_proba(y_test_pred_nn_prob.reshape(-1, 1))
    ]
    return np.asarray(platt_calibrated_prob_test)


def getIsotonicCalibratedProbs(
    y_calibrate, y_calibrate_pred_nn_prob, y_test_pred_nn_prob
):
    iso_reg = IsotonicRegression(out_of_bounds="clip").fit(
        y_calibrate_pred_nn_prob, y_calibrate.flatten()
    )
    isotonic_calibrated_prob_test = iso_reg.predict(y_test_pred_nn_prob)
    return np.asarray(isotonic_calibrated_prob_test)


def getBetaCalibrationProbs(
    y_calibrate, y_calibrate_pred_nn_prob, y_test_pred_nn_prob
):
    from betacal import BetaCalibration

    beta_reg = BetaCalibration(parameters="abm").fit(
        y_calibrate_pred_nn_prob, y_calibrate.flatten()
    )
    beta_reg_prob_test = beta_reg.predict(y_test_pred_nn_prob)
    return np.asarray(beta_reg_prob_test)


def fit_beta_mixture(classes, predictions, test_preds):
    def clean_data(data):
        data = np.array(data)
        data = data[~np.isnan(data)]
        data = data[(data > 0) & (data < 1)]
        return data

    pos_sample = clean_data(predictions[classes == 1])
    neg_sample = clean_data(predictions[classes == 0])

    def fit_beta_distribution(sample_data):
        try:
            params = beta.fit(sample_data, floc=0, fscale=1)
            return params
        except Exception as e:
            print(f"Error fitting beta distribution: {e}")
            return np.nan, np.nan, np.nan

    plp_beta_params = fit_beta_distribution(pos_sample)
    blb_beta_params = fit_beta_distribution(neg_sample)

    if np.any(np.isnan(plp_beta_params)) or np.any(np.isnan(blb_beta_params)):
        print("Beta mixture fitting failed.")
        return np.full_like(test_preds, np.nan, dtype=float)

    pos_pdf = beta(plp_beta_params[0], plp_beta_params[1]).pdf
    neg_pdf = beta(blb_beta_params[0], blb_beta_params[1]).pdf

    pos_probs = pos_pdf(test_preds)
    neg_probs = neg_pdf(test_preds)
    pos_neg_sum = pos_probs + neg_probs
    posteriors = np.divide(
        pos_probs, pos_neg_sum, out=np.zeros_like(pos_probs), where=pos_neg_sum != 0
    )
    return posteriors


def fit_truncnorm_mixture(classes, predictions, test_preds):
    def clean_data(data):
        data = np.array(data)
        data = data[~np.isnan(data)]
        data = data[(data > 0) & (data < 1)]
        return data

    def fit_truncnorm(data):
        lower = 0.0
        upper = 1.0

        def negative_log_likelihood(params):
            mu, sigma = params
            if sigma <= 0:
                return np.inf
            dist = truncnorm(
                (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma
            )
            return -np.sum(dist.logpdf(data))

        init_params = [np.mean(data), np.std(data)]
        bounds = [(lower, upper), (1e-3, None)]
        result = minimize(
            negative_log_likelihood, init_params, bounds=bounds, method="L-BFGS-B"
        )
        return {"mu": result.x[0], "sigma": result.x[1]}

    lower = 0.0
    upper = 1.0

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


def getSmIsotonicCalibratedProbs(classes, predictions, test_preds):
    predictions = np.array(predictions)
    classes = np.array(classes)
    test_preds = np.array(test_preds)

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
    return smopred


def getSplineCalibProbs(y_calibrate, y_calibrate_pred_nn_prob, y_test_pred_nn_prob):
    splinecalib = mli.SplineCalib()
    splinecalib.fit(
        y_calibrate_pred_nn_prob.reshape(-1), y_calibrate.reshape(-1)
    )
    splinecalib_prob_test = splinecalib.predict(
        y_test_pred_nn_prob.reshape(-1)
    )
    return np.asarray(splinecalib_prob_test)


# =====================================================================
# Bootstrap wrappers for each method (used with Pool.map)
# =====================================================================

def run_mixgauss_calib(seed):
    y_sample, y_pred_sample = select_oob(
        y=y_calibrate_global, y_pred=y_calibrate_pred_prob_global, seed=seed
    )
    sample_indicators = make_sample_indicators(y_sample)
    return getMixGaussCalib(
        observations=y_pred_sample,
        sample_indicators=sample_indicators,
        yall_pred=y_test_pred_prob_global,
        priors=alpha_global,
    )


def run_platt_calibration(seed):
    y_sample, y_pred_sample = select_oob(
        y=y_calibrate_global, y_pred=y_calibrate_pred_prob_global, seed=seed
    )
    return getPlattCalibratedProbs(
        y_sample, y_pred_sample, y_test_pred_prob_global
    )


def run_weighted_platt_calibration(seed):
    y_sample, y_pred_sample = select_oob(
        y=y_calibrate_global, y_pred=y_calibrate_pred_prob_global, seed=seed
    )

    pnratio = np.mean(y_sample)
    pnratio = np.clip(pnratio, 1e-6, 1 - 1e-6)
    w0 = (1 - alpha_global) / (2 * (1 - pnratio))
    w1 = alpha_global / (2 * pnratio)

    return getWeightedPlattCalibratedProbs(
        y_sample, y_pred_sample, y_test_pred_prob_global, w0, w1
    )


def run_isotonic_calibration(seed):
    y_sample, y_pred_sample = select_oob(
        y=y_calibrate_global, y_pred=y_calibrate_pred_prob_global, seed=seed
    )
    return getIsotonicCalibratedProbs(
        y_sample, y_pred_sample, y_test_pred_prob_global
    )


def run_beta_calibration(seed):
    y_sample, y_pred_sample = select_oob(
        y=y_calibrate_global, y_pred=y_calibrate_pred_prob_global, seed=seed
    )
    return getBetaCalibrationProbs(
        y_sample, y_pred_sample, y_test_pred_prob_global
    )


def run_beta_mixture_calibration(seed):
    y_sample, y_pred_sample = select_oob(
        y=y_calibrate_global, y_pred=y_calibrate_pred_prob_global, seed=seed
    )
    posteriors = fit_beta_mixture(
        y_sample, y_pred_sample, y_test_pred_prob_global
    )
    return posteriors


def run_truncnorm_mixture_calibration(seed):
    y_sample, y_pred_sample = select_oob(
        y=y_calibrate_global, y_pred=y_calibrate_pred_prob_global, seed=seed
    )
    return fit_truncnorm_mixture(
        y_sample, y_pred_sample, y_test_pred_prob_global
    )


def run_smooth_isotonic_calibration(seed):
    y_sample, y_pred_sample = select_oob(
        y=y_calibrate_global, y_pred=y_calibrate_pred_prob_global, seed=seed
    )
    return getSmIsotonicCalibratedProbs(
        y_sample, y_pred_sample, y_test_pred_prob_global
    )


def run_spline_calibration(seed):
    y_sample, y_pred_sample = select_oob(
        y=y_calibrate_global, y_pred=y_calibrate_pred_prob_global, seed=seed
    )
    return getSplineCalibProbs(
        y_sample, y_pred_sample, y_test_pred_prob_global
    )

def run_monopost_calibration(seed):
    """
    Bootstrap replicate for MonoPostNN:
      - resample (balanced) from calibration set using 'seed'
      - train a MonoPost ensemble on the bootstrap sample
      - return predictions on the *fixed* test set (y_test_pred_prob_global)
    """
    # OOB/bootstrap sample from calibration data
    y_sample, y_pred_sample = select_oob(
        y=y_calibrate_global, y_pred=y_calibrate_pred_prob_global, seed=seed
    )

    # MonoPost expects X as (n,1)
    monopost_point, _ = run_monopost_ensemble(
        y_pred_sample.reshape(-1, 1),
        y_sample,
        y_test_pred_prob_global,
        alpha=alpha_global,
    )

    # We only need the robust mean posterior per test sample
    return monopost_point

# =====================================================================
# Main
# =====================================================================

def main():
    global outdir

    parser = getParser()
    args = parser.parse_args()

    predictor = args.predictor
    method = args.method
    seed = args.seed * 828
    outdir = args.outdir
    alpha = args.alpha
    n_calibrate = args.n_calibrate
    n_test = args.n_test
    gene = args.clustn
    pnratio_calibrate = args.pnratio_calibrate
    pnratio_test = args.pnratio_test
    np.random.seed(seed)

    win_frac = args.win_frac
    gnfrac = args.gnfrac

    if args.n_calibrate is not None:
        n_calibrate = args.n_calibrate
    if args.n_test is not None:
        n_test = args.n_test

    outdir = os.path.join(
        outdir, f"{gene}_{predictor}_{method}_Ntrain{n_calibrate}"
    )
    os.makedirs(outdir, exist_ok=True)
    np.random.seed(seed)

    # Main outputs file (point estimates for all methods)
    main_outfile = os.path.join(
        outdir, f"{predictor}_simu_{method}{seed/828}_calib_outputs_others.csv"
    )
    if os.path.exists(main_outfile):
        print(f"{main_outfile} exists, will not overwrite main point estimates.")
    else:
        # Load simulation data
        with open(
            f"{outdir}/{predictor}_simu_{method}{seed/828}.pkl", "rb"
        ) as f:
            simudat = pickle.load(f)

        y_calibrate_pred_prob = simudat["y_calibrate_pred_prob"]
        y_calibrate = simudat["y_calibrate"]
        y_test_pred_prob = simudat["y_test_pred_prob"]
        y_unlabelled_pred_prob = simudat["y_unlabelled_pred_prob"]
        true_posterior = simudat["true_posterior"]

        # sample_indicators for MixGauss
        sample_indicators = make_sample_indicators(y_calibrate)

        # weights for Weighted Platt on full data
        w0_full = (1 - alpha) / (2 * (1 - pnratio_calibrate))
        w1_full = alpha / (2 * pnratio_calibrate)

        # main point estimates for each method
        monopost_point, monopost_ensemble = run_monopost_ensemble(
            y_calibrate_pred_prob.reshape(-1, 1),
            y_calibrate,
            y_test_pred_prob,
            alpha=alpha,
            #num_ensemble=10,
            #epochs=300,
        )

        mix_gauss_main = getMixGaussCalib(
            y_calibrate_pred_prob.flatten(),
            sample_indicators,
            y_test_pred_prob.flatten(),
            alpha,
        )
        platt_main = getPlattCalibratedProbs(
            y_calibrate, y_calibrate_pred_prob, y_test_pred_prob
        )
        weighted_platt_main = getWeightedPlattCalibratedProbs(
            y_calibrate,
            y_calibrate_pred_prob,
            y_test_pred_prob,
            w0_full,
            w1_full,
        )
        isotonic_main = getIsotonicCalibratedProbs(
            y_calibrate, y_calibrate_pred_prob, y_test_pred_prob
        )
        smooth_iso_main = getSmIsotonicCalibratedProbs(
            y_calibrate, y_calibrate_pred_prob, y_test_pred_prob
        )
        beta_main = getBetaCalibrationProbs(
            y_calibrate, y_calibrate_pred_prob, y_test_pred_prob
        )
        beta_mix_main = fit_beta_mixture(
            y_calibrate, y_calibrate_pred_prob, y_test_pred_prob
        )
        spline_main = getSplineCalibProbs(
            y_calibrate, y_calibrate_pred_prob, y_test_pred_prob
        )
        truncnorm_main = fit_truncnorm_mixture(
            y_calibrate, y_calibrate_pred_prob, y_test_pred_prob
        )

        calib_outputs_main = pd.DataFrame(
            {
                "True": np.asarray(true_posterior).flatten(),
                "BetaMixture": np.asarray(beta_mix_main).flatten(),
                "TruncNorm": np.asarray(truncnorm_main).flatten(),
                "MixSkewNorm": np.asarray(mix_gauss_main).flatten(),
                "WeightedPlatt": np.asarray(weighted_platt_main).flatten(),
                "Platt": np.asarray(platt_main).flatten(),
                "Isotonic": np.asarray(isotonic_main).flatten(),
                "SmoothIsotonic": np.asarray(smooth_iso_main).flatten(),
                "Beta": np.asarray(beta_main).flatten(),
                "SplineCalib": np.asarray(spline_main).flatten(),
                "MonoPostNN": np.asarray(monopost_point).flatten(),
            }
        )

        # apply transform for methods estimated under data prior != alpha
        if pnratio_calibrate != alpha:
            trans_cols = [
                "BetaMixture",
                "TruncNorm",
                "Platt",
                "Isotonic",
                "SmoothIsotonic",
                "Beta",
                "SplineCalib",
            ]
            for col in trans_cols:
                calib_outputs_main[col] = transform(
                    data_prior=pnratio_calibrate,
                    est_prior=alpha,
                    posterior=calib_outputs_main[col].values,
                )

        calib_outputs_main.to_csv(main_outfile, index=False)
        print(f"Wrote main point estimates to {main_outfile}")

    # -----------------------------------------------------------------
    # Bootstrap for others + MonoPost ensemble quantiles
    # -----------------------------------------------------------------

    # Outputs for percentiles
    p95_outfile = os.path.join(
        outdir, f"{predictor}_simu_{method}{seed/828}_calib_outputs_P95_others.csv"
    )
    b95_outfile = os.path.join(
        outdir, f"{predictor}_simu_{method}{seed/828}_calib_outputs_B95_others.csv"
    )
    p50_outfile = os.path.join(
        outdir, f"{predictor}_simu_{method}{seed/828}_calib_outputs_P50_others.csv"
    )
    b50_outfile = os.path.join(
        outdir, f"{predictor}_simu_{method}{seed/828}_calib_outputs_B50_others.csv"
    )

    if (
        os.path.exists(p95_outfile)
        and os.path.exists(b95_outfile)
        and os.path.exists(p50_outfile)
        and os.path.exists(b50_outfile)
    ):
        print("All percentile output files exist, skipping bootstrap.")
        return

    # Load simudat again for bootstrap + MonoPost ensemble
    with open(
        f"{outdir}/{predictor}_simu_{method}{seed/828}.pkl", "rb"
    ) as f:
        simudat = pickle.load(f)

    y_calibrate_pred_prob = simudat["y_calibrate_pred_prob"]
    y_calibrate = simudat["y_calibrate"]
    y_test_pred_prob = simudat["y_test_pred_prob"]
    true_posterior = simudat["true_posterior"]

    # Re-run MonoPost ensemble (if you want to reuse, you could cache)
    monopost_point, monopost_ensemble = run_monopost_ensemble(
        y_calibrate_pred_prob.reshape(-1, 1),
        y_calibrate,
        y_test_pred_prob,
        alpha=alpha,
        #num_ensemble=10,
        #epochs=300,
    )  # monopost_ensemble: (num_ensemble, n_test)

    # No need to transform MonoPostNN posterior
    mono_ensemble_transformed = monopost_ensemble

    # -----------------------------------------------------------------
    # Bootstrap for other calibration methods
    # -----------------------------------------------------------------
    B = 100               # bootstrap reps
    bst_seeds = [int(828* (b)) for b in range(B)]

    with Pool(
        processes=10,
        initializer=initialize_data,
        initargs=(y_calibrate, y_calibrate_pred_prob, y_test_pred_prob, alpha),
    ) as pool:
        start = time.time()
        tmpbeta_outputs = pool.map(run_beta_calibration, bst_seeds)
        print(f"[Beta Calibration] done in {time.time() - start:.2f} sec")

        start = time.time()
        tmpbeta_mix_outputs = pool.map(run_beta_mixture_calibration, bst_seeds)
        print(f"[Beta Mixture] done in {time.time() - start:.2f} sec")

        start = time.time()
        tmpplatt_outputs = pool.map(run_platt_calibration, bst_seeds)
        print(f"[Platt] done in {time.time() - start:.2f} sec")

        start = time.time()
        tmpwplatt_outputs = pool.map(run_weighted_platt_calibration, bst_seeds)
        print(f"[Weighted Platt] done in {time.time() - start:.2f} sec")

        start = time.time()
        tmpiso_outputs = pool.map(run_isotonic_calibration, bst_seeds)
        print(f"[Isotonic] done in {time.time() - start:.2f} sec")

        start = time.time()
        tmpsmoothed_iso_outputs = pool.map(
            run_smooth_isotonic_calibration, bst_seeds
        )
        print(f"[Smoothed Isotonic] done in {time.time() - start:.2f} sec")

        start = time.time()
        tmpspline_calib_outputs = pool.map(run_spline_calibration, bst_seeds)
        print(f"[Spline] done in {time.time() - start:.2f} sec")

        start = time.time()
        tmpmixgauss_outputs = pool.map(run_mixgauss_calib, bst_seeds)
        print(f"[MixGauss] done in {time.time() - start:.2f} sec")

        start = time.time()
        tmptrnormix_outputs = pool.map(
            run_truncnorm_mixture_calibration, bst_seeds
        )
        print(f"[TruncNorm Mixture] done in {time.time() - start:.2f} sec")

        # --- MonoPostNN OOB bootstrap ---
        start = time.time()
        tmpmonopost_outputs = pool.map(
            run_monopost_calibration, bst_seeds
        )
        print(f"[MonoPostNN bootstrap] done in {time.time() - start:.2f} sec")


    # Convert lists of arrays to 2D arrays (B, n_test)
    tmpbeta_outputs = np.asarray(tmpbeta_outputs)
    tmpbeta_mix_outputs = np.asarray(tmpbeta_mix_outputs)
    tmpplatt_outputs = np.asarray(tmpplatt_outputs)
    tmpwplatt_outputs = np.asarray(tmpwplatt_outputs)
    tmpiso_outputs = np.asarray(tmpiso_outputs)
    tmpsmoothed_iso_outputs = np.asarray(tmpsmoothed_iso_outputs)
    tmpspline_calib_outputs = np.asarray(tmpspline_calib_outputs)
    tmpmixgauss_outputs = np.asarray(tmpmixgauss_outputs)
    tmptrnormix_outputs = np.asarray(tmptrnormix_outputs)
    tmpmonopost_outputs = np.asarray(tmpmonopost_outputs)

    # Apply transform (data_prior -> est_prior) to appropriate methods
    if pnratio_calibrate != alpha:
        beta_outputs = np.vstack(
            [
                transform(
                    data_prior=pnratio_calibrate,
                    est_prior=alpha,
                    posterior=row,
                )
                for row in tmpbeta_outputs
            ]
        )
        beta_mix_outputs = np.vstack(
            [
                transform(
                    data_prior=pnratio_calibrate,
                    est_prior=alpha,
                    posterior=row,
                )
                for row in tmpbeta_mix_outputs
            ]
        )
        platt_outputs = np.vstack(
            [
                transform(
                    data_prior=pnratio_calibrate,
                    est_prior=alpha,
                    posterior=row,
                )
                for row in tmpplatt_outputs
            ]
        )
        iso_outputs = np.vstack(
            [
                transform(
                    data_prior=pnratio_calibrate,
                    est_prior=alpha,
                    posterior=row,
                )
                for row in tmpiso_outputs
            ]
        )
        smoothed_iso_outputs = np.vstack(
            [
                transform(
                    data_prior=pnratio_calibrate,
                    est_prior=alpha,
                    posterior=row,
                )
                for row in tmpsmoothed_iso_outputs
            ]
        )
        spline_calib_outputs = np.vstack(
            [
                transform(
                    data_prior=pnratio_calibrate,
                    est_prior=alpha,
                    posterior=row,
                )
                for row in tmpspline_calib_outputs
            ]
        )
        trnormix_outputs = np.vstack(
            [
                transform(
                    data_prior=pnratio_calibrate,
                    est_prior=alpha,
                    posterior=row,
                )
                for row in tmptrnormix_outputs
            ]
        )
    else:
        beta_outputs = tmpbeta_outputs
        beta_mix_outputs = tmpbeta_mix_outputs
        platt_outputs = tmpplatt_outputs
        iso_outputs = tmpiso_outputs
        smoothed_iso_outputs = tmpsmoothed_iso_outputs
        spline_calib_outputs = tmpspline_calib_outputs
        trnormix_outputs = tmptrnormix_outputs

    # WeightedPlatt and MixGauss are left untransformed, as in your original code
    wplatt_outputs = tmpwplatt_outputs
    mixgauss_outputs = tmpmixgauss_outputs

    # Benign (B) side arrays (1 - flipped)
    all_benign_beta_outputs = 1 - np.flip(beta_outputs, axis=1)
    all_benign_platt_outputs = 1 - np.flip(platt_outputs, axis=1)
    all_benign_wplatt_outputs = 1 - np.flip(wplatt_outputs, axis=1)
    all_benign_iso_outputs = 1 - np.flip(iso_outputs, axis=1)
    all_benign_smoothed_iso_outputs = 1 - np.flip(
        smoothed_iso_outputs, axis=1
    )
    all_benign_spline_calib_outputs = 1 - np.flip(
        spline_calib_outputs, axis=1
    )
    all_benign_mixgauss_outputs = 1 - np.flip(mixgauss_outputs, axis=1)
    all_benign_trnormix_outputs = 1 - np.flip(trnormix_outputs, axis=1)
    all_benign_beta_mix_outputs = 1 - np.flip(beta_mix_outputs, axis=1)

    # MonoPostNN bootstrap (P and B side)
    mono_p_boot = tmpmonopost_outputs                      # shape (B_mono, n_test)
    mono_b_boot = 1 - np.flip(mono_p_boot, axis=1)

    # Percentiles across bootstrap / ensemble dimension
    def pctl(arr, q):
        return np.percentile(arr, q, axis=0)

    # P side
    calib_p95 = pd.DataFrame(
        {
            "True": np.asarray(true_posterior).flatten(),
            "BetaMixture": pctl(beta_mix_outputs, 5),
            "TruncNorm": pctl(trnormix_outputs, 5),
            "MixSkewNorm": pctl(mixgauss_outputs, 5),
            "WeightedPlatt": pctl(wplatt_outputs, 5),
            "Platt": pctl(platt_outputs, 5),
            "Isotonic": pctl(iso_outputs, 5),
            "SmoothIsotonic": pctl(smoothed_iso_outputs, 5),
            "Beta": pctl(beta_outputs, 5),
            "SplineCalib": pctl(spline_calib_outputs, 5),
            "MonoPostNN": pctl(mono_p_boot, 5),
        }
    )
    calib_p50 = pd.DataFrame(
        {
            "True": np.asarray(true_posterior).flatten(),
            "BetaMixture": pctl(beta_mix_outputs, 50),
            "TruncNorm": pctl(trnormix_outputs, 50),
            "MixSkewNorm": pctl(mixgauss_outputs, 50),
            "WeightedPlatt": pctl(wplatt_outputs, 50),
            "Platt": pctl(platt_outputs, 50),
            "Isotonic": pctl(iso_outputs, 50),
            "SmoothIsotonic": pctl(smoothed_iso_outputs, 50),
            "Beta": pctl(beta_outputs, 50),
            "SplineCalib": pctl(spline_calib_outputs, 50),
            "MonoPostNN": pctl(mono_p_boot, 50),
        }
    )

    # B side
    calib_b95 = pd.DataFrame(
        {
            "True": 1 - np.flip(np.asarray(true_posterior).flatten()),
            "BetaMixture": pctl(all_benign_beta_mix_outputs, 5),
            "TruncNorm": pctl(all_benign_trnormix_outputs, 5),
            "MixSkewNorm": pctl(all_benign_mixgauss_outputs, 5),
            "WeightedPlatt": pctl(all_benign_wplatt_outputs, 5),
            "Platt": pctl(all_benign_platt_outputs, 5),
            "Isotonic": pctl(all_benign_iso_outputs, 5),
            "SmoothIsotonic": pctl(all_benign_smoothed_iso_outputs, 5),
            "Beta": pctl(all_benign_beta_outputs, 5),
            "SplineCalib": pctl(all_benign_spline_calib_outputs, 5),
            "MonoPostNN": pctl(mono_b_boot, 5),
        }
    )
    calib_b50 = pd.DataFrame(
        {
            "True": 1 - np.flip(np.asarray(true_posterior).flatten()),
            "BetaMixture": pctl(all_benign_beta_mix_outputs, 50),
            "TruncNorm": pctl(all_benign_trnormix_outputs, 50),
            "MixSkewNorm": pctl(all_benign_mixgauss_outputs, 50),
            "WeightedPlatt": pctl(all_benign_wplatt_outputs, 50),
            "Platt": pctl(all_benign_platt_outputs, 50),
            "Isotonic": pctl(all_benign_iso_outputs, 50),
            "SmoothIsotonic": pctl(all_benign_smoothed_iso_outputs, 50),
            "Beta": pctl(all_benign_beta_outputs, 50),
            "SplineCalib": pctl(all_benign_spline_calib_outputs, 50),
            "MonoPostNN": pctl(mono_b_boot, 50),
        }
    )

    calib_p95.to_csv(p95_outfile, index=False)
    calib_b95.to_csv(b95_outfile, index=False)
    calib_p50.to_csv(p50_outfile, index=False)
    calib_b50.to_csv(b50_outfile, index=False)

    print(f"Wrote P95 percentiles to {p95_outfile}")
    print(f"Wrote B95 percentiles to {b95_outfile}")
    print(f"Wrote P50 percentiles to {p50_outfile}")
    print(f"Wrote B50 percentiles to {b50_outfile}")


if __name__ == "__main__":
    main()

