#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os, sys, pickle, argparse
from scipy.stats import beta, truncnorm
from scipy.integrate import quad
from scipy.optimize import minimize
from skewt_scipy.skewt import skewt
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------
parser = argparse.ArgumentParser(description="Fit best distribution to labeled dataset")
parser.add_argument("--input", required=True, help="Path to labeled data CSV/TSV")
parser.add_argument("--feature_col", required=True, help="Column name for the feature to fit")
parser.add_argument("--label_col", required=True, help="Column name for the label (0/1 or BLB/PLP)")
parser.add_argument("--dist_name", default="Feature", help="Name of feature for labeling")
parser.add_argument("--outdir", default="./fit_results", help="Output directory")
parser.add_argument("--plot", action="store_true", help="Generate PDF plots of fits")
args = parser.parse_args()

# -----------------------------------------------------------
# Setup paths and data
# -----------------------------------------------------------
os.makedirs(args.outdir, exist_ok=True)
dat = pd.read_csv(args.input, sep=None, engine="python")  # auto-detects comma or tab

# Standardize labels
labdat = dat[[args.feature_col, args.label_col]].dropna()
if labdat[args.label_col].dtype == object:
    labdat["label"] = labdat[args.label_col].str.contains("PLP", case=False, na=False).astype(int)
else:
    labdat["label"] = labdat[args.label_col].astype(int)
labdat = labdat.rename(columns={args.feature_col: "score"})[["score", "label"]]
print(f"Loaded {labdat.shape[0]} rows. Positives: {labdat.label.sum()}, Negatives: {(labdat.label==0).sum()}")

# -----------------------------------------------------------
# Distribution fitting functions
# -----------------------------------------------------------
def clean_data(data):
    data = np.array(data)
    data = data[~np.isnan(data)]
    return data[(data > 0) & (data < 1)]

def fit_truncnorm(data):
    eps = 1e-6
    mean, std = np.mean(data), np.std(data)
    a_norm = (0 - mean) / std
    b_norm = (1 - mean) / std
    initial_guess = [a_norm, b_norm, mean, std]

    def neg_log_likelihood(params):
        a_norm, b_norm, loc, scale = params
        scale = max(scale, eps)
        dist = truncnorm(a_norm, b_norm, loc=loc, scale=scale)
        log_pdf = dist.logpdf(data)
        if np.any(np.isnan(log_pdf)) or np.any(np.isinf(log_pdf)):
            return np.inf
        return -np.sum(log_pdf)

    constraints = [
        {"type": "ineq", "fun": lambda p: p[2]},  # loc >= 0
        {"type": "ineq", "fun": lambda p: 1 - p[2]},  # loc <= 1
        {"type": "ineq", "fun": lambda p: p[3]},  # scale >= 0
    ]
    result = minimize(neg_log_likelihood, initial_guess, constraints=constraints)
    a_norm, b_norm, loc, scale = result.x
    a_norm = (0 - loc) / scale
    b_norm = (1 - loc) / scale
    return (a_norm, b_norm, loc, scale)

def compute_log_likelihood(distribution, params, data):
    if distribution.name == "truncnorm":
        a, b, loc, scale = params
        dist = distribution(a=a, b=b, loc=loc, scale=scale)
        pdf_vals = dist.pdf(data)
    else:
        dist = distribution(*params)
        pdf_vals = dist.pdf(data)
    pdf_vals = np.clip(pdf_vals, 1e-12, None)
    return np.sum(np.log(pdf_vals))

# -----------------------------------------------------------
# Fit candidate distributions
# -----------------------------------------------------------
def fit_distributions(df):
    blbrev = clean_data(df[df.label == 0]["score"])
    plprev = clean_data(df[df.label == 1]["score"])

    print(f"Fitting on N_pos={len(plprev)}, N_neg={len(blbrev)}")

    # Beta
    plp_beta_params = beta.fit(plprev, floc=0, fscale=1)
    blb_beta_params = beta.fit(blbrev, floc=0, fscale=1)
    plp_beta_ll = compute_log_likelihood(beta, plp_beta_params, plprev)
    blb_beta_ll = compute_log_likelihood(beta, blb_beta_params, blbrev)

    # TruncNorm
    plp_truncnorm_params = fit_truncnorm(plprev)
    blb_truncnorm_params = fit_truncnorm(blbrev)
    plp_truncnorm_ll = compute_log_likelihood(truncnorm, plp_truncnorm_params, plprev)
    blb_truncnorm_ll = compute_log_likelihood(truncnorm, blb_truncnorm_params, blbrev)

    return {
        "PLP": {"Beta": {"params": plp_beta_params, "ll": plp_beta_ll},
                "TruncNorm": {"params": plp_truncnorm_params, "ll": plp_truncnorm_ll}},
        "BLB": {"Beta": {"params": blb_beta_params, "ll": blb_beta_ll},
                "TruncNorm": {"params": blb_truncnorm_params, "ll": blb_truncnorm_ll}},
    }

fit_params = fit_distributions(labdat)

# -----------------------------------------------------------
# Select best model
# -----------------------------------------------------------
best_model = max(
    ["Beta", "TruncNorm"],
    key=lambda d: fit_params["PLP"][d]["ll"] + fit_params["BLB"][d]["ll"]
)
print(f" Best-fit distribution for {args.dist_name}: {best_model}")

# Save parameters
with open(f"{args.outdir}/{args.dist_name}_fit_params.pkl", "wb") as f:
    pickle.dump(fit_params, f)

# -----------------------------------------------------------
# Optional plotting
# -----------------------------------------------------------
if args.plot:
    x = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(figsize=(8, 5))
    plprev = labdat[labdat.label == 1]["score"]
    blbrev = labdat[labdat.label == 0]["score"]

    ax.hist(plprev, bins=30, density=True, alpha=0.4, color="blue", label="PLP Data")
    ax.hist(blbrev, bins=30, density=True, alpha=0.4, color="green", label="BLB Data")

    for cls, color in [("PLP", "blue"), ("BLB", "green")]:
        for d in ["Beta", "TruncNorm"]:
            params = fit_params[cls][d]["params"]
            if d == "Beta":
                y = beta.pdf(x, *params[:2])
            else:
                y = truncnorm.pdf(x, *params)
            ax.plot(x, y, label=f"{cls} {d} Fit (LL={fit_params[cls][d]['ll']:.1f})", lw=2, color=color, linestyle="--" if d=="TruncNorm" else "-")

    ax.set_title(f"Distribution fits for {args.dist_name}")
    ax.set_xlabel(f"{args.dist_name} Score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/{args.dist_name}_fit_plot.png", dpi=150)
    plt.close()
