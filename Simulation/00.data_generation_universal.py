#!/usr/bin/env python3
import os, sys, pickle, argparse
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

# Import your data generator classes
from GaussianMixDataGenerator.data.datagen import BetaDG
from GaussianMixDataGenerator.data.datagen import MVNormalMixDG as GMM
from GaussianMixDataGenerator.data.skewt_datagen import TruncSkewTDG
from GaussianMixDataGenerator.data.skewt_beta_datagen import Beta_TruncSkewTDG

# -----------------------------------------------------------
# Argument parser
# -----------------------------------------------------------
parser = argparse.ArgumentParser(description="Universal data generator based on fitted distribution parameters")
parser.add_argument("--param_file", required=True, help="Path to fitted parameter pickle file")
parser.add_argument("--method", required=True, help="Distribution model to use (Beta, TruncNorm, TruncSkewt, BetaSkewt, etc.)")
parser.add_argument("--dist_name", default="Feature", help="Name of the feature/distribution")
parser.add_argument("--outdir", default="./simu_results", help="Output directory for generated data")
parser.add_argument("--n_calibrate", type=int, default=1000, help="Number of samples for calibration set")
parser.add_argument("--n_test", type=int, default=1000, help="Number of samples for test set")
parser.add_argument("--pnratio_calibrate", type=float, default=0.5, help="Pos/Neg ratio in calibration set")
parser.add_argument("--pnratio_test", type=float, default=0.5, help="Pos/Neg ratio in test set")
parser.add_argument("--alpha", type=float, default=0.5, help="Mixing proportion (prior)")
parser.add_argument("--seed", type=int, default=828, help="Random seed")
parser.add_argument("--plot", action="store_true", help="Plot generated distributions")
args = parser.parse_args()

# -----------------------------------------------------------
# Utility
# -----------------------------------------------------------
np.random.seed(args.seed)
os.makedirs(args.outdir, exist_ok=True)

with open(args.param_file, "rb") as f:
    fit_para = pickle.load(f)

print(f"  Loaded parameter file: {args.param_file}")
print(f"  Available distributions: {list(fit_para.keys())}")

datlo, dathi = 0, 1  # assumed truncated domain

# -----------------------------------------------------------
# Build data generator dynamically
# -----------------------------------------------------------
def build_generator(method):
    if method == "Beta":
        alpha_pos = [fit_para["PLP"]["Beta"]["params"][0]]
        beta_pos = [fit_para["PLP"]["Beta"]["params"][1]]
        alpha_neg = [fit_para["BLB"]["Beta"]["params"][0]]
        beta_neg = [fit_para["BLB"]["Beta"]["params"][1]]
        return BetaDG(alpha_pos, beta_pos, [1.0], alpha_neg, beta_neg, [1.0], None)

    elif method == "TruncNorm":
        pos_params = fit_para["PLP"]["TruncNorm"]["params"]
        neg_params = fit_para["BLB"]["TruncNorm"]["params"]
        return GMM(pos_params, [1.0], neg_params, [1.0], None)

    elif method == "TruncSkewt":
        a_pos, df_pos, loc_pos, scale_pos, _ = fit_para["PLP"]["TruncSkewt"]["params"]
        a_neg, df_neg, loc_neg, scale_neg, _ = fit_para["BLB"]["TruncSkewt"]["params"]
        return TruncSkewTDG([a_pos], [df_pos], [loc_pos], [scale_pos], [datlo], [dathi], [1.0],
                            [a_neg], [df_neg], [loc_neg], [scale_neg], [datlo], [dathi], [1.0], alpha=None)

    elif method == "BetaSkewt":
        alpha_pos = [fit_para["PLP"]["Beta"]["params"][0]]
        beta_pos = [fit_para["PLP"]["Beta"]["params"][1]]
        a_neg, df_neg, loc_neg, scale_neg, _ = fit_para["BLB"]["TruncSkewt"]["params"]
        return Beta_TruncSkewTDG(alpha_pos, beta_pos, [1.0], [a_neg], [df_neg], [loc_neg], [scale_neg],
                                 [datlo], [dathi], [1.0], alpha=None)
    else:
        raise ValueError(f"Unsupported method: {method}")

gmm = build_generator(args.method)
print(f"  Data generator initialized for method: {args.method}")

# -----------------------------------------------------------
# Generate data
# -----------------------------------------------------------
def sort_xy(x, y):
    idx = np.argsort(x.flatten())
    return x.flatten()[idx], y.flatten()[idx]

X_cal, y_cal = gmm.pn_data(args.n_calibrate, args.pnratio_calibrate)[0:2]
X_test, y_test = gmm.pn_data(args.n_test, args.pnratio_test)[0:2]
X_unl, y_unl = gmm.pn_data(10000, args.alpha)[0:2]
true_posterior = gmm.pn_posterior(X_test, args.alpha)

X_cal, y_cal = sort_xy(X_cal, y_cal)
X_test, y_test = sort_xy(X_test, y_test)

# -----------------------------------------------------------
# Save simulation outputs
# -----------------------------------------------------------
simu = {
    "y_calibrate_pred_prob": X_cal,
    "y_calibrate": y_cal,
    "y_test_pred_prob": X_test,
    "y_test": y_test,
    "y_unlabelled_pred_prob": X_unl,
    "true_posterior": true_posterior,
    "alpha": args.alpha,
    "n_calibrate": args.n_calibrate,
    "pnratio_calibrate": args.pnratio_calibrate,
}

fname = os.path.join(args.outdir, f"{args.dist_name}_simu_{args.method}_seed{args.seed}.pkl")
with open(fname, "wb") as f:
    pickle.dump(simu, f)

print(f"  Simulation saved to: {fname}")

# -----------------------------------------------------------
# Optional plotting
# -----------------------------------------------------------
if args.plot:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,5))
    plt.hist(X_cal, bins=30, alpha=0.5, density=True, label="Calibration")
    plt.hist(X_test, bins=30, alpha=0.5, density=True, label="Test")
    plt.title(f"Simulated distributions ({args.method})")
    plt.xlabel("Feature Value")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{args.dist_name}_simu_{args.method}_seed{args.seed}.png"), dpi=150)
    plt.close()
