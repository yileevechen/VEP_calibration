import os
import sys
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from GaussianMixDataGenerator.data.datagen import BetaDG
    from GaussianMixDataGenerator.data.datagen import MVNormalMixDG as GMM
    from GaussianMixDataGenerator.data.skewt_datagen import TruncSkewTDG
    from GaussianMixDataGenerator.data.skewt_beta_datagen import (
        Beta_TruncSkewTDG,
        TruncSkewT_TruncCauchyDG,
        TruncCauchy_TruncSkewTDG,
        TruncCauchyDG,
    )
except ImportError as e:
    raise ImportError(
        "Missing dependency: GaussianMixDataGenerator.\n\n"
        "Install it using:\n"
        "  pip install git+https://github.com/shajain/GaussianMixDataGenerator.git\n\n"
        "Or clone manually:\n"
        "  git clone https://github.com/shajain/GaussianMixDataGenerator.git\n"
        "  pip install -e GaussianMixDataGenerator\n"
    ) from e


@dataclass
class DataGenConfig:
    outdir: str 

def _fit_params_path(cfg: DataGenConfig, gene: str, predictor: str) -> str:
    return os.path.join(cfg.outdir, gene, f"{gene}_{predictor}_fit_params.pkl")

def _load_fit_params(cfg: DataGenConfig, gene: str, predictor: str) -> dict:
    parafn = _fit_params_path(cfg, gene, predictor)


def buildSkewTBetaMixDataGenerator(cfg: DataGenConfig, gene: str, predictor: str, lo: float, hi: float):
    fit_para = _load_fit_params(cfg, gene, predictor)
    alpha_pos = [fit_para["PLP"]["Beta"]["params"][0]]
    beta_pos = [fit_para["PLP"]["Beta"]["params"][1]]

    a_neg, df_neg, loc_neg, scale_neg, rng_neg = fit_para["BLB"]["TruncSkewt"]["params"]  # noqa: F841

    p_pos = [1.0]
    p_neg = [1.0]
    alpha = None
    return Beta_TruncSkewTDG(
        alpha_pos, beta_pos, p_pos,
        [a_neg], [df_neg], [loc_neg], [scale_neg],
        [lo], [hi], p_neg, alpha
    )


def buildBetaCauchyMixDataGenerator(cfg: DataGenConfig, gene: str, predictor: str, lo: float, hi: float):
    fit_para = _load_fit_params(cfg, gene, predictor)
    alpha_pos = [fit_para["PLP"]["Beta"]["params"][0]]
    beta_pos = [fit_para["PLP"]["Beta"]["params"][1]]

    a_neg, loc_neg, scale_neg, rng_neg = fit_para["BLB"]["TruncSkewCauchy"]["params"]  # noqa: F841
    df_neg = 1

    p_pos = [1.0]
    p_neg = [1.0]
    alpha = None
    return Beta_TruncSkewTDG(
        alpha_pos, beta_pos, p_pos,
        [a_neg], [df_neg], [loc_neg], [scale_neg],
        [lo], [hi], p_neg, alpha
    )


def buildSkewCauchyMixDataGenerator(cfg: DataGenConfig, gene: str, predictor: str, lo: float, hi: float):
    fit_para = _load_fit_params(cfg, gene, predictor)
    a_pos, loc_pos, scale_pos, rng_pos = fit_para["PLP"]["TruncSkewCauchy"]["params"]  # noqa: F841
    a_neg, loc_neg, scale_neg, rng_neg = fit_para["BLB"]["TruncSkewCauchy"]["params"]  # noqa: F841

    df_pos = 1
    df_neg = 1
    p_pos = [1.0]
    p_neg = [1.0]
    alpha = None

    return TruncSkewTDG(
        [a_pos], [df_pos], [loc_pos], [scale_pos], [lo], [hi], p_pos,
        [a_neg], [df_neg], [loc_neg], [scale_neg], [lo], [hi], p_neg, alpha=None
    )


def buildSkewCauchy_SkewTMixDataGenerator(cfg: DataGenConfig, gene: str, predictor: str, lo: float, hi: float):
    fit_para = _load_fit_params(cfg, gene, predictor)
    a_pos, loc_pos, scale_pos, rng_pos = fit_para["PLP"]["TruncSkewCauchy"]["params"]  # noqa: F841
    a_neg, df_neg, loc_neg, scale_neg, rng_neg = fit_para["BLB"]["TruncSkewt"]["params"]  # noqa: F841
    df_pos = 1

    p_pos = [1.0]
    p_neg = [1.0]
    alpha = None

    return TruncSkewTDG(
        [a_pos], [df_pos], [loc_pos], [scale_pos], [lo], [hi], p_pos,
        [a_neg], [df_neg], [loc_neg], [scale_neg], [lo], [hi], p_neg, alpha=None
    )


def buildSkewT_SkewCauchyMixDataGenerator(cfg: DataGenConfig, gene: str, predictor: str, lo: float, hi: float):
    fit_para = _load_fit_params(cfg, gene, predictor)
    a_pos, df_pos, loc_pos, scale_pos, rng_pos = fit_para["PLP"]["TruncSkewt"]["params"]  # noqa: F841
    a_neg, loc_neg, scale_neg, rng_neg = fit_para["BLB"]["TruncSkewCauchy"]["params"]  # noqa: F841
    df_neg = 1

    p_pos = [1.0]
    p_neg = [1.0]
    alpha = None

    return TruncSkewTDG(
        [a_pos], [df_pos], [loc_pos], [scale_pos], [lo], [hi], p_pos,
        [a_neg], [df_neg], [loc_neg], [scale_neg], [lo], [hi], p_neg, alpha=None
    )


def buildBetaMixDataGenerator(cfg: DataGenConfig, gene: str, predictor: str):
    fit_para = _load_fit_params(cfg, gene, predictor)
    alpha_pos = [fit_para["PLP"]["Beta"]["params"][0]]
    beta_pos = [fit_para["PLP"]["Beta"]["params"][1]]
    alpha_neg = [fit_para["BLB"]["Beta"]["params"][0]]
    beta_neg = [fit_para["BLB"]["Beta"]["params"][1]]
    p_pos = [1.0]
    p_neg = [1.0]
    alpha = None
    return BetaDG(alpha_pos, beta_pos, p_pos, alpha_neg, beta_neg, p_neg, alpha)


def buildGaussianMixDataGenerator(cfg: DataGenConfig, gene: str, predictor: str, lo: float, hi: float):
    fit_para = _load_fit_params(cfg, gene, predictor)
    pos_params = fit_para["PLP"]["TruncNorm"]["params"]
    neg_params = fit_para["BLB"]["TruncNorm"]["params"]
    p_pos = [1.0]
    p_neg = [1.0]
    alpha = None
    print("__________Current GaussianMixData Para:__________")
    print(f"pos params: {pos_params}; neg params: {neg_params}")
    return GMM(pos_params, p_pos, neg_params, p_neg, alpha)


def buildSkewtMixDataGenerator(cfg: DataGenConfig, gene: str, predictor: str, lo: float, hi: float):
    fit_para = _load_fit_params(cfg, gene, predictor)
    a_pos, df_pos, loc_pos, scale_pos, rng_pos = fit_para["PLP"]["TruncSkewt"]["params"]  # noqa: F841
    a_neg, df_neg, loc_neg, scale_neg, rng_neg = fit_para["BLB"]["TruncSkewt"]["params"]  # noqa: F841
    p_pos = [1.0]
    p_neg = [1.0]
    alpha = None

    return TruncSkewTDG(
        [a_pos], [df_pos], [loc_pos], [scale_pos], [lo], [hi], p_pos,
        [a_neg], [df_neg], [loc_neg], [scale_neg], [lo], [hi], p_neg, alpha=None
    )


def generate_simulation_data(
    gene: str,
    predictor: str,
    method: str,
    seed_index: int,
    outdir: str,
    labfn: str,
    alpha: float,
    n_calibrate: int,
    n_test: int,
    pnratio_calibrate: float,
    pnratio_test: float,
    cfg: DataGenConfig | None = None,
) -> str:
    if cfg is None:
        raise ValueError("cfg must be provided with outdir")
    
    seed = seed_index * 828
    np.random.seed(seed)

    labdat = pd.read_table(labfn, header=None)  # noqa: F841
    datlo = 0.0
    dathi = 1.0

    sim_outdir = os.path.join(outdir, f"{gene}_{predictor}_{method}_Ntrain{n_calibrate}")
    os.makedirs(sim_outdir, exist_ok=True)

    if method == "BetaSkewt":
        gmm = buildSkewTBetaMixDataGenerator(cfg, gene=gene, predictor=predictor, lo=datlo, hi=dathi)
    elif method == "TruncSkewCauchy":
        gmm = buildSkewCauchyMixDataGenerator(cfg, gene=gene, predictor=predictor, lo=datlo, hi=dathi)
    elif method == "CauchySkewt":
        gmm = buildSkewCauchy_SkewTMixDataGenerator(cfg, gene=gene, predictor=predictor, lo=datlo, hi=dathi)
    elif method == "SkewtCauchy":
        gmm = buildSkewT_SkewCauchyMixDataGenerator(cfg, gene=gene, predictor=predictor, lo=datlo, hi=dathi)
    elif method == "Beta":
        gmm = buildBetaMixDataGenerator(cfg, gene=gene, predictor=predictor)
    elif method == "TruncNorm":
        gmm = buildGaussianMixDataGenerator(cfg, gene=gene, predictor=predictor, lo=datlo, hi=dathi)
    elif method == "TruncSkewt":
        gmm = buildSkewtMixDataGenerator(cfg, gene=gene, predictor=predictor, lo=datlo, hi=dathi)
    elif method == "BetaCauchy":
        gmm = buildBetaCauchyMixDataGenerator(cfg, gene=gene, predictor=predictor, lo=datlo, hi=dathi)
    else:
        raise ValueError(f"Unknown method: {method}")

    if gmm is None:
        os.rmdir(sim_outdir)
        raise RuntimeError("Generator construction failed; output directory removed.")

    X_calibrate, y_calibrate = gmm.pn_data(n_calibrate, pnratio_calibrate)[0:2]
    sorted_idx = np.argsort(X_calibrate.flatten())
    X_calibrate = X_calibrate.flatten()[sorted_idx]
    y_calibrate = y_calibrate.flatten()[sorted_idx]

    X_test, y_test = gmm.pn_data(n_test, alpha)[0:2]
    sorted_idx = np.argsort(X_test.flatten())
    X_test = X_test.flatten()[sorted_idx]
    y_test = y_test.flatten()[sorted_idx]

    xu, yu = gmm.pn_data(10000, alpha)[0:2]
    true_posterior = gmm.pn_posterior(X_test, alpha)

    y_calibrate_pred_prob = X_calibrate
    y_test_pred_prob = X_test
    y_unlabelled_pred_prob = xu

    fname = os.path.join(sim_outdir, f"{gene}_{predictor}_simu_{method}_seed{seed_index}.pkl")

    simudat = {
        "y_calibrate_pred_prob": y_calibrate_pred_prob,
        "y_calibrate": y_calibrate,
        "y_test_pred_prob": y_test_pred_prob,
        "y_test": y_test,
        "y_unlabelled_pred_prob": y_unlabelled_pred_prob,
        "true_posterior": true_posterior,
        "alpha": alpha,
        "n_calibrate": n_calibrate,
        "pnratio_calibrate": pnratio_calibrate,
    }

    with open(fname, "wb") as f:
        pickle.dump(simudat, f)

    return fname


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate simulation data for one gene/score/method.")
    parser.add_argument("--predictor", type=str, required=True)
    parser.add_argument("--gene", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--n_calibrate", type=int, required=True)
    parser.add_argument("--n_test", type=int, required=True)
    parser.add_argument("--pnratio_calibrate", type=float, required=True)
    parser.add_argument("--pnratio_test", type=float, required=True)

    args = parser.parse_args()
    cfg = DataGenConfig(outdir=args.outdir)

    generate_simulation_data(
        gene=args.gene,
        predictor=args.predictor,
        method=args.method,
        seed_index=args.seed,
        outdir=args.outdir,
        alpha=args.alpha,
        n_calibrate=args.n_calibrate,
        n_test=args.n_test,
        pnratio_calibrate=args.pnratio_calibrate,
        pnratio_test=args.pnratio_test,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
