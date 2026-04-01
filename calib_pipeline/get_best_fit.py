import os
import sys
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import beta, truncnorm
from scipy.integrate import quad
from skewt_scipy.skewt import skewt


@dataclass
class GetBestFitConfig:
    """Configuration for best-fit distribution step."""

    outdir: str
    skip_if_recent_hours: float = 0.0

    def fit_params_pickle(self, gene: str, predictor: str) -> str:
        return os.path.join(self.outdir, gene, f"{gene}_{predictor}_fit_params.pkl")

    def simu_info_file(self, gene: str, predictor: str) -> str:
        return os.path.join(self.outdir, gene, f"{gene}_{predictor}_SimuInfo.txt")

    def fig_file(self, gene: str, predictor: str) -> str:
        return os.path.join(self.outdir, gene, f"{gene}_{predictor}_fit_dist.png")

def _is_file_recent(filepath: str, hours: float) -> bool:
    if hours <= 0:
        return False
    if not os.path.exists(filepath):
        return False
    import time

    current_time = time.time()
    file_mtime = os.path.getmtime(filepath)
    return (current_time - file_mtime) <= hours * 3600.0


def _fit_distributions(labfn: str) -> dict:
    dat = pd.read_table(labfn, header=None)
    blbrev = dat[dat[1] == 0][0]
    plprev = dat[dat[1] == 1][0]
    print(f"The input N_p = {len(plprev)}; N_b = {len(blbrev)}.")

    def clean_data(data):
        data = np.array(data)
        data = data[~np.isnan(data)]
        data = data[(data > 0) & (data < 1)]
        return data

    def truncate_skewt_data(data, trunclim):
        return [e for e in data if trunclim[0] < e < trunclim[1]]

    def get_out_data_skewt(a, df, loc, scale, rnge, indatalen):
        indata = []
        outdata = []
        while True:
            d = skewt.rvs(a=a, df=df, loc=loc, scale=scale, size=1)[0]
            if rnge[0] < d < rnge[1]:
                indata.append(d)
            else:
                outdata.append(d)
            if len(indata) > indatalen:
                break
        return outdata

    def compute_log_likelihood_skewt(distribution, params, data):
        a, df, loc, scale, rnge = params
        dist = distribution(a=a, df=df, loc=loc, scale=scale)

        cdf_min = dist.cdf(rnge[0])
        cdf_max = dist.cdf(rnge[1])
        norm_const = cdf_max - cdf_min

        data_arr = np.asarray(data)
        data_arr = data_arr[(data_arr > rnge[0]) & (data_arr < rnge[1])]

        pdf_vals = dist.pdf(data_arr) / norm_const
        pdf_vals = np.clip(pdf_vals, 1e-12, None)
        return np.sum(np.log(pdf_vals))

    def compute_log_likelihood_skewcauchy(distribution, params, data):
        a, loc, scale, rnge = params
        dist = distribution(a=a, df=1, loc=loc, scale=scale)

        cdf_min = dist.cdf(rnge[0])
        cdf_max = dist.cdf(rnge[1])
        norm_const = cdf_max - cdf_min

        data_arr = np.asarray(data)
        data_arr = data_arr[(data_arr > rnge[0]) & (data_arr < rnge[1])]

        pdf_vals = dist.pdf(data_arr) / norm_const
        pdf_vals = np.clip(pdf_vals, 1e-12, None)
        return np.sum(np.log(pdf_vals))

    def fit_truncskewt(pbdat, clnsig="B"):
        skewttruncdata = np.array(pbdat).flatten()
        rnge = [0.0, 1.0]
        oritruncdata = truncate_skewt_data(skewttruncdata, rnge)
        data = oritruncdata

        if clnsig == "B":
            a, df, loc, scale = 3.0, 3.0, 0.1, 1.0
        else:
            a, df, loc, scale = -0.5, 3.0, 0.85, 1.0

        outdata = get_out_data_skewt(a, df, loc, scale, rnge, len(oritruncdata))
        data = outdata + oritruncdata

        skewt_para = (a, df, loc, scale, rnge)
        skewt_ll = compute_log_likelihood_skewt(skewt, skewt_para, oritruncdata)

        patience_limit = 25
        patience_counter = 0
        threshold = 0.5
        min_iterations = 100

        print("initial paras for trunc-skewt:", a, df, loc, scale, "ll:", skewt_ll)

        global_best_ll = skewt_ll
        global_best_params = (a, df, loc, scale)

        for i in range(1000):
            data_arr = np.array(data)
            an, dfn, locn, scalen = skewt.fit(data_arr[np.isfinite(data_arr)])
            skewt_para = (an, dfn, locn, scalen, rnge)
            skewt_lln = compute_log_likelihood_skewt(skewt, skewt_para, oritruncdata)
            print(clnsig, "__iteration", i, ":", an, dfn, locn, scalen, "; ll:", skewt_lln)

            if skewt_lln > global_best_ll:
                global_best_ll = skewt_lln
                global_best_params = (an, dfn, locn, scalen)

            if skewt_lln > skewt_ll + threshold:
                a, df, loc, scale = an, dfn, locn, scalen
                skewt_ll = skewt_lln
                patience_counter = 0
            else:
                patience_counter += 1

            if i >= min_iterations and patience_counter >= patience_limit:
                print(f"No significant LL improvement for {patience_limit} iterations. Stopping early.")
                break

            outdata = get_out_data_skewt(an, dfn, locn, scalen, rnge, len(oritruncdata))
            data = outdata + oritruncdata

        a, df, loc, scale = global_best_params
        skewt_ll = global_best_ll
        print("final paras for trunc-skewt:", a, df, loc, scale, "ll:", skewt_ll)
        return (a, df, loc, scale, rnge)

    def get_out_data_skewcauchy(a, loc, scale, rnge, indatalen):
        indata = []
        outdata = []
        while True:
            d = skewt.rvs(a=a, df=1, loc=loc, scale=scale, size=1)[0]
            if rnge[0] < d < rnge[1]:
                indata.append(d)
            else:
                outdata.append(d)
            if len(indata) > indatalen:
                break
        return outdata

    def fit_truncskewcauchy(pbdat, clnsig="B"):
        skewcauchytruncdata = np.array(pbdat).flatten()
        rnge = [0.0, 1.0]
        oritruncdata = truncate_skewt_data(skewcauchytruncdata, rnge)
        data = oritruncdata

        if clnsig == "B":
            a, loc, scale = -0.8, 0.1, 1.0
        else:
            a, loc, scale = 0.8, 0.85, 1.0

        outdata = get_out_data_skewcauchy(a, loc, scale, rnge, len(oritruncdata))
        data = outdata + oritruncdata

        skewc_para = (a, loc, scale, rnge)
        skewc_ll = compute_log_likelihood_skewcauchy(skewt, skewc_para, oritruncdata)

        patience_limit = 25
        patience_counter = 0
        threshold = 0.5
        min_iterations = 100

        global_best_ll = skewc_ll
        global_best_params = (a, loc, scale)

        print("initial paras for skew-cauchy:", a, loc, scale, "ll:", skewc_ll)

        for i in range(1000):
            data_arr = np.array(data)
            an, dfn, locn, scalen = skewt.fit(data_arr[np.isfinite(data_arr)], fdf=1)
            skewc_para = (an, locn, scalen, rnge)
            skewc_lln = compute_log_likelihood_skewcauchy(skewt, skewc_para, oritruncdata)
            print(clnsig, "__iteration", i, ":", an, dfn, locn, scalen, "; ll:", skewc_lln)

            if skewc_lln > global_best_ll:
                global_best_ll = skewc_lln
                global_best_params = (an, locn, scalen)

            if skewc_lln > skewc_ll + threshold:
                a, loc, scale = an, locn, scalen
                skewc_ll = skewc_lln
                patience_counter = 0
            else:
                patience_counter += 1

            if i >= min_iterations and patience_counter >= patience_limit:
                print(f"No significant LL improvement for {patience_limit} iterations. Stopping early.")
                break

            outdata = get_out_data_skewcauchy(an, locn, scalen, rnge, len(oritruncdata))
            data = outdata + oritruncdata

        a, loc, scale = global_best_params
        skewc_ll = global_best_ll
        print("final paras for skew-cauchy:", a, loc, scale, "ll:", skewc_ll)
        return (a, loc, scale, rnge)

    def fit_truncnorm(data_arr):
        from scipy.optimize import minimize

        eps = 1e-6
        data_arr = np.asarray(data_arr)
        mean, std = np.mean(data_arr), np.std(data_arr)
        if std <= 0:
            std = eps

        a_norm = (0.0 - mean) / std
        b_norm = (1.0 - mean) / std
        initial_guess = [a_norm, b_norm, mean, std]

        def neg_log_likelihood(params):
            a_norm, b_norm, loc, scale = params
            scale = max(scale, eps)
            dist = truncnorm(a_norm, b_norm, loc=loc, scale=scale)
            log_pdf = dist.logpdf(data_arr)
            if np.any(np.isnan(log_pdf)) or np.any(np.isinf(log_pdf)):
                return np.inf
            return -np.sum(log_pdf)

        constraints = (
            {"type": "ineq", "fun": lambda params: params[2]},
            {"type": "ineq", "fun": lambda params: 1.0 - params[2]},
            {"type": "ineq", "fun": lambda params: params[3]},
        )
        result = minimize(neg_log_likelihood, initial_guess, constraints=constraints)
        a_norm, b_norm, loc, scale = result.x
        a_norm = (0.0 - loc) / scale
        b_norm = (1.0 - loc) / scale
        return (a_norm, b_norm, loc, scale)

    def compute_log_likelihood(distribution, params, data_arr):
        data_arr = np.asarray(data_arr)
        if distribution.name == "truncnorm":
            a_norm, b_norm, loc, scale = params
            dist = distribution(a=a_norm, b=b_norm, loc=loc, scale=scale)
            pdf_vals = dist.pdf(data_arr)
        else:
            dist = distribution(*params)
            pdf_vals = dist.pdf(data_arr)
        pdf_vals = np.clip(pdf_vals, 1e-12, None)
        return np.sum(np.log(pdf_vals))

    plprev_arr = clean_data(plprev)
    blbrev_arr = clean_data(blbrev)

    plp_beta_params = beta.fit(plprev_arr, floc=0, fscale=1)
    blb_beta_params = beta.fit(blbrev_arr, floc=0, fscale=1)

    plp_truncnorm_params = fit_truncnorm(plprev_arr)
    blb_truncnorm_params = fit_truncnorm(blbrev_arr)

    b_skewt_para = fit_truncskewt(pbdat=blbrev_arr, clnsig="B")
    print(b_skewt_para)
    p_skewt_para = fit_truncskewt(pbdat=plprev_arr, clnsig="P")
    print(p_skewt_para)

    b_skewc_para = fit_truncskewcauchy(pbdat=blbrev_arr, clnsig="B")
    print(b_skewc_para)
    p_skewc_para = fit_truncskewcauchy(pbdat=plprev_arr, clnsig="P")
    print(p_skewc_para)

    plp_beta_ll = compute_log_likelihood(beta, plp_beta_params, plprev_arr)
    blb_beta_ll = compute_log_likelihood(beta, blb_beta_params, blbrev_arr)

    plp_truncnorm_ll = compute_log_likelihood(truncnorm, plp_truncnorm_params, plprev_arr)
    blb_truncnorm_ll = compute_log_likelihood(truncnorm, blb_truncnorm_params, blbrev_arr)

    plp_skewt_ll = compute_log_likelihood_skewt(skewt, p_skewt_para, plprev_arr)
    blb_skewt_ll = compute_log_likelihood_skewt(skewt, b_skewt_para, blbrev_arr)

    plp_skewc_ll = compute_log_likelihood_skewcauchy(skewt, p_skewc_para, plprev_arr)
    blb_skewc_ll = compute_log_likelihood_skewcauchy(skewt, b_skewc_para, blbrev_arr)

    return {
        "PLP": {
            "Beta": {"params": plp_beta_params, "log_likelihood": plp_beta_ll},
            "TruncNorm": {"params": plp_truncnorm_params, "log_likelihood": plp_truncnorm_ll},
            "TruncSkewt": {"params": p_skewt_para, "log_likelihood": plp_skewt_ll},
            "TruncSkewCauchy": {"params": p_skewc_para, "log_likelihood": plp_skewc_ll},
        },
        "BLB": {
            "Beta": {"params": blb_beta_params, "log_likelihood": blb_beta_ll},
            "TruncNorm": {"params": blb_truncnorm_params, "log_likelihood": blb_truncnorm_ll},
            "TruncSkewt": {"params": b_skewt_para, "log_likelihood": blb_skewt_ll},
            "TruncSkewCauchy": {"params": b_skewc_para, "log_likelihood": blb_skewc_ll},
        },
    }


def _select_best_method(fit_params: dict) -> str:
    max_ll_p = -np.inf
    max_ll_b = -np.inf
    max_ll = -np.inf
    max_d = None
    max_d_p = None
    max_d_b = None

    dists = ["Beta", "TruncNorm", "TruncSkewt", "TruncSkewCauchy"]

    for d in dists:
        llp = fit_params["PLP"][d]["log_likelihood"]
        llb = fit_params["BLB"][d]["log_likelihood"]
        sum_ll = llp + llb

        if sum_ll > max_ll:
            max_d = d
            max_ll = sum_ll
        if llp > max_ll_p:
            max_ll_p = llp
            max_d_p = d
        if llb > max_ll_b:
            max_ll_b = llb
            max_d_b = d

    if max_d_p == max_d_b:
        final_max_d = max_d_p
    elif max_d_p == "Beta" and max_d_b == "TruncSkewt":
        final_max_d = "BetaSkewt"
    elif max_d_p == "TruncSkewCauchy" and max_d_b == "TruncSkewt":
        final_max_d = "CauchySkewt"
    elif max_d_p == "TruncSkewt" and max_d_b == "TruncSkewCauchy":
        final_max_d = "SkewtCauchy"
    elif max_d_p == "Beta" and max_d_b == "TruncSkewCauchy":
        final_max_d = "BetaCauchy"
    else:
        final_max_d = max_d

    return final_max_d


def _write_outputs(
    gene: str,
    predictor: str,
    labeled_file: str,
    cfg: GetBestFitConfig,
    fit_params: dict,
) -> str:
    labdat = pd.read_table(labeled_file, header=None)

    n_calibrate = len(labdat)
    pnratio_calibrate = (labdat[1] == 1).sum() / len(labdat)

    method = _select_best_method(fit_params)

    print("__________________________________________________________________")
    print(f"The best fit distribution for {predictor} scores is {method}.")
    print("__________________________________________________________________")

    os.makedirs(os.path.join(cfg.outdir, gene), exist_ok=True)

    with open(cfg.fit_params_pickle(gene, predictor), "wb") as f:
        pickle.dump(fit_params, f)

    with open(cfg.simu_info_file(gene, predictor), "w") as f:
        f.write(f"pnr:{pnratio_calibrate}\n")
        f.write(f"nsamp:{n_calibrate}\n")
        f.write(f"method:{method}\n")

    return method


def _make_figure(
    gene: str,
    predictor: str,
    labeled_file: str,
    cfg: GetBestFitConfig,
    fit_params: dict,
) -> None:
    dat = pd.read_table(labeled_file, header=None)
    blbrev = dat[dat[1] == 0][0]
    plprev = dat[dat[1] == 1][0]

    x = np.linspace(0, 1, 500)

    plp_beta_params = fit_params["PLP"]["Beta"]["params"]
    blb_beta_params = fit_params["BLB"]["Beta"]["params"]
    plp_beta_pdf = beta.pdf(x, *plp_beta_params[:2])
    blb_beta_pdf = beta.pdf(x, *blb_beta_params[:2])

    plp_truncnorm_params = fit_params["PLP"]["TruncNorm"]["params"]
    blb_truncnorm_params = fit_params["BLB"]["TruncNorm"]["params"]
    plp_truncnorm_pdf = truncnorm.pdf(x, *plp_truncnorm_params)
    blb_truncnorm_pdf = truncnorm.pdf(x, *blb_truncnorm_params)

    p_skewt_para = fit_params["PLP"]["TruncSkewt"]["params"]
    b_skewt_para = fit_params["BLB"]["TruncSkewt"]["params"]
    composp_skewt = skewt(
        a=p_skewt_para[0], df=p_skewt_para[1], loc=p_skewt_para[2], scale=p_skewt_para[3]
    )
    areap_skewt, _ = quad(composp_skewt.pdf, 0, 1)
    plp_pdf_skewt = composp_skewt.pdf(x)

    composb_skewt = skewt(
        a=b_skewt_para[0], df=b_skewt_para[1], loc=b_skewt_para[2], scale=b_skewt_para[3]
    )
    areab_skewt, _ = quad(composb_skewt.pdf, 0, 1)
    blb_pdf_skewt = composb_skewt.pdf(x)

    p_skewc_para = fit_params["PLP"]["TruncSkewCauchy"]["params"]
    b_skewc_para = fit_params["BLB"]["TruncSkewCauchy"]["params"]

    composp_skewc = skewt(
        a=p_skewc_para[0], df=1, loc=p_skewc_para[1], scale=p_skewc_para[2]
    )
    areap_skewc, _ = quad(composp_skewc.pdf, 0, 1)
    plp_pdf_skewc = composp_skewc.pdf(x)

    composb_skewc = skewt(
        a=b_skewc_para[0], df=1, loc=b_skewc_para[1], scale=b_skewc_para[2]
    )
    areab_skewc, _ = quad(composb_skewc.pdf, 0, 1)
    blb_pdf_skewc = composb_skewc.pdf(x)

    plt.figure(figsize=(10, 6))

    plt.hist(plprev, bins=30, density=True, alpha=0.4, color="blue", label="PLP Data")
    plt.plot(
        x,
        plp_beta_pdf,
        "b-",
        lw=2,
        label=(
            "PLP Beta Fit\n"
            f"α={plp_beta_params[0]:.2f}, β={plp_beta_params[1]:.2f}, "
            f"LL={fit_params['PLP']['Beta']['log_likelihood']:.2f}"
        ),
    )
    plt.plot(
        x,
        plp_truncnorm_pdf,
        "b--",
        lw=2,
        label=(
            "PLP TruncNorm Fit\n"
            f"μ={plp_truncnorm_params[2]:.2f}, σ={plp_truncnorm_params[3]:.2f}, "
            f"LL={fit_params['PLP']['TruncNorm']['log_likelihood']:.2f}"
        ),
    )
    plt.plot(
        x,
        plp_pdf_skewt / areap_skewt,
        "b-.",
        lw=2,
        label=(
            "PLP Skew-t Fit\n"
            f"α={p_skewt_para[0]:.2f}, df={p_skewt_para[1]:.2f}, "
            f"loc={p_skewt_para[2]:.1f}, scale={p_skewt_para[3]:.1f}, "
            f"LL={fit_params['PLP']['TruncSkewt']['log_likelihood']:.2f}"
        ),
    )
    plt.plot(
        x,
        plp_pdf_skewc / areap_skewc,
        "b:",
        lw=2,
        label=(
            "PLP Skewed Cauchy Fit\n"
            f"α={p_skewc_para[0]:.2f}, loc={p_skewc_para[1]:.1f}, scale={p_skewc_para[2]:.1f}, "
            f"LL={fit_params['PLP']['TruncSkewCauchy']['log_likelihood']:.2f}"
        ),
    )

    plt.hist(blbrev, bins=30, density=True, alpha=0.4, color="green", label="BLB Data")
    plt.plot(
        x,
        blb_beta_pdf,
        "g-",
        lw=2,
        label=(
            "BLB Beta Fit\n"
            f"α={blb_beta_params[0]:.2f}, β={blb_beta_params[1]:.2f}, "
            f"LL={fit_params['BLB']['Beta']['log_likelihood']:.2f}"
        ),
    )
    plt.plot(
        x,
        blb_truncnorm_pdf,
        "g--",
        lw=2,
        label=(
            "BLB TruncNorm Fit\n"
            f"μ={blb_truncnorm_params[2]:.2f}, σ={blb_truncnorm_params[3]:.2f}, "
            f"LL={fit_params['BLB']['TruncNorm']['log_likelihood']:.2f}"
        ),
    )
    plt.plot(
        x,
        blb_pdf_skewt / areab_skewt,
        "g-.",
        lw=2,
        label=(
            "BLB Skew-t Fit\n"
            f"α={b_skewt_para[0]:.2f}, df={b_skewt_para[1]:.2f}, "
            f"loc={b_skewt_para[2]:.1f}, scale={b_skewt_para[3]:.1f}, "
            f"LL={fit_params['BLB']['TruncSkewt']['log_likelihood']:.2f}"
        ),
    )
    plt.plot(
        x,
        blb_pdf_skewc / areab_skewc,
        "g:",
        lw=2,
        label=(
            "BLB Skewed Cauchy Fit\n"
            f"α={b_skewc_para[0]:.2f}, loc={b_skewc_para[1]:.1f}, scale={b_skewc_para[2]:.1f}, "
            f"LL={fit_params['BLB']['TruncSkewCauchy']['log_likelihood']:.2f}"
        ),
    )

    plt.title(f"{gene} P/B Distributions with Fitted Models")
    plt.xlabel(f"{predictor} Score")
    plt.ylabel("Density")
    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()
    plt.savefig(cfg.fig_file(gene, predictor), dpi=150)
    plt.close()


def fit_best_distribution(
    gene: str,
    predictor: str,
    labeled_file: str,
    cfg: GetBestFitConfig,
    make_plot: bool = False,
) -> dict:
    if not os.path.exists(labeled_file):
        raise FileNotFoundError(f"Labeled file not found: {labeled_file}")

    base_out = os.path.join(cfg.outdir, gene)
    os.makedirs(base_out, exist_ok=True)

    figfn = cfg.fig_file(gene, predictor)
    if make_plot and _is_file_recent(figfn, cfg.skip_if_recent_hours):
        print(f"{figfn} is recent; skipping recomputation.")
        return {}

    # --- Core fitting ---
    fit_params = _fit_distributions(labeled_file)

    # --- Outputs ---
    method = _write_outputs(
        gene=gene,
        predictor=predictor,
        labeled_file=labeled_file,
        cfg=cfg,
        fit_params=fit_params,
    )

    print(f"Selected method for {gene}, {predictor}: {method}")

    if make_plot:
        _make_figure(
            gene=gene,
            predictor=predictor,
            labeled_file=labeled_file,
            cfg=cfg,
            fit_params=fit_params,
        )

    return fit_params

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", required=True)
    parser.add_argument("--predictor", required=True)
    parser.add_argument("--labeled", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()

    cfg = GetBestFitConfig(outdir=args.outdir)

    fit_best_distribution(
        gene=args.gene,
        predictor=args.predictor,
        labeled_file=args.labeled,
        cfg=cfg,
        make_plot=args.plot,
    )


if __name__ == "__main__":
    main()

