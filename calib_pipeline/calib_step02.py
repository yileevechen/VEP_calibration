import sys
import os
import re
import math
import pickle
import numpy as np
import pandas as pd
from Tavtigian.tavtigianutils import (
    get_tavtigian_c,
    get_tavtigian_thresholds,
    get_tavtigian_plr,
)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _parse_simu_info(infofile: str):
    """Parse pnr, nsamp, method from new{gene}_{predictor}_SimuInfo.txt."""
    pnr = nsamp = method = None
    with open(infofile, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("pnr"):
                pnr = float(line.split(":")[1])
            elif line.startswith("nsamp"):
                nsamp = int(line.split(":")[1])
            elif line.startswith("method"):
                method = line.split(":")[1]
    if pnr is None or nsamp is None or method is None:
        raise ValueError(f"Missing pnr/nsamp/method in {infofile}")
    return pnr, nsamp, method

def get_lr(alpha: float):
    c = get_tavtigian_c(alpha)
    return get_tavtigian_thresholds(c, alpha), (c ** (1 / 8))[0], (c ** (-1 / 8))[0]

def metric3_p(dtrue, dcalib, Post_p, Post_b, lr_supp_pos):
    dtrue = np.asarray(dtrue, dtype=float).flatten()
    dcalib = np.asarray(dcalib, dtype=float).flatten()
    assert len(dtrue) == len(dcalib)

    # Cap to avoid 0/1 extremes
    dtrue[dtrue >= Post_p[0]] = Post_p[0]
    dcalib[dcalib >= Post_p[0]] = Post_p[0]
    dtrue[dtrue <= (1 - Post_b[0])] = (1 - Post_b[0])
    dcalib[dcalib <= (1 - Post_b[0])] = (1 - Post_b[0])

    oddstr = dtrue / (1 - dtrue)
    oddscl = dcalib / (1 - dcalib)
    oddsratio = oddscl / oddstr
    misest_p = np.log(oddsratio) / math.log(lr_supp_pos)
    return misest_p


def metric3_b(dtrue, dcalib, Post_p, Post_b, lr_supp_pos):
    dtrue = np.asarray(dtrue, dtype=float).flatten()
    dcalib = np.asarray(dcalib, dtype=float).flatten()
    assert len(dtrue) == len(dcalib)

    dtrue[dtrue <= (1 - Post_p[0])] = (1 - Post_p[0])
    dcalib[dcalib <= (1 - Post_p[0])] = (1 - Post_p[0])
    dtrue[dtrue >= Post_b[0]] = Post_b[0]
    dcalib[dcalib >= Post_b[0]] = Post_b[0]

    dtrue_b = dtrue
    dcalib_b = dcalib
    oddstr = dtrue_b / (1 - dtrue_b)
    oddscl = dcalib_b / (1 - dcalib_b)
    oddsratio = oddscl / oddstr
    misest_b = np.log(oddsratio) / math.log(lr_supp_pos)
    return misest_b


def get_pp3_frac(oddsratio, Post_p, dtrue, dcalib):
    threshold = Post_p[4]
    mask = (dcalib > threshold) | (dtrue > threshold)
    if mask.sum() == 0:
        return (0, 0.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    pp3_pos = int(np.argmax(mask))
    or_supp = oddsratio[mask]
    finite_odds = or_supp[np.isfinite(or_supp)]

    if finite_odds.size == 0:
        return (pp3_pos, 0.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    pp3_sum = float(np.sum(np.abs(finite_odds)))
    pp3_50p, pp3_75p, pp3_90p, pp3_max = np.percentile(finite_odds, [50, 75, 90, 100])

    conds = finite_odds >= 1
    pp3_over1 = float(np.sum(conds) / len(finite_odds))

    filtered_odds = finite_odds[finite_odds >= 1]
    if len(filtered_odds) > 0:
        misest_pp3_over1 = float(np.mean(filtered_odds))
        misest90_pp3_over1 = float(np.percentile(filtered_odds, 90))
    else:
        misest_pp3_over1 = np.nan
        misest90_pp3_over1 = np.nan

    return (pp3_pos, pp3_sum, pp3_50p, pp3_75p, pp3_90p, pp3_max, pp3_over1, misest_pp3_over1, misest90_pp3_over1)


def get_bp4_frac(oddsratio, Post_b, dtrue, dcalib):
    threshold = Post_b[4]
    mask = (dcalib > threshold) | (dtrue > threshold)
    flip_mask = np.flip(mask)
    if mask.sum() == 0:
        return (0, 0.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    bp4_pos = int(np.argmin(flip_mask))
    or_supp_b = oddsratio[mask]
    finite_odds = or_supp_b[np.isfinite(or_supp_b)]

    if finite_odds.size == 0:
        return (bp4_pos, 0.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    bp4_sum = float(np.sum(np.abs(finite_odds)))
    bp4_50p, bp4_75p, bp4_90p, bp4_max = np.percentile(finite_odds, [50, 75, 90, 100])
    bp4_under1 = float(np.sum(finite_odds >= 1) / len(finite_odds))

    filtered_odds = finite_odds[finite_odds >= 1]
    if len(filtered_odds) > 0:
        misest_bp4_under1 = float(np.mean(filtered_odds))
        misest90_bp4_under1 = float(np.percentile(filtered_odds, 90))
    else:
        misest_bp4_under1 = np.nan
        misest90_bp4_under1 = np.nan

    return (bp4_pos, bp4_sum, bp4_50p, bp4_75p, bp4_90p, bp4_max, bp4_under1, misest_bp4_under1, misest90_bp4_under1)


# ----------------------------------------------------------------------
# Main metric computation (local + others, MonoPostNN from *others.csv)
# ----------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compute calibration metrics")

    parser.add_argument("--gene", required=True)
    parser.add_argument("--predictor", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--prior", required=True, type=float)

    args = parser.parse_args()
    gene = args.gene
    predictor = args.predictor
    outdir = args.outdir
    alpha = args.prior

    print(f"[Step02] gene={gene}, predictor={predictor}")
    gene_dir = os.path.join(outdir, gene)
    
    # --- Read SimuInfo ---
    infofile = os.path.join(gene_dir, f"{gene}_{predictor}_SimuInfo.txt")
    if not os.path.exists(infofile):
        raise FileNotFoundError(f"Missing SimuInfo: {infofile}")

    pnr, nsamp, method = _parse_simu_info(infofile)
    
    # --- Path to simulation outputs ---
    path = os.path.join(gene_dir, f"{gene}_{predictor}_{method}_Ntrain{nsamp}")
    print(f"Using path: {path}")

    (Post_p, Post_b), lr_supp_pos, lr_supp_neg = get_lr(alpha)
    
    print(f"Working on path: {path}")
    print(f"(Post_p, Post_b): {(Post_p, Post_b)}; lr_supp_pos={lr_supp_pos}; lr_supp_neg={lr_supp_neg}")

    # DataFrames to accumulate per-iteration metrics
    df_pp3_50ps = df_pp3_75ps = df_pp3_90ps = df_pp3_maxs = None
    df_pp3_fracs = df_pp3_misest = df_pp3_misest90 = None
    df_bp4_50ps = df_bp4_75ps = df_bp4_90ps = df_bp4_maxs = None
    df_bp4_fracs = df_bp4_misest = df_bp4_misest90 = None
    df_all_fracs = df_ave_misests = df_ave_abs_diff = None

    if True:
        for i in range(1, 31):
            print(f"[{gene} {predictor} {method}] iteration {i}")

            pkfn = os.path.join(path, f"{predictor}_simu_{method}{i}.0.pkl")
            if not os.path.exists(pkfn):
                print(f"  Missing {pkfn}, skipping")
                continue
            with open(pkfn, "rb") as f:
                _ = pickle.load(f)  # not used directly, but keep for parity

            # ---- LOCAL calib outputs ----
            local_main = os.path.join(path, f"{predictor}_simu_{method}{i}.0_calib_outputs.csv")
            local_p95 = os.path.join(path, f"{predictor}_simu_{method}{i}.0_calib_outputs_P95.csv")
            local_b95 = os.path.join(path, f"{predictor}_simu_{method}{i}.0_calib_outputs_B95.csv")

            # ---- OTHERS calib outputs (include MonoPostNN column) ----
            oth_main = os.path.join(path, f"{predictor}_simu_{method}{i}.0_calib_outputs_others.csv")
            oth_p95 = os.path.join(path, f"{predictor}_simu_{method}{i}.0_calib_outputs_P95_others.csv")
            oth_b95 = os.path.join(path, f"{predictor}_simu_{method}{i}.0_calib_outputs_B95_others.csv")

            # Need at least one of local/others to exist; otherwise skip
            if not (os.path.exists(local_main) or os.path.exists(oth_main)):
                print(f"  No calib_outputs CSV for iteration {i}, skipping")
                continue

            # For each iteration, we build combined misest dicts:
            misest_all = {}     # continuous misest from _calib_outputs (p side)
            misest_p_all = {}   # from P95
            misest_b_all = {}   # from B95
            pp3_50ps = {}
            pp3_75ps = {}
            pp3_90ps = {}
            pp3_maxs = {}
            pp3_fracs = {}
            pp3_misest = {}
            pp3_misest90 = {}

            bp4_50ps = {}
            bp4_75ps = {}
            bp4_90ps = {}
            bp4_maxs = {}
            bp4_fracs = {}
            bp4_misest = {}
            bp4_misest90 = {}

            all_fracs = {}
            ave_misests = {}
            ave_abs_diff = {}

            # ------------- LOCAL calibration part -------------
            if os.path.exists(local_main) and os.path.exists(local_p95) and os.path.exists(local_b95):
                calibd = pd.read_csv(local_main)
                calibd_p = pd.read_csv(local_p95)
                calibd_b = pd.read_csv(local_b95)

                local_methods = [c for c in calibd.columns if c != "True"]
                for op in local_methods:
                    key_main = f"{op}_local"
                    # misest (continuous)
                    misest_all[key_main] = metric3_p(
                        dtrue=calibd["True"],
                        dcalib=calibd[op],
                        Post_p=Post_p,
                        Post_b=Post_b,
                        lr_supp_pos=lr_supp_pos,
                    )

                local_p_methods = [c for c in calibd_p.columns if c != "True"]
                local_b_methods = [c for c in calibd_b.columns if c != "True"]

                # Should match, but we guard anyway
                for op in set(local_p_methods) & set(local_b_methods):
                    key = f"{op}_local"

                    misest_p_all[key] = metric3_p(
                        dtrue=calibd_p["True"],
                        dcalib=calibd_p[op],
                        Post_p=Post_p,
                        Post_b=Post_b,
                        lr_supp_pos=lr_supp_pos,
                    )
                    misest_b_all[key] = metric3_b(
                        dtrue=calibd_b["True"],
                        dcalib=calibd_b[op],
                        Post_p=Post_p,
                        Post_b=Post_b,
                        lr_supp_pos=lr_supp_pos,
                    )

                    # PP3/BP4 summary
                    (
                        pp3_pos,
                        pp3_sum,
                        pp3_50ps[key],
                        pp3_75ps[key],
                        pp3_90ps[key],
                        pp3_maxs[key],
                        pp3_fracs[key],
                        pp3_misest[key],
                        pp3_misest90[key],
                    ) = get_pp3_frac(
                        oddsratio=misest_p_all[key],
                        Post_p=Post_p,
                        dtrue=calibd_p["True"],
                        dcalib=calibd_p[op],
                    )

                    (
                        bp4_pos,
                        bp4_sum,
                        bp4_50ps[key],
                        bp4_75ps[key],
                        bp4_90ps[key],
                        bp4_maxs[key],
                        bp4_fracs[key],
                        bp4_misest[key],
                        bp4_misest90[key],
                    ) = get_bp4_frac(
                        oddsratio=misest_b_all[key],
                        Post_b=Post_b,
                        dtrue=calibd_b["True"],
                        dcalib=calibd_b[op],
                    )

                    # Make over VeryStrong difference zero on pathogenic side
                    tmp_misest_p = misest_p_all[key].copy()
                    mask_vs = (calibd_p["True"].values > Post_p[0]) & (calibd_p[op].values > Post_p[0])
                    tmp_misest_p[mask_vs] = 0
                    misest_p_all[key] = tmp_misest_p

                    # overall fraction of |misest| >= 1 (pathogenic side)
                    all_fracs[key] = float(np.sum(np.abs(misest_p_all[key]) >= 1) / len(misest_p_all[key]))

                    # average misestimation (abs in IR, plus PP3/BP4 sums) / total
                    ave_misests[key] = float(
                        (np.sum(np.abs(misest_all[key][bp4_pos:pp3_pos])) + pp3_sum + bp4_sum)
                        / len(misest_all[key])
                    )

                    # average absolute difference between True and calibrated (P-side P95 table)
                    ave_abs_diff[key] = float(np.mean(np.abs(calibd_p["True"].values - calibd_p[op].values)))

            # ------------- OTHER calibration methods part -------------
            if os.path.exists(oth_main) and os.path.exists(oth_p95) and os.path.exists(oth_b95):
                calibd_o = pd.read_csv(oth_main)
                calibd_o_p = pd.read_csv(oth_p95)
                calibd_o_b = pd.read_csv(oth_b95)

                other_methods = [c for c in calibd_o.columns if c != "True"]
                for op in other_methods:
                    key_main = f"{op}_others"
                    misest_all[key_main] = metric3_p(
                        dtrue=calibd_o["True"],
                        dcalib=calibd_o[op],
                        Post_p=Post_p,
                        Post_b=Post_b,
                        lr_supp_pos=lr_supp_pos,
                    )

                other_p_methods = [c for c in calibd_o_p.columns if c != "True"]
                other_b_methods = [c for c in calibd_o_b.columns if c != "True"]

                for op in set(other_p_methods) & set(other_b_methods):
                    key = f"{op}_others"

                    misest_p_all[key] = metric3_p(
                        dtrue=calibd_o_p["True"],
                        dcalib=calibd_o_p[op],
                        Post_p=Post_p,
                        Post_b=Post_b,
                        lr_supp_pos=lr_supp_pos,
                    )
                    misest_b_all[key] = metric3_b(
                        dtrue=calibd_o_b["True"],
                        dcalib=calibd_o_b[op],
                        Post_p=Post_p,
                        Post_b=Post_b,
                        lr_supp_pos=lr_supp_pos,
                    )

                    (
                        pp3_pos,
                        pp3_sum,
                        pp3_50ps[key],
                        pp3_75ps[key],
                        pp3_90ps[key],
                        pp3_maxs[key],
                        pp3_fracs[key],
                        pp3_misest[key],
                        pp3_misest90[key],
                    ) = get_pp3_frac(
                        oddsratio=misest_p_all[key],
                        Post_p=Post_p,
                        dtrue=calibd_o_p["True"],
                        dcalib=calibd_o_p[op],
                    )

                    (
                        bp4_pos,
                        bp4_sum,
                        bp4_50ps[key],
                        bp4_75ps[key],
                        bp4_90ps[key],
                        bp4_maxs[key],
                        bp4_fracs[key],
                        bp4_misest[key],
                        bp4_misest90[key],
                    ) = get_bp4_frac(
                        oddsratio=misest_b_all[key],
                        Post_b=Post_b,
                        dtrue=calibd_o_b["True"],
                        dcalib=calibd_o_b[op],
                    )

                    tmp_misest_p = misest_p_all[key].copy()
                    mask_vs = (calibd_o_p["True"].values > Post_p[0]) & (calibd_o_p[op].values > Post_p[0])
                    tmp_misest_p[mask_vs] = 0
                    misest_p_all[key] = tmp_misest_p

                    all_fracs[key] = float(np.sum(np.abs(misest_p_all[key]) >= 1) / len(misest_p_all[key]))
                    ave_misests[key] = float(
                        (np.sum(np.abs(misest_all[key][bp4_pos:pp3_pos])) + pp3_sum + bp4_sum)
                        / len(misest_all[key])
                    )
                    ave_abs_diff[key] = float(np.mean(np.abs(calibd_o_p["True"].values - calibd_o_p[op].values)))

            # Nothing for this iteration?
            if len(all_fracs) == 0:
                print(f"  No usable local/others tables for iteration {i}, skipping.")
                continue

            # Attach gene + predictor + iter label
            clustn = str(gene) + "_" + str(predictor)
            idx_pp3 = f"{clustn}_pp3_50ps_{i}"
            idx_bp4 = f"{clustn}_bp4_50ps_{i}"
            idx_all = f"{clustn}_all_fracs_{i}"
            idx_ave = f"{clustn}_ave_misests_{i}"
            idx_abs = f"{clustn}_ave_abs_diff_{i}"

            # Initialize or append DataFrames
            if df_pp3_50ps is None:
                df_pp3_50ps = pd.DataFrame(pp3_50ps, index=[idx_pp3])
                df_pp3_75ps = pd.DataFrame(pp3_75ps, index=[f"{clustn}_pp3_75ps_{i}"])
                df_pp3_90ps = pd.DataFrame(pp3_90ps, index=[f"{clustn}_pp3_90ps_{i}"])
                df_pp3_maxs = pd.DataFrame(pp3_maxs, index=[f"{clustn}_pp3_maxs_{i}"])
                df_pp3_fracs = pd.DataFrame(pp3_fracs, index=[f"{clustn}_pp3_fracs_{i}"])
                df_pp3_misest = pd.DataFrame(pp3_misest, index=[f"{clustn}_pp3_misest_{i}"])
                df_pp3_misest90 = pd.DataFrame(pp3_misest90, index=[f"{clustn}_pp3_misest90_{i}"])

                df_bp4_50ps = pd.DataFrame(bp4_50ps, index=[idx_bp4])
                df_bp4_75ps = pd.DataFrame(bp4_75ps, index=[f"{clustn}_bp4_75ps_{i}"])
                df_bp4_90ps = pd.DataFrame(bp4_90ps, index=[f"{clustn}_bp4_90ps_{i}"])
                df_bp4_maxs = pd.DataFrame(bp4_maxs, index=[f"{clustn}_bp4_maxs_{i}"])
                df_bp4_fracs = pd.DataFrame(bp4_fracs, index=[f"{clustn}_bp4_fracs_{i}"])
                df_bp4_misest = pd.DataFrame(bp4_misest, index=[f"{clustn}_bp4_misest_{i}"])
                df_bp4_misest90 = pd.DataFrame(bp4_misest90, index=[f"{clustn}_bp4_misest90_{i}"])

                df_all_fracs = pd.DataFrame(all_fracs, index=[idx_all])
                df_ave_misests = pd.DataFrame(ave_misests, index=[idx_ave])
                df_ave_abs_diff = pd.DataFrame(ave_abs_diff, index=[idx_abs])
            else:
                df_pp3_50ps = pd.concat([df_pp3_50ps, pd.DataFrame(pp3_50ps, index=[idx_pp3])])
                df_pp3_75ps = pd.concat([df_pp3_75ps, pd.DataFrame(pp3_75ps, index=[f"{clustn}_pp3_75ps_{i}"])])
                df_pp3_90ps = pd.concat([df_pp3_90ps, pd.DataFrame(pp3_90ps, index=[f"{clustn}_pp3_90ps_{i}"])])
                df_pp3_maxs = pd.concat([df_pp3_maxs, pd.DataFrame(pp3_maxs, index=[f"{clustn}_pp3_maxs_{i}"])])
                df_pp3_fracs = pd.concat([df_pp3_fracs, pd.DataFrame(pp3_fracs, index=[f"{clustn}_pp3_fracs_{i}"])])
                df_pp3_misest = pd.concat([df_pp3_misest, pd.DataFrame(pp3_misest, index=[f"{clustn}_pp3_misest_{i}"])])
                df_pp3_misest90 = pd.concat([df_pp3_misest90, pd.DataFrame(pp3_misest90, index=[f"{clustn}_pp3_misest90_{i}"])])

                df_bp4_50ps = pd.concat([df_bp4_50ps, pd.DataFrame(bp4_50ps, index=[idx_bp4])])
                df_bp4_75ps = pd.concat([df_bp4_75ps, pd.DataFrame(bp4_75ps, index=[f"{clustn}_bp4_75ps_{i}"])])
                df_bp4_90ps = pd.concat([df_bp4_90ps, pd.DataFrame(bp4_90ps, index=[f"{clustn}_bp4_90ps_{i}"])])
                df_bp4_maxs = pd.concat([df_bp4_maxs, pd.DataFrame(bp4_maxs, index=[f"{clustn}_bp4_maxs_{i}"])])
                df_bp4_fracs = pd.concat([df_bp4_fracs, pd.DataFrame(bp4_fracs, index=[f"{clustn}_bp4_fracs_{i}"])])
                df_bp4_misest = pd.concat([df_bp4_misest, pd.DataFrame(bp4_misest, index=[f"{clustn}_bp4_misest_{i}"])])
                df_bp4_misest90 = pd.concat([df_bp4_misest90, pd.DataFrame(bp4_misest90, index=[f"{clustn}_bp4_misest90_{i}"])])

                df_all_fracs = pd.concat([df_all_fracs, pd.DataFrame(all_fracs, index=[idx_all])])
                df_ave_misests = pd.concat([df_ave_misests, pd.DataFrame(ave_misests, index=[idx_ave])])
                df_ave_abs_diff = pd.concat([df_ave_abs_diff, pd.DataFrame(ave_abs_diff, index=[idx_abs])])

    # If nothing accumulated, just exit quietly
    if df_pp3_50ps is None:
        print("No metrics computed (no iterations with valid outputs). Exiting.")
        return

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    dir_path = os.path.join(gene_dir, f"{predictor}_{method}_calib_metric")
    os.makedirs(dir_path, exist_ok=True)

    df_pp3_50ps.to_csv(os.path.join(dir_path, "pp3_50ps_combined.csv"))
    df_pp3_75ps.to_csv(os.path.join(dir_path, "pp3_75ps_combined.csv"))
    df_pp3_90ps.to_csv(os.path.join(dir_path, "pp3_90ps_combined.csv"))
    df_pp3_maxs.to_csv(os.path.join(dir_path, "pp3_maxs_combined.csv"))
    df_pp3_fracs.to_csv(os.path.join(dir_path, "pp3_fracs_combined.csv"))
    df_pp3_misest.to_csv(os.path.join(dir_path, "pp3_misest_combined.csv"))
    df_pp3_misest90.to_csv(os.path.join(dir_path, "pp3_misest90_combined.csv"))

    df_bp4_50ps.to_csv(os.path.join(dir_path, "bp4_50ps_combined.csv"))
    df_bp4_75ps.to_csv(os.path.join(dir_path, "bp4_75ps_combined.csv"))
    df_bp4_90ps.to_csv(os.path.join(dir_path, "bp4_90ps_combined.csv"))
    df_bp4_maxs.to_csv(os.path.join(dir_path, "bp4_maxs_combined.csv"))
    df_bp4_fracs.to_csv(os.path.join(dir_path, "bp4_fracs_combined.csv"))
    df_bp4_misest.to_csv(os.path.join(dir_path, "bp4_misest_combined.csv"))
    df_bp4_misest90.to_csv(os.path.join(dir_path, "bp4_misest90_combined.csv"))

    df_all_fracs.to_csv(os.path.join(dir_path, "all_fracs_combined.csv"))
    df_ave_misests.to_csv(os.path.join(dir_path, "ave_misests_combined.csv"))
    df_ave_abs_diff.to_csv(os.path.join(dir_path, "ave_abs_diff_combined.csv"))

    print(f"Saved combined metric CSVs under: {dir_path}")


if __name__ == "__main__":
    main()
