import sys
import os
import re
import math
import pickle
import numpy as np
import pandas as pd

TAV_BASE = "/sc/arion/projects/pejaverlab/IGVF/users/cheny60/analysis/calibrationexp-main/calib_decision_tree/clingen-svi-comp_calibration_python-master"
if TAV_BASE not in sys.path:
    sys.path.append(TAV_BASE)

from Tavtigian.tavtigianutils import (
    get_tavtigian_c,
    get_tavtigian_thresholds,
    get_tavtigian_plr,
)

# ----------------------------------------------------------------------
# Paths / constants
# ----------------------------------------------------------------------

BASE = "/sc/arion/projects/pejaverlab/IGVF/users/cheny60/analysis/calibrationexp-main"
PRIOR_BASE = os.path.join(BASE, "calib_decision_tree", "inheritance_analysis", "prior_gnomad")
CALIB_BASE = os.path.join(BASE, "single_gene_calibration_pipeline")
GENE_DIST_LIST = os.path.join(CALIB_BASE, "00.gene_dist.list")


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------

def _read_line(filename: str, idx: int) -> str:
    """Return 1-based line idx from filename."""
    with open(filename, "r") as f:
        for i, line in enumerate(f, start=1):
            if i == idx:
                return line.strip()
    raise IndexError(f"File {filename} has fewer than {idx} lines.")


def _parse_simu_info(infofile: str):
    """Parse pnr, nsamp, method from new{gene}_{dist}_SimuInfo.txt."""
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


def _read_alpha_prior(gene: str) -> float:
    """
    Read median prior alpha from prior_gnomad ... cluster_gnomad.res_btstrap.txt,
    robust to malformed numbers like '0..07801725'.
    """
    prior_file = os.path.join(
        PRIOR_BASE,
        gene,
        "uniboot_rus_pca_notfiltmp2train",
        "cluster_gnomad.res_btstrap.txt",
    )
    if not os.path.exists(prior_file):
        raise FileNotFoundError(f"Prior file not found: {prior_file}")

    with open(prior_file, "r") as f:
        txt = f.read()

    # Extract values after "The prior is:"
    raw_vals = re.findall(r"The prior is:\s*([0-9.]+)", txt)
    vals = []
    for s in raw_vals:
        try:
            vals.append(float(s))
        except ValueError:
            # Skip malformed floats like "0..078..."
            continue

    if not vals:
        raise ValueError(f"No valid prior values found in {prior_file}")

    vals = sorted(vals)
    n = len(vals)
    if n % 2 == 1:
        return vals[n // 2]
    else:
        return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


# ----------------------------------------------------------------------
# Tavtigian-related helpers
# ----------------------------------------------------------------------

def get_lr(alpha: float):
    c = get_tavtigian_c(alpha)
    return get_tavtigian_thresholds(c, alpha), (c ** (1 / 8))[0], (c ** (-1 / 8))[0]


def monoincrease_transform_p(posterior):
    posterior = np.array(posterior, dtype=float)

    # "Fix" early dip
    if len(posterior) > 0:
        min_idx = np.argmin(posterior[0:min(100, len(posterior))])
        posterior[:min_idx] = posterior[min_idx]

        max_idx = np.argmax(posterior)
        posterior[max_idx + 1:] = np.maximum.accumulate(posterior[max_idx + 1:])
        posterior[max_idx + 1:] = posterior[max_idx]

    return posterior


def monoincrease_transform_b(posterior):
    posterior = np.array(posterior, dtype=float)

    if len(posterior) > 0:
        min_idx = np.argmin(posterior[0:min(100, len(posterior))])
        posterior[:min_idx] = posterior[min_idx]

        max_idx = np.argmax(posterior)
        posterior[max_idx + 1:] = np.maximum.accumulate(posterior[max_idx + 1:])
        posterior[max_idx + 1:] = posterior[max_idx]

    return posterior


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
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m calib_pipeline.calib_step02 ARRAY_IDX")

    array_idx = int(sys.argv[1])

    # ----- Resolve gene & dist from 00.gene_dist.list -----
    line = _read_line(GENE_DIST_LIST, array_idx)
    parts = line.split()
    if len(parts) < 2:
        raise ValueError(f"Line {array_idx} in {GENE_DIST_LIST} does not have 'gene dist'")
    gene, dist = parts[0], parts[1]

    gene_dir = os.path.join(CALIB_BASE, gene)
    os.makedirs(gene_dir, exist_ok=True)

    # ----- alpha (median prior) -----
    alpha = _read_alpha_prior(gene)
    (Post_p, Post_b), lr_supp_pos, lr_supp_neg = get_lr(alpha=alpha)

    # ----- pnr, nsamp, method from SimuInfo -----
    infofile = os.path.join(gene_dir, f"new{gene}_{dist}_SimuInfo.txt")
    if not os.path.exists(infofile):
        raise FileNotFoundError(f"SimuInfo file not found: {infofile}")

    pnr, nsamp, method = _parse_simu_info(infofile)
    n_calibrate = nsamp

    # Base path for this gene / dist / method
    path = os.path.join(gene_dir, f"{dist}_{gene}_{method}_Ntrain{n_calibrate}")
    print(f"Working on path: {path}")
    print(f"(Post_p, Post_b): {(Post_p, Post_b)}; lr_supp_pos={lr_supp_pos}; lr_supp_neg={lr_supp_neg}")

    # DataFrames to accumulate per-iteration metrics
    df_pp3_50ps = df_pp3_75ps = df_pp3_90ps = df_pp3_maxs = None
    df_pp3_fracs = df_pp3_misest = df_pp3_misest90 = None
    df_bp4_50ps = df_bp4_75ps = df_bp4_90ps = df_bp4_maxs = None
    df_bp4_fracs = df_bp4_misest = df_bp4_misest90 = None
    df_all_fracs = df_ave_misests = df_ave_abs_diff = None

    # We only have clustn=1 in these scripts, but keep it explicit for extension
    for clustn in range(1, 2):
        for i in range(1, 31):
            print(f"[{gene} {dist} {method}] iteration {i}")

            pkfn = os.path.join(path, f"{dist}_simu_{method}{i}.0.pkl")
            if not os.path.exists(pkfn):
                print(f"  Missing {pkfn}, skipping")
                continue
            with open(pkfn, "rb") as f:
                _ = pickle.load(f)  # not used directly, but keep for parity

            # ---- LOCAL calib outputs ----
            local_main = os.path.join(path, f"{dist}_simu_{method}{i}.0_calib_outputs.csv")
            local_p95 = os.path.join(path, f"{dist}_simu_{method}{i}.0_calib_outputs_P95.csv")
            local_b95 = os.path.join(path, f"{dist}_simu_{method}{i}.0_calib_outputs_B95.csv")

            # ---- OTHERS calib outputs (include MonoPostNN column) ----
            oth_main = os.path.join(path, f"{dist}_simu_{method}{i}.0_calib_outputs_others.csv")
            oth_p95 = os.path.join(path, f"{dist}_simu_{method}{i}.0_calib_outputs_P95_others.csv")
            oth_b95 = os.path.join(path, f"{dist}_simu_{method}{i}.0_calib_outputs_B95_others.csv")

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

            # ------------- LOCAL part -------------
            if os.path.exists(local_main) and os.path.exists(local_p95) and os.path.exists(local_b95):
                calibd = pd.read_csv(local_main)
                calibd_p = pd.read_csv(local_p95)
                calibd_b = pd.read_csv(local_b95)

                # enforce monotonic transformation on "True"
                calibd["True"] = monoincrease_transform_p(calibd["True"])
                calibd_p["True"] = monoincrease_transform_p(calibd_p["True"])
                calibd_b["True"] = monoincrease_transform_b(calibd_b["True"])

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

            # ------------- OTHERS part (including MonoPostNN from CSV) -------------
            if os.path.exists(oth_main) and os.path.exists(oth_p95) and os.path.exists(oth_b95):
                calibd_o = pd.read_csv(oth_main)
                calibd_o_p = pd.read_csv(oth_p95)
                calibd_o_b = pd.read_csv(oth_b95)

                calibd_o["True"] = monoincrease_transform_p(calibd_o["True"])
                calibd_o_p["True"] = monoincrease_transform_p(calibd_o_p["True"])
                calibd_o_b["True"] = monoincrease_transform_b(calibd_o_b["True"])

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

            # Attach clustn + iter label
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
    dir_path = os.path.join(path, f"../{dist}_{method}_calib_metric")
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
