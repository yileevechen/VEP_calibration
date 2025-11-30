import sys
import os
import re
import subprocess
from statistics import median


# ---------------------------------------------------------------------
# Constants / paths
# ---------------------------------------------------------------------

BASE_DIR = (
    "/sc/arion/projects/pejaverlab/IGVF/users/cheny60/analysis/"
    "calibrationexp-main/single_gene_calibration_pipeline"
)

GENE_SEED_LIST = os.path.join(BASE_DIR, "01.gene_seed_AM_MP2.list")

PRIOR_BASE = (
    "/sc/arion/projects/pejaverlab/IGVF/users/cheny60/analysis/"
    "calibrationexp-main/calib_decision_tree/inheritance_analysis/prior_gnomad"
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _read_line_from_file(path: str, idx: int) -> str:
    """Return 1-based line `idx` from `path` (stripped)."""
    with open(path, "r") as f:
        for i, line in enumerate(f, start=1):
            if i == idx:
                return line.strip()
    raise IndexError(f"File {path} has fewer than {idx} lines.")


def _parse_sim_info(sim_info_path: str):
    """
    Parse new{gene}_{dist}_SimuInfo.txt and return (pnr, nsamp, method).
    Expected lines contain 'pnr', 'nsamp', 'method' as 'key:value'.
    """
    pnr = None
    nsamp = None
    method = None

    with open(sim_info_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, val = [x.strip() for x in line.split(":", 1)]
            if key == "pnr":
                pnr = float(val)
            elif key == "nsamp":
                nsamp = int(val)
            elif key == "method":
                method = val

    if pnr is None or nsamp is None or method is None:
        raise ValueError(f"Missing pnr/nsamp/method in {sim_info_path}")

    return pnr, nsamp, method


def _read_priors_from_file(path: str):
    """
    Return list of valid float priors parsed from 'The prior is: <num>' lines.
    Skips malformed numbers like '0..07801725'.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r") as f:
        txt = f.read()

    raw_vals = re.findall(r"The prior is:\s*([0-9.]+)", txt)
    vals = []
    for s in raw_vals:
        try:
            vals.append(float(s))
        except ValueError:
            # skip malformed like "0..07801725"
            continue

    if not vals:
        raise ValueError(f"No valid priors parsed from {path}")

    return vals


def _compute_median_prior(gene: str) -> float:
    """
    Compute median prior for a gene from cluster_gnomad.res_btstrap.txt.
    """
    prior_file = os.path.join(
        PRIOR_BASE,
        gene,
        "uniboot_rus_pca_notfiltmp2train",
        "cluster_gnomad.res_btstrap.txt",
    )
    vals = _read_priors_from_file(prior_file)
    return float(median(vals))


def _run_python_module(module: str, args: list[str]):
    """
    Convenience wrapper for calling: python -m <module> <args...>
    Raises if the subprocess fails.
    """
    cmd = ["python", "-m", module] + args
    print(f"[RUN] {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------
# Main per-index logic
# ---------------------------------------------------------------------

def run_calibration_step_for_index(array_idx: int) -> None:
    """
    For a given ARRAY_IDX:
      1. Look up (gene, dist, seed) from 01.gene_seed_AM_MP2.list
      2. Read SimuInfo for that gene/dist
      3. Compute alpha from prior file
      4. Run local calibration *unless* local B95 output already exists
      5. Run other calibration
    """
    # --- 1. Parse gene/dist/seed from list file ---
    line = _read_line_from_file(GENE_SEED_LIST, array_idx)
    # Expect: gene dist seed
    parts = line.split()
    if len(parts) < 3:
        raise ValueError(
            f"Line {array_idx} in {GENE_SEED_LIST} does not have 3 fields: '{line}'"
        )
    gene, dist, seed_str = parts[0], parts[1], parts[2]
    seed = int(seed_str)

    # --- 2. SimuInfo file: pnr, nsamp, method ---
    gene_dir = os.path.join(BASE_DIR, gene)
    sim_info_path = os.path.join(gene_dir, f"new{gene}_{dist}_SimuInfo.txt")
    if not os.path.exists(sim_info_path):
        raise FileNotFoundError(f"SimuInfo file not found: {sim_info_path}")

    pnratio_calibrate, n_calibrate, method = _parse_sim_info(sim_info_path)
    pnratio_test = pnratio_calibrate  # same as in your bash wrapper
    n_test = 1000

    # --- 3. Compute alpha (median prior) ---
    alpha = _compute_median_prior(gene)
    print(f"Median prior for {gene} = {alpha}")

    # --- 4. Build local-calibration output path and decide whether to skip ---
    # local_calib.py and other_calib.py use:
    #   outdir = os.path.join(outdir, f"{dist}_{gene}_{method}_Ntrain{n_calibrate}")
    # and file name pattern:
    #   f"{dist}_simu_{method}{seed}_calib_outputs_B95.csv"
    local_outdir = os.path.join(
        gene_dir, f"{dist}_{gene}_{method}_Ntrain{n_calibrate}"
    )
    local_b95_path = os.path.join(
        local_outdir, f"{dist}_simu_{method}{seed}_calib_outputs_B95.csv"
    )

    # --- 4a. Run local calibration if needed ---
    if os.path.exists(local_b95_path):
        print(
            f"[SKIP] Local calibration already exists for "
            f"{gene} {dist} seed {seed} at:\n  {local_b95_path}"
        )
    else:
        print(
            f"[LOCAL] Running local calibration for {gene}, {dist}, seed {seed}"
        )
        _run_python_module(
            "calib_pipeline.local_calib",
            [
                "--seed",
                str(seed),
                "--dist",
                dist,
                "--clustn",
                gene,
                "--pnratio_calibrate",
                str(pnratio_calibrate),
                "--pnratio_test",
                str(pnratio_test),
                "--alpha",
                str(alpha),
                f"--n_calibrate={n_calibrate}",
                "--n_test=1000",
                f"--outdir={gene_dir}",
                "--method",
                method,
            ],
        )

    # --- 5. Always run other calibration (it has its own internal skip checks) ---
    print(
        f"[OTHER] Running other calibration for {gene}, {dist}, seed {seed}"
    )
    _run_python_module(
        "calib_pipeline.other_calib",
        [
            "--seed",
            str(seed),
            "--dist",
            dist,
            "--clustn",
            gene,
            "--pnratio_calibrate",
            str(pnratio_calibrate),
            "--pnratio_test",
            str(pnratio_test),
            "--alpha",
            str(alpha),
            f"--n_calibrate={n_calibrate}",
            "--n_test=1000",
            f"--outdir={gene_dir}",
            "--method",
            method,
        ],
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print(
            "Usage: python -m calib_pipeline.calib_step01 <ARRAY_IDX>",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        array_idx = int(sys.argv[1])
    except ValueError:
        print("ARRAY_IDX must be an integer.", file=sys.stderr)
        sys.exit(1)

    run_calibration_step_for_index(array_idx)


if __name__ == "__main__":
    main()

