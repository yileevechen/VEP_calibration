import sys
import os
import re
import subprocess
from statistics import median


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _parse_sim_info(sim_info_path: str):
    """
    Parse {gene}_{predictor}_SimuInfo.txt and return (pnr, nsamp, method).
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

def _run_python_module(module: str, args: list[str]):
    """
    Convenience wrapper for calling: python -m <module> <args...>
    Raises if the subprocess fails.
    """
    cmd = ["python", "-m", module] + args
    print(f"[RUN] {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True)

def run_calibration(
    gene: str,
    predictor: str,
    base_dir: str,
    alpha: float,
    seed: int,
    n_test: int = 1000,
):
    """
    Run calibration using outputs from step00.

    Inputs:
      - gene
      - predictor
      - base_dir (user-provided output dir)
      - alpha (user-provided prior)
      - seed
    """

    gene_dir = os.path.join(base_dir, gene)

    # --- 1. Read SimuInfo ---
    sim_info_path = os.path.join(
        gene_dir, f"{gene}_{predictor}_SimuInfo.txt"
    )
    if not os.path.exists(sim_info_path):
        raise FileNotFoundError(f"SimuInfo file not found: {sim_info_path}")

    pnratio_calibrate, n_calibrate, method = _parse_sim_info(sim_info_path)
    pnratio_test = pnratio_calibrate

    print(f"[INFO] gene={gene}, predictor={predictor}, method={method}")
    print(f"[INFO] alpha={alpha}, seed={seed}")

    # --- 2. Output directory (consistent with step00) ---
    sim_outdir = os.path.join(
        gene_dir,
        f"{predictor}_{gene}_{method}_Ntrain{n_calibrate}",
    )

    # --- 3. Check local calibration output ---
    local_b95_path = os.path.join(
        sim_outdir,
        f"{predictor}_simu_{method}{seed}_calib_outputs_B95.csv",
    )

    # --- 4. Local calibration ---
    if os.path.exists(local_b95_path):
        print(f"[SKIP] Local calibration exists:\n  {local_b95_path}")
    else:
        print(f"[RUN] Local calibration: {gene}, {predictor}, seed={seed}")
        _run_python_module(
            "calib_pipeline.local_calib",
            [
                "--seed", str(seed),
                "--predictor", predictor,
                "--clustn", gene,
                "--pnratio_calibrate", str(pnratio_calibrate),
                "--pnratio_test", str(pnratio_test),
                "--alpha", str(alpha),
                f"--n_calibrate={n_calibrate}",
                f"--n_test={n_test}",
                f"--outdir={gene_dir}",
                "--method", method,
            ],
        )

    # --- 5. Other calibration ---
    print(f"[RUN] Other calibration: {gene}, {predictor}, seed={seed}")
    _run_python_module(
        "calib_pipeline.other_calib",
        [
            "--seed", str(seed),
            "--predictor", predictor,
            "--clustn", gene,
            "--pnratio_calibrate", str(pnratio_calibrate),
            "--pnratio_test", str(pnratio_test),
            "--alpha", str(alpha),
            f"--n_calibrate={n_calibrate}",
            f"--n_test={n_test}",
            f"--outdir={gene_dir}",
            "--method", method,
        ],
    )

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run calibration step (Step01).")

    parser.add_argument("--gene", required=True)
    parser.add_argument("--predictor", required=True)
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n_test", type=int, default=1000)

    args = parser.parse_args()

    run_calibration(
        gene=args.gene,
        predictor=args.predictor,
        base_dir=args.base_dir,
        alpha=args.alpha,
        seed=args.seed,
        n_test=args.n_test,
    )

if __name__ == "__main__":
    main()
