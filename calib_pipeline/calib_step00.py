import os
import argparse
import pandas as pd
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from calib_pipeline.get_best_fit import fit_best_distribution, GetBestFitConfig
from calib_pipeline.data_generation import generate_simulation_data, DataGenConfig


@dataclass
class Step00Config:
    n_test: int = 1000  #number of samples in the test set
    n_seeds: int = 30   #number of simulation repeats
    n_jobs: int = 6

def _validate_inputs(labeled_path: str, unlabeled_path: str):
    if not os.path.exists(labeled_path):
        raise FileNotFoundError(f"Labeled file not found: {labeled_path}")
    if not os.path.exists(unlabeled_path):
        raise FileNotFoundError(f"Unlabeled file not found: {unlabeled_path}")

    # Validate labeled format
    df_lab = pd.read_csv(labeled_path, sep="\t", header=None)
    if df_lab.shape[1] != 2:
        raise ValueError("Labeled file must have exactly 2 columns: score, label")

    if not set(df_lab[1].unique()).issubset({0, 1}):
        raise ValueError("Labels must be binary (0/1)")

    # Validate unlabeled format
    df_unlab = pd.read_csv(unlabeled_path, sep="\t", header=None)
    if df_unlab.shape[1] != 1:
        raise ValueError("Unlabeled file must have exactly 1 column: score")

    print(f"[OK] Input validation passed: {labeled_path}, {unlabeled_path}")


def run_simulation(
    gene: str,
    predictor: str,
    alpha: float,
    outdir: str,
    labeled_path: str,
    unlabeled_path: str,
    cfg: Step00Config,
):
    print(f"[Step00] gene={gene}, predictor={predictor}, prior={alpha}")

    base_out = os.path.join(outdir, gene)
    os.makedirs(base_out, exist_ok=True)

    # --- Validate user-provided inputs ---
    _validate_inputs(labeled_path, unlabeled_path)

    # --- Step 1: Fit distribution using user data ---
    gb_cfg = GetBestFitConfig(base_output_dir=outdir)

    fit_best_distribution(
        gene=gene,
        predictor=predictor,
        labeled_file=labeled_path,
        unlabeled_file=unlabeled_path,
        cfg=gb_cfg,
        make_plot=True,
    )

    # --- Step 2: Read simulation info ---
    simu_info_path = os.path.join(base_out, f"{gene}_{predictor}_SimuInfo.txt")
    if not os.path.exists(simu_info_path):
        raise FileNotFoundError(f"SimuInfo file not found: {simu_info_path}")

    pnr = None
    nsamp = None
    method = None

    with open(simu_info_path) as f:
        for line in f:
            if line.startswith("pnr:"):
                pnr = float(line.strip().split(":", 1)[1])
            elif line.startswith("nsamp:"):
                nsamp = int(line.strip().split(":", 1)[1])
            elif line.startswith("method:"):
                method = line.strip().split(":", 1)[1]

    if pnr is None or nsamp is None or method is None:
        raise ValueError(f"Incomplete SimuInfo in {simu_info_path}")

    print(f"SimuInfo: pnr={pnr}, nsamp={nsamp}, method={method}")

    # --- Step 3: Simulation ---
    dg_cfg = DataGenConfig(base_output_dir=outdir)

    def _worker(seed_idx: int):
        return generate_simulation_data(
            gene=gene,
            predictor=predictor,
            method=method,
            seed_index=seed_idx,
            outdir=base_out,
            alpha=float(alpha),
            n_calibrate=int(nsamp),
            n_test=cfg.n_test,
            pnratio_calibrate=float(pnr),
            pnratio_test=float(pnr),
            cfg=dg_cfg,
            labeled_file=labeled_path,
            unlabeled_file=unlabeled_path,
        )

    futures = []
    with ThreadPoolExecutor(max_workers=cfg.n_jobs) as ex:
        for s in range(1, cfg.n_seeds + 1):
            futures.append(ex.submit(_worker, s))

        for fut in as_completed(futures):
            try:
                fname = fut.result()
                print(f"Generated simulation file: {fname}")
            except Exception as e:
                print(f"Simulation worker error: {e}")

    print(f"[Step00] Done for gene={gene}, predictor={predictor}")
	
def main():
    parser = argparse.ArgumentParser(
        description="Run simulation pipeline with user-provided labeled/unlabeled data"
    )

    parser.add_argument("--gene", required=True)
    parser.add_argument("--predictor", required=True)
    parser.add_argument("--prior", required=True, type=float)
    parser.add_argument("--outdir", required=True)

    parser.add_argument("--labeled", required=True, help="Path to labeled file")
    parser.add_argument("--unlabeled", required=True, help="Path to unlabeled file")

    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--n_seeds", type=int, default=30)
    parser.add_argument("--n_jobs", type=int, default=6)

    args = parser.parse_args()

    cfg = Step00Config(
        n_test=args.n_test,
        n_seeds=args.n_seeds,
        n_jobs=args.n_jobs,
    )

    run_simulation(
        gene=args.gene,
        predictor=args.predictor,
        alpha=args.prior,
        outdir=args.outdir,
        labeled_path=args.labeled,
        unlabeled_path=args.unlabeled,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
