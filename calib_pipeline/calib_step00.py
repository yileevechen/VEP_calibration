\
import os
import sys
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from calib_pipeline.prepare_data import prepare_gene_data, PrepareDataPaths
from calib_pipeline.get_best_fit import fit_best_distribution, GetBestFitConfig
from calib_pipeline.data_generation import generate_simulation_data, DataGenConfig


@dataclass
class Step00Config:
    base_dir: str = (
        "/sc/arion/projects/pejaverlab/IGVF/users/cheny60/analysis/"
        "calibrationexp-main/single_gene_calibration_pipeline"
    )
    gene_dist_list: str = (
        "/sc/arion/projects/pejaverlab/IGVF/users/cheny60/analysis/"
        "calibrationexp-main/prior_notfiltmp2train_calib_decision_tree/00.gene_dist.list"
    )
    prior_base: str = (
        "/sc/arion/projects/pejaverlab/IGVF/users/cheny60/analysis/"
        "calibrationexp-main/calib_decision_tree/inheritance_analysis/prior_gnomad"
    )
    n_test: int = 1000
    n_seeds: int = 30
    n_jobs: int = 6


def _read_gene_dist_by_index(list_path: str, idx: int) -> tuple[str, str]:
    with open(list_path) as f:
        for i, line in enumerate(f, start=1):
            if i == idx:
                parts = line.strip().split()
                if len(parts) < 2:
                    raise ValueError(f"Line {idx} in {list_path} does not have gene and dist.")
                return parts[0], parts[1]
    raise IndexError(f"Index {idx} out of range for {list_path}.")


def _compute_median_prior(prior_base: str, gene: str) -> float:
    import re
    prior_path = os.path.join(
        prior_base,
        gene,
        "uniboot_rus_pca_notfiltmp2train",
        "cluster_gnomad.res_btstrap.txt",
    )
    if not os.path.exists(prior_path):
        raise FileNotFoundError(f"Prior file not found for {gene}: {prior_path}")

    with open(prior_path) as f:
        txt = f.read()
    vals = [float(x) for x in re.findall(r"The prior is:\s*([0-9.]+)", txt) if ".." not in x]
    if not vals:
        raise ValueError(f"No valid priors found in {prior_path}")
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    if n % 2 == 1:
        return vals_sorted[n // 2]
    return (vals_sorted[n // 2 - 1] + vals_sorted[n // 2]) / 2.0


def run_simulation_step_for_index(array_idx: int, cfg: Step00Config | None = None):
    if cfg is None:
        cfg = Step00Config()

    gene, dist = _read_gene_dist_by_index(cfg.gene_dist_list, array_idx)
    print(f"[Step00] ARRAY_IDX={array_idx} -> gene={gene}, dist={dist}")

    alpha = _compute_median_prior(cfg.prior_base, gene)
    print(f"Median prior for {gene} = {alpha}")

    base_out = os.path.join(cfg.base_dir, gene)
    os.makedirs(base_out, exist_ok=True)

    prep_paths = PrepareDataPaths(base_output_dir=cfg.base_dir)
    prepare_gene_data(gene, paths=prep_paths)

    gb_cfg = GetBestFitConfig(base_output_dir=cfg.base_dir)
    fit_best_distribution(gene, dist, cfg=gb_cfg, make_plot=True)

    simu_info_path = os.path.join(base_out, f"new{gene}_{dist}_SimuInfo.txt")
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

    dg_cfg = DataGenConfig(base_output_dir=cfg.base_dir)

    def _worker(seed_idx: int):
        return generate_simulation_data(
            gene=gene,
            dist=dist,
            method=method,
            seed_index=seed_idx,
            outdir=base_out,
            alpha=float(alpha),
            n_calibrate=int(nsamp),
            n_test=cfg.n_test,
            pnratio_calibrate=float(pnr),
            pnratio_test=float(pnr),
            cfg=dg_cfg,
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

    print(f"[Step00] Done for gene={gene}, dist={dist}")


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python -m calib_pipeline.sim_data_step00 ARRAY_IDX")

    array_idx = int(sys.argv[1])
    run_simulation_step_for_index(array_idx)


if __name__ == "__main__":
    main()
