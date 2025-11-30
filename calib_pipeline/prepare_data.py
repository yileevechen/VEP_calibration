import os
import sys
import itertools
import numpy as np
import pandas as pd
from dataclasses import dataclass


def parse_preds(s: str) -> dict:
    """Parse the VEP-style prediction string into AF fields."""
    pred_data = {
        "gnomADe_AF": np.nan,
        "gnomADg_AF": np.nan,
    }
    if pd.isna(s):
        return pred_data

    for item in str(s).split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            if key in pred_data:
                try:
                    pred_data[key] = float(value)
                except ValueError:
                    pred_data[key] = np.nan
    return pred_data


def get_aa(row: pd.Series) -> str:
    """Reproduce original AA construction: ref/alt from col 10, position from col 9."""
    ref, alt = str(row[10]).split("/")
    return f"{ref}{row[9]}{alt}"


@dataclass
class PrepareDataPaths:
    """Configuration for input/output paths used in the prepare-data step."""

    base_output_dir: str = (
        "/sc/arion/projects/pejaverlab/IGVF/users/cheny60/analysis/"
        "calibrationexp-main/single_gene_calibration_pipeline"
    )

    mp2_training_file: str = (
        "/sc/arion/projects/pejaverlab/IGVF/data/mutpred2_actual_training_data/"
        "mp2_actual_training_data.txt"
    )

    revel_training_file: str = (
        "/sc/arion/projects/pejaverlab/IGVF/users/cheny60/data/"
        "clustering_variants_revel_training_overlap.csv"
    )

    clinvar_scores_file: str = (
        "/sc/arion/projects/pejaverlab/IGVF/users/cheny60/analysis/prior_est/"
        "data/test_pipeline/ClinVar_2025-01/aasnvGRCh38_2025-01.preds_clean.csv"
    )

    gnomad_vepanno_dir: str = (
        "/sc/arion/projects/pejaverlab/IGVF/users/cheny60/analysis/"
        "calibrationexp-main/cluster_data/00.mutpred_dat/gnomAD_vepanno"
    )

    gnomad_preds_dir: str = (
        "/sc/arion/projects/pejaverlab/IGVF/users/cheny60/analysis/"
        "calibrationexp-main/cluster_data/00.mutpred_dat/gnomAD_preds"
    )

    gnomad_scores_dir: str = (
        "/sc/arion/projects/pejaverlab/IGVF/users/cheny60/analysis/prior_est/"
        "data/test_pipeline/gnomAD"
    )

    mp2_pred_dir: str = (
        "/sc/arion/projects/pejaverlab/IGVF/users/cheny60/analysis/prior_est/"
        "data/test_pipeline/MutPred_precompute"
    )

    mp2_feat_dir: str = (
        "/sc/arion/projects/pejaverlab/IGVF/users/cheny60/analysis/prior_est/"
        "data/test_pipeline/MutPred_precompute"
    )


def _check_files_exist(file_map: dict[str, str]):
    """Check that all expected files exist; raise FileNotFoundError otherwise."""
    for name, path in file_map.items():
        if not os.path.exists(path):
            print(f"[X] ERROR: {name} missing: {path}")
            raise FileNotFoundError(f"Missing required file: {path}")
        else:
            print(f"[OK] {name} exists: {path}")


def prepare_gene_data(gene: str, paths: PrepareDataPaths | None = None) -> None:
    """Main entry point: prepare labeled and unlabeled data for one gene."""
    if paths is None:
        paths = PrepareDataPaths()

    base_out = os.path.join(paths.base_output_dir, gene)
    os.makedirs(base_out, exist_ok=True)

    mp2_trfn = paths.mp2_training_file
    revel_trfn = paths.revel_training_file
    scfn = paths.clinvar_scores_file

    gnvepfn = os.path.join(paths.gnomad_vepanno_dir, f"{gene}.gnomAD.v4.0.vepanno.txt")
    if not os.path.exists(gnvepfn) or os.path.getsize(gnvepfn) == 0:
        gnvepfn = os.path.join(paths.gnomad_preds_dir, f"{gene}.gnomAD.v4.0.vepPreds.txt")

    gnscfn = os.path.join(paths.gnomad_scores_dir, gene, f"{gene}.gnomAD.v4.1.0.scores.csv")
    if not os.path.exists(gnscfn):
        gnscfn = os.path.join(paths.gnomad_preds_dir, f"{gene}.gnomAD.v4.0.preds.csv")

    mp2fn = os.path.join(paths.mp2_pred_dir, f"{gene}.mutpred_preds.txt")
    mp2featfn = os.path.join(paths.mp2_feat_dir, f"{gene}.mutpred.feat.csv")

    file_list = {
        "mp2_trfn": mp2_trfn,
        "revel_trfn": revel_trfn,
        "scfn": scfn,
        "gnvepfn": gnvepfn,
        "gnscfn": gnscfn,
        "mp2fn": mp2fn,
        "mp2featfn": mp2featfn,
    }
    _check_files_exist(file_list)

    mp2_tr = pd.read_table(mp2_trfn, header=None)
    revel_tr = pd.read_csv(revel_trfn)
    sc = pd.read_csv(scfn, index_col=0)
    gnvep = pd.read_table(gnvepfn, header=None)
    gnsc = pd.read_csv(gnscfn, index_col=0)
    mp2sc = pd.read_table(mp2fn, header=None).rename(columns={0: "AA", 1: "mp2"})
    mp2feat = pd.read_csv(mp2featfn, index_col=0)

    gsc = sc[sc.GeneSymbol == gene]
    gpbsc = gsc[gsc.merg_clinvar_sig.str.contains("PLP|BLB")]
    gpbsc = gpbsc[gpbsc["AA"] == gpbsc["clnv_aa"]]
    print(f"Raw P/B ClinVar counts: {gpbsc.shape[0]}.")

    # AM labeled data
    am_lab = gpbsc[["am_pathogenicity", "merg_clinvar_sig"]].dropna(subset=["am_pathogenicity"])
    if am_lab.shape[0] == 0:
        amfn = os.path.join(base_out, f"{gene}_am_scores.txt")
        if os.path.exists(amfn):
            amscr = pd.read_table(amfn, header=None)
            am_lab = (
                gpbsc[["AA", "am_pathogenicity", "merg_clinvar_sig"]]
                .merge(amscr, left_on="AA", right_on=1, how="left")
                .dropna(subset=[2])[["merg_clinvar_sig", 2]]
            )
            am_lab = am_lab.rename(columns={2: "am_pathogenicity"})

    am_lab["merg_clinvar_sig"] = am_lab["merg_clinvar_sig"].map({"PLP": 1, "BLB": 0})
    am_lab.sort_values("am_pathogenicity").to_csv(
        os.path.join(base_out, f"{gene}_AM_labeled.txt"),
        sep="\t",
        header=False,
        index=False,
    )

    # MP2 labeled data
    gids = gsc.GeneID.unique()
    assert len(gids) == 1, f"Expected one GeneID for {gene}, got {gids}"
    gid = str(gids[0])

    tmpmp2_vars: list[list[str]] = []
    for row in mp2_tr[mp2_tr[4] == gid][1]:
        tmpmp2_vars.append(row.split(","))
    mp2_vars = set(itertools.chain.from_iterable(tmpmp2_vars))

    clean_gpbsc = gpbsc[~gpbsc.AA.isin(mp2_vars)]
    clean_gpbsc = clean_gpbsc.merge(mp2sc, on="AA", how="left").dropna(subset=["mp2"])

    clean_gpbsc_plp = clean_gpbsc[clean_gpbsc["merg_clinvar_sig"] == "PLP"]
    mp2feat_plp = mp2feat[mp2feat.index.isin(clean_gpbsc_plp.AA)]
    mp2feat_plp.to_csv(os.path.join(base_out, f"clean{gene}_PLP.mutpred.feat.csv"))

    print(f"P/B ClinVar counts without MP2 training variants: {clean_gpbsc.shape[0]}.")
    clean_gpbsc["merg_clinvar_sig"] = clean_gpbsc["merg_clinvar_sig"].map({"PLP": 1, "BLB": 0})
    clean_gpbsc.sort_values("mp2")[["mp2", "merg_clinvar_sig"]].to_csv(
        os.path.join(base_out, f"{gene}_MP2_labeled.txt"),
        sep="\t",
        header=False,
        index=False,
    )

    # REVEL labeled data
    revel_vars = revel_tr[revel_tr.gene_symbol == gene]
    clean_gpbsc_r = gpbsc[~gpbsc.AA.isin(revel_vars.protein_variant)].dropna(subset=["REVEL"])
    clean_gpbsc_r["merg_clinvar_sig"] = clean_gpbsc_r["merg_clinvar_sig"].map({"PLP": 1, "BLB": 0})
    print(f"P/B ClinVar counts without REVEL training variants: {clean_gpbsc_r.shape[0]}.")
    clean_gpbsc_r.sort_values("REVEL")[["REVEL", "merg_clinvar_sig"]].to_csv(
        os.path.join(base_out, f"{gene}_REVEL_labeled.txt"),
        sep="\t",
        header=False,
        index=False,
    )

    # gnomAD unlabeled data
    gnvep["AA"] = gnvep.apply(get_aa, axis=1)
    print(f"Before filter by AF, gnomAD variant count: {gnvep.shape[0]}.")

    pred_df = gnvep[13].apply(parse_preds).apply(pd.Series)
    gnvep = pd.concat([gnvep[["AA"]], pred_df], axis=1)
    gnvep["gnomADe_AF"] = pd.to_numeric(gnvep["gnomADe_AF"], errors="coerce")
    gnvep["gnomADg_AF"] = pd.to_numeric(gnvep["gnomADg_AF"], errors="coerce")

    mask = (
        (gnvep["gnomADe_AF"].notna() & (gnvep["gnomADe_AF"] < 0.01))
        |
        (
            gnvep["gnomADe_AF"].isna()
            & gnvep["gnomADg_AF"].notna()
            & (gnvep["gnomADg_AF"] < 0.01)
        )
        |
        (gnvep["gnomADe_AF"].isna() & gnvep["gnomADg_AF"].isna())
    )

    gnvep = gnvep[mask]
    print(f"After filter by AF, gnomAD variant count: {gnvep.shape[0]}.")

    final_gn = gnvep.merge(gnsc, on="AA", how="left")
    print(f"Before filter by SpliceAI, gnomAD variant count: {final_gn.shape[0]}.")

    if "SpliceAI_pred" in final_gn.columns:
        pred_str = final_gn["SpliceAI_pred"].fillna("0|0|0|0|0")
        splice_values = pred_str.str.split("|", expand=True)
        final_gn[["DS_AG", "DS_AL", "DS_DG", "DS_DL"]] = splice_values.iloc[:, 1:5].astype(float)
        splice_mask = (final_gn[["DS_AG", "DS_AL", "DS_DG", "DS_DL"]] < 0.2).any(axis=1)
        missing_mask = final_gn["SpliceAI_pred"].isna()
        final_gn = final_gn[splice_mask | missing_mask]
    else:
        final_gn = final_gn[
            (final_gn[["DS_AG", "DS_AL", "DS_DG", "DS_DL"]] < 0.2).any(axis=1)
        ]
    print(f"After filter by SpliceAI, gnomAD variant count: {final_gn.shape[0]}.")

    # AM unlabeled data
    CANDIDATES = ["am_patho", "am_pathogenicity", "AM"]
    present = [c for c in CANDIDATES if c in final_gn.columns]
    score_col = present[0] if present else None

    out_am = os.path.join(base_out, f"{gene}_AM_unlabeled.txt")
    if score_col is not None and final_gn.dropna(subset=[score_col]).shape[0] == 0:
        amfn = os.path.join(base_out, f"{gene}_am_scores.txt")
        if os.path.exists(amfn):
            amscr = pd.read_table(amfn, header=None)
            am_unlab = (
                final_gn[["AA", score_col]]
                .merge(amscr, left_on="AA", right_on=1, how="left")
                .dropna(subset=[2])[2]
            )
            am_unlab.to_csv(out_am, sep="\t", header=False, index=False)
    else:
        if "am_patho" in final_gn.columns:
            final_gn.dropna(subset=["am_patho"])["am_patho"].to_csv(
                out_am, sep="\t", header=False, index=False
            )
        elif "am_pathogenicity" in final_gn.columns:
            final_gn.dropna(subset=["am_pathogenicity"])["am_pathogenicity"].to_csv(
                out_am, sep="\t", header=False, index=False
            )

    # MP2 unlabeled data
    clean_gn_mp2 = final_gn[~final_gn.AA.isin(mp2_vars)]
    clean_gn_mp2 = clean_gn_mp2.merge(mp2sc, on="AA", how="left").dropna(subset=["mp2"])
    print(f"After removing MP2 training variants, gnomAD variant count: {clean_gn_mp2.shape[0]}.")
    clean_gn_mp2["mp2"].to_csv(
        os.path.join(base_out, f"{gene}_MP2_unlabeled.txt"),
        sep="\t",
        header=False,
        index=False,
    )
    mp2feat_gn = mp2feat[mp2feat.index.isin(clean_gn_mp2.AA)]
    mp2feat_gn.to_csv(os.path.join(base_out, f"clean{gene}.gnomAD.v4.0.mp2feat.csv"))

    # REVEL unlabeled data
    clean_gn_rev = final_gn[~final_gn.AA.isin(revel_vars.protein_variant)].dropna(
        subset=["REVEL"]
    )
    print(f"After removing REVEL training variants, gnomAD variant count: {clean_gn_rev.shape[0]}.")
    clean_gn_rev["REVEL"].to_csv(
        os.path.join(base_out, f"{gene}_REVEL_unlabeled.txt"),
        sep="\t",
        header=False,
        index=False,
    )


def main():
    """CLI wrapper to mimic: python 00.prepare_data.py GENE"""
    if len(sys.argv) != 2:
        sys.exit("Usage: python -m calib_pipeline.prepare_data GENE")
    gene = sys.argv[1]
    prepare_gene_data(gene)


if __name__ == "__main__":
    main()
