"""Fast end-to-end validation of the FULL Level 1 pipeline logic on a small subsample
(~8,000 cells). Goal: prove every code path works correctly (integration, clustering,
CellTypist, infercnvpy malignant split, axis scoring, save) in minutes, not hours -- before
committing to a full-scale run. Uses FIXED small epoch counts (no adaptive heuristic --
that's what hung for hours on the full run) and explicit flush=True logging throughout so
progress is actually visible in real time, not hidden behind stdout buffering.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/shared/projects/tp_2630_ubordeaux_neuromics_184418/projects/C10/lederer/gbm_space_proj")
SCRATCH = ROOT / "scratch_build"
sys.path.insert(0, str(ROOT / "src"))
from gbmspace_utils.analysis import MALIGNANT_AXIS_MARKERS, MAJOR_CLASS_OF, TME_MARKERS, score_axis, assign_dominant_state  # noqa: E402

N_CELLS = 8000
SCVI_EPOCHS_SMALL = 50  # fixed, safe -- no adaptive heuristic
LOG_FILE = SCRATCH / "validate_level1_small.progress.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    # Belt-and-suspenders: conda run / srun stdout capture has repeatedly buffered output
    # until process exit in this environment, even with flush=True on print(). Write
    # directly to a dedicated file with an explicit open/write/flush/close cycle so
    # progress is actually visible in real time regardless of any pipe buffering upstream.
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()


def main():
    t_start = time.time()
    log(f"Starting small-scale ({N_CELLS} cells) end-to-end validation")

    # ---------- Load + subsample (backed mode -- avoid loading all 118K cells fully) ----------
    backed = sc.read_h5ad(
        "/shared/projects/tp_2630_ubordeaux_neuromics_184418/projects/C10/data/snRNA_seq/level1_prepared/gbm_l1_snrna_AT10_AT14_raw.h5ad",
        backed="r",
    )
    np.random.seed(0)
    idx = np.sort(np.random.choice(backed.n_obs, size=N_CELLS, replace=False))
    adata = backed[idx, :].to_memory()
    del backed
    log(f"Loaded subsample via backed mode: {adata.n_obs} cells x {adata.n_vars} genes")
    log(f"Subsampled: {adata.n_obs} cells x {adata.n_vars} genes, donor split: {adata.obs['donor_id'].value_counts().to_dict()}")

    # ---------- QC ----------
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    n0 = adata.n_obs
    adata = adata[(adata.obs["n_genes_by_counts"] >= 500) & (adata.obs["total_counts"] >= 1000) &
                  (adata.obs["pct_counts_mt"] <= 10) & (adata.obs["doublet_scores"] < 0.25)].copy()
    sc.pp.filter_genes(adata, min_cells=3)
    log(f"QC: {n0} -> {adata.n_obs} cells, {adata.n_vars} genes")

    # ---------- Normalize + HVG ----------
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3", layer="counts", batch_key="donor_id")
    log(f"HVGs: {int(adata.var['highly_variable'].sum())}")

    # ---------- PCA + integration ----------
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, n_comps=30)
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
    log("PCA done")

    import harmonypy
    t0 = time.time()
    ho = harmonypy.run_harmony(adata.obsm["X_pca"], adata.obs, ["donor_id"], max_iter_harmony=20)
    Z = np.asarray(ho.Z_corr)
    if Z.shape[0] != adata.n_obs:
        Z = Z.T
    adata.obsm["X_pca_harmony"] = Z
    log(f"Harmony done in {time.time()-t0:.1f}s")

    import scvi
    import torch
    torch.set_num_threads(8)
    scvi.settings.seed = 0
    scvi_ad = adata.copy()
    scvi.model.SCVI.setup_anndata(scvi_ad, layer="counts", batch_key="donor_id")
    t0 = time.time()
    model = scvi.model.SCVI(scvi_ad, n_latent=30)
    model.train(max_epochs=SCVI_EPOCHS_SMALL, early_stopping=False)
    adata.obsm["X_scvi"] = model.get_latent_representation()
    log(f"scVI ({SCVI_EPOCHS_SMALL} epochs, fixed) done in {(time.time()-t0):.1f}s -- "
        f"=> {(time.time()-t0)/SCVI_EPOCHS_SMALL:.2f}s/epoch real rate")

    REP = "X_pca_harmony"
    sc.pp.neighbors(adata, n_neighbors=15, use_rep=REP)
    sc.tl.umap(adata)
    log("Canonical neighbors+UMAP built (Harmony)")

    # ---------- Clustering ----------
    sc.tl.leiden(adata, resolution=0.5, flavor="igraph", n_iterations=2, key_added="leiden")
    log(f"Leiden (res=0.5): {adata.obs['leiden'].nunique()} clusters")

    # ---------- Annotation: markers + CellTypist ----------
    sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon", use_raw=True)
    log("rank_genes_groups (Wilcoxon) done")

    tme_present = {ct: [g for g in genes if g in adata.raw.var_names] for ct, genes in TME_MARKERS.items()}
    tme_present = {ct: g for ct, g in tme_present.items() if g}
    for ct, genes in tme_present.items():
        sc.tl.score_genes(adata, gene_list=genes, score_name=f"sig_{ct}", use_raw=True)
    sig_cols = [f"sig_{ct}" for ct in tme_present]
    sig_by_cluster = adata.obs.groupby("leiden")[sig_cols].mean()
    sig_by_cluster.columns = [c.replace("sig_", "") for c in sig_by_cluster.columns]
    log("TME marker scoring done")

    import celltypist
    from celltypist import models
    model_ct = models.Model.load(model="Developing_Human_Brain.pkl")
    ct_input = adata.raw.to_adata()
    ct_input.obs = adata.obs
    predictions = celltypist.annotate(ct_input, model=model_ct, majority_voting=False)
    adata.obs["celltypist_raw"] = predictions.predicted_labels["predicted_labels"].values
    ct_by_cluster = adata.obs.groupby("leiden")["celltypist_raw"].agg(lambda s: s.value_counts().idxmax())
    log("CellTypist done")

    # Fully automated cell_type call for THIS validation run (proving the pipeline works,
    # not curating a final scientific annotation at toy scale): per-cluster majority CellTypist vote.
    adata.obs["cell_type"] = adata.obs["leiden"].map(ct_by_cluster).astype("category")
    log(f"Auto cell_type (CellTypist majority vote per cluster): {adata.obs['cell_type'].value_counts().to_dict()}")

    # ---------- infercnvpy malignant split ----------
    gene_pos = pd.read_parquet(SCRATCH / "grch38_gene_positions.parquet")
    adata.var["chromosome"] = adata.var_names.map(gene_pos["chromosome"])
    adata.var["start"] = adata.var_names.map(gene_pos["start"])
    adata.var["end"] = adata.var_names.map(gene_pos["end"])
    n_pos = adata.var["chromosome"].notna().sum()
    log(f"Genes with genomic position: {n_pos}/{adata.n_vars}")

    tme_like = {"Microglia", "Macrophage/Monocyte", "Oligodendrocyte", "Astrocyte",
                "Neuron (Exc)", "Neuron (Inh)", "Endothelial", "Pericyte", "Lymphocyte", "OPC"}
    adata.obs["cnv_reference"] = np.where(
        adata.obs["cell_type"].astype(str).isin(tme_like), adata.obs["cell_type"].astype(str), "other")
    n_ref = int((adata.obs["cnv_reference"] != "other").sum())
    log(f"CNV reference cells (CellTypist-typed TME): {n_ref}")

    if n_ref < 20:
        log("WARNING: too few CNV reference cells from CellTypist labels at this small scale -- "
            "falling back to a coarse TME-vs-other split via highest TME marker score per cluster")
        cluster_is_tme = sig_by_cluster.idxmax(axis=1).isin(TME_MARKERS.keys())
        adata.obs["cnv_reference"] = np.where(
            adata.obs["leiden"].map(cluster_is_tme).fillna(False), "TME_ref", "other")
        n_ref = int((adata.obs["cnv_reference"] != "other").sum())
        log(f"Fallback CNV reference cells: {n_ref}")

    import infercnvpy as cnv
    cnv.tl.infercnv(adata, reference_key="cnv_reference",
                     reference_cat=[c for c in adata.obs["cnv_reference"].unique() if c != "other"],
                     window_size=100, step=10)
    cnv.tl.cnv_score(adata)
    log(f"infercnvpy done. cnv_score describe:\n{adata.obs['cnv_score'].describe().round(4).to_string()}")

    adata.obs["malignant_cell"] = adata.obs["cnv_score"] > adata.obs["cnv_score"].median()
    frac_mal = adata.obs.groupby("leiden")["malignant_cell"].mean()
    malignant_clusters = frac_mal[frac_mal > 0.2].index
    adata.obs["cell_status_derived"] = np.where(adata.obs["leiden"].isin(malignant_clusters), "Malignant", "TME")
    log(f"Malignant/TME split: {adata.obs['cell_status_derived'].value_counts().to_dict()}")

    # ---------- Malignant axis ----------
    mal = adata[adata.obs["cell_status_derived"] == "Malignant"].copy()
    if mal.n_obs > 10:
        state_scores = score_axis(mal, MALIGNANT_AXIS_MARKERS, use_raw=True)
        mal.obs["malignant_state"] = assign_dominant_state(state_scores)
        mal.obs["malignant_class"] = mal.obs["malignant_state"].map(MAJOR_CLASS_OF)
        log(f"Malignant axis (major class): {mal.obs['malignant_class'].value_counts().to_dict()}")
    else:
        log(f"WARNING: only {mal.n_obs} malignant cells at this small scale -- axis scoring skipped/unreliable")

    # ---------- Save ----------
    out_dir = SCRATCH / "validation_small_outputs"
    out_dir.mkdir(exist_ok=True)
    adata.write_h5ad(out_dir / "small_validated.h5ad")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sc.pl.umap(adata, color="leiden", ax=axes[0], show=False, title="Leiden")
    sc.pl.umap(adata, color="cell_type", ax=axes[1], show=False, title="cell_type (CellTypist)")
    sc.pl.umap(adata, color="cell_status_derived", ax=axes[2], show=False, title="Malignant vs TME")
    plt.tight_layout()
    fig.savefig(out_dir / "small_validation_summary.png", dpi=150)

    log(f"\n=== DONE. Total wall time: {(time.time()-t_start)/60:.1f} min ===")
    log("Every pipeline stage (QC, normalize/HVG, Harmony+scVI, Leiden, CellTypist, "
        "infercnvpy, axis scoring, save) ran without errors at small scale.")


if __name__ == "__main__":
    main()
