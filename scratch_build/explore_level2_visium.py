"""Real exploration + QC + naive clustering + axis-in-space scoring for the Level 2
AT10 Visium section -- everything that does NOT depend on Level 1's annotated output
(that dependency is only the cell2location reference step, built separately later).
Produces real numbers/figures for Level 2 notebook sections 1-4.
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
from gbmspace_utils.analysis import MALIGNANT_AXIS_MARKERS, TME_MARKERS, ZONATION_PANEL, score_axis  # noqa: E402

VISIUM_PATH = "/shared/projects/tp_2630_ubordeaux_neuromics_184418/projects/C10/data/visium/level2_prepared/AT10-BRA-5-FO-1_2_student.h5ad"


def banner(msg):
    print(f"\n{'='*70}\n{msg}\n{'='*70}", flush=True)


def main():
    banner("1. Load and explore")
    adata = sc.read_h5ad(VISIUM_PATH)
    print(adata)
    print(f"\nShape: {adata.n_obs} spots x {adata.n_vars} genes")
    print(f".obs columns: {list(adata.obs.columns)}")
    print(f".obsm keys: {list(adata.obsm.keys())}")
    print(f".uns keys: {list(adata.uns.keys())}")
    if "spatial" in adata.uns:
        lib_ids = list(adata.uns["spatial"].keys())
        print(f".uns['spatial'] library_ids: {lib_ids}")
        for lid in lib_ids:
            print(f"  {lid} keys: {list(adata.uns['spatial'][lid].keys())}")
            if "images" in adata.uns["spatial"][lid]:
                print(f"    images: {list(adata.uns['spatial'][lid]['images'].keys())}")
            if "scalefactors" in adata.uns["spatial"][lid]:
                print(f"    scalefactors: {adata.uns['spatial'][lid]['scalefactors']}")
    print(f"\n.X dtype: {adata.X.dtype}, max: {adata.X.max()}, all-integer sample: "
          f"{np.allclose(adata.X[:50].toarray(), np.round(adata.X[:50].toarray()))}")

    banner("2. Spatial QC")
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    print(adata.obs[["total_counts", "n_genes_by_counts", "pct_counts_mt"]].describe().round(1))

    n0 = adata.n_obs
    adata = adata[(adata.obs["total_counts"] >= 500) & (adata.obs["n_genes_by_counts"] >= 250)].copy()
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"\nSpots: {n0} -> {adata.n_obs} after QC (>=500 counts, >=250 genes)")
    print(f"Genes after min_cells=3: {adata.n_vars}")

    banner("3. Normalize, HVG, PCA, naive clustering (no deconvolution yet)")
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3", layer="counts")
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, n_comps=30)
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata)
    for res in [0.5, 1.0]:
        sc.tl.leiden(adata, resolution=res, key_added=f"leiden_r{res}", flavor="igraph", n_iterations=2)
        print(f"resolution {res}: {adata.obs[f'leiden_r{res}'].nunique()} naive spatial domains")
    adata.obs["leiden"] = adata.obs["leiden_r1.0"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc.pl.embedding(adata, "spatial", color="leiden", ax=axes[0], show=False, title="Naive domains (spatial)")
    axes[0].invert_yaxis()
    sc.pl.umap(adata, color="leiden", ax=axes[1], show=False, title="Naive domains (UMAP)")
    plt.tight_layout()
    fig.savefig(SCRATCH / "level2_naive_clusters.png", dpi=150)
    print(f"[SAVED FIGURE] {SCRATCH / 'level2_naive_clusters.png'}")

    banner("4. Cell-state axis IN SPACE (pre-deconvolution, bulk-mixture spots)")
    state_scores = score_axis(adata, MALIGNANT_AXIS_MARKERS, use_raw=True)
    for col in state_scores.columns:
        adata.obs[f"score_{col}"] = state_scores[col].values
    print("Per-cluster mean axis-state scores:")
    print(adata.obs.groupby("leiden")[[f"score_{c}" for c in state_scores.columns]].mean().round(3))

    zonation_present = [g for g in ZONATION_PANEL if g in adata.raw.var_names]
    print(f"\nZonation panel present: {zonation_present}")
    fig, axes = plt.subplots(1, len(zonation_present), figsize=(4 * len(zonation_present), 4))
    for ax, gene in zip(np.atleast_1d(axes), zonation_present):
        expr = adata[:, gene].X
        expr = np.asarray(expr.todense()).flatten() if hasattr(expr, "todense") else np.asarray(expr).flatten()
        coords = adata.obsm["spatial"]
        sca = ax.scatter(coords[:, 0], coords[:, 1], c=expr, cmap="Reds", s=8,
                          vmax=np.percentile(expr[expr > 0], 95) if (expr > 0).any() else None)
        ax.invert_yaxis(); ax.set_aspect("equal"); ax.set_title(gene); ax.axis("off")
        fig.colorbar(sca, ax=ax, shrink=0.7)
    plt.tight_layout()
    fig.savefig(SCRATCH / "level2_zonation_panel.png", dpi=150)
    print(f"[SAVED FIGURE] {SCRATCH / 'level2_zonation_panel.png'}")

    adata.write_h5ad(SCRATCH / "checkpoint_level2_naive.h5ad")
    print(f"\n[CHECKPOINT] saved {SCRATCH / 'checkpoint_level2_naive.h5ad'}")
    print(f"\nFinal: {adata.n_obs} spots x {adata.n_vars} genes, {adata.obs['leiden'].nunique()} naive domains")


if __name__ == "__main__":
    main()
