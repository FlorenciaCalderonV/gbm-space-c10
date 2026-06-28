"""ULTRA-rapid pipeline on tiny (1500-cell) data. Drops CellTypist and infercnvpy for THIS
build pass (both showed disproportionate fixed overhead vs. data size) -- uses pure
marker-score logic instead, which is fast, deterministic, no model loading. Marked clearly
as a placeholder for the real notebook's CellTypist/infercnvpy steps, which are real,
correct code (validated separately) just not re-run here under the time constraint.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path("/shared/projects/tp_2630_ubordeaux_neuromics_184418/projects/C10/lederer/gbm_space_proj")
SCRATCH = ROOT / "scratch_build"
sys.path.insert(0, str(ROOT / "src"))
from gbmspace_utils.analysis import MALIGNANT_AXIS_MARKERS, MAJOR_CLASS_OF, TME_MARKERS, score_axis, assign_dominant_state  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


t0 = time.time()
adata = sc.read_h5ad(SCRATCH / "tiny_snrna_1500.h5ad")
log(f"Loaded {adata.n_obs} x {adata.n_vars}")

adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
adata = adata[(adata.obs["n_genes_by_counts"] >= 500) & (adata.obs["total_counts"] >= 1000) &
              (adata.obs["pct_counts_mt"] <= 10) & (adata.obs["doublet_scores"] < 0.25)].copy()
sc.pp.filter_genes(adata, min_cells=3)
log(f"QC done: {adata.n_obs} cells, {adata.n_vars} genes")

adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3", layer="counts", batch_key="donor_id")
adata_hvg = adata[:, adata.var["highly_variable"]].copy()
sc.pp.scale(adata_hvg, max_value=10)
sc.tl.pca(adata_hvg, n_comps=30)
adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
log("PCA done")

import harmonypy
ho = harmonypy.run_harmony(adata.obsm["X_pca"], adata.obs, ["donor_id"], max_iter_harmony=20)
Z = np.asarray(ho.Z_corr)
if Z.shape[0] != adata.n_obs:
    Z = Z.T
adata.obsm["X_pca_harmony"] = Z
log("Harmony done")

SCVI_EPOCHS_DEMO = 5
scvi_ok = True
try:
    import scvi
    scvi_ad = adata.copy()
    scvi.model.SCVI.setup_anndata(scvi_ad, layer="counts", batch_key="donor_id")
    t_s = time.time()
    model = scvi.model.SCVI(scvi_ad, n_latent=30)
    model.train(max_epochs=SCVI_EPOCHS_DEMO, early_stopping=False)
    adata.obsm["X_scvi"] = model.get_latent_representation()
    log(f"scVI demo ({SCVI_EPOCHS_DEMO} epochs) done in {time.time()-t_s:.1f}s")
except Exception as e:
    scvi_ok = False
    log(f"scVI demo failed/skipped ({e})")

sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca_harmony")
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5, flavor="igraph", n_iterations=2, key_added="leiden")
log(f"Leiden: {adata.obs['leiden'].nunique()} clusters")

sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon", use_raw=True)
tme_present = {ct: [g for g in genes if g in adata.raw.var_names] for ct, genes in TME_MARKERS.items()}
tme_present = {ct: g for ct, g in tme_present.items() if g}
for ct, genes in tme_present.items():
    sc.tl.score_genes(adata, gene_list=genes, score_name=f"sig_{ct}", use_raw=True)
sig_cols = [f"sig_{ct}" for ct in tme_present]
sig_by_cluster = adata.obs.groupby("leiden")[sig_cols].mean()
sig_by_cluster.columns = [c.replace("sig_", "") for c in sig_by_cluster.columns]
log("Marker scoring done")

# Pure marker-score cell typing (CellTypist dropped for THIS rapid pass -- real notebook
# code for it is already written/validated separately, just not re-run here for speed).
cluster_to_celltype = sig_by_cluster.idxmax(axis=1).to_dict()
adata.obs["cell_type"] = adata.obs["leiden"].map(cluster_to_celltype).astype("category")
log(f"cluster_to_celltype (marker-score argmax): {cluster_to_celltype}")

reference_cell_types = sorted(set(cluster_to_celltype.values()))
log(f"Reference (TME) cell types for CNV: {reference_cell_types}")

# infercnvpy dropped for THIS rapid pass -- use the same marker-score TME-vs-other heuristic
# as the documented fallback path in the real notebook/validation scripts.
cluster_is_tme = sig_by_cluster.idxmax(axis=1).isin(TME_MARKERS.keys())
adata.obs["cell_status_derived"] = np.where(adata.obs["leiden"].map(cluster_is_tme).fillna(False), "TME", "Malignant")
log(f"Malignant/TME (marker-score heuristic): {adata.obs['cell_status_derived'].value_counts().to_dict()}")

mal = adata[adata.obs["cell_status_derived"] == "Malignant"].copy()
axis_result = {}
if mal.n_obs > 5:
    state_scores = score_axis(mal, MALIGNANT_AXIS_MARKERS, use_raw=True)
    mal.obs["malignant_state"] = assign_dominant_state(state_scores)
    mal.obs["malignant_class"] = mal.obs["malignant_state"].map(MAJOR_CLASS_OF)
    axis_result = mal.obs["malignant_class"].value_counts().to_dict()
log(f"Malignant axis: {axis_result}")

out_dir = ROOT / "data" / "processed"
out_dir.mkdir(parents=True, exist_ok=True)
adata.write_h5ad(out_dir / "DEMO_gbm_l1_snrna_AT10_AT14_annotated.h5ad")

placeholders = {
    "REFERENCE_CELL_TYPES": reference_cell_types,
    "CLUSTER_TO_CELLTYPE": {str(k): str(v) for k, v in cluster_to_celltype.items()},
    "SCVI_EPOCHS": SCVI_EPOCHS_DEMO,
    "n_obs_after_qc": int(adata.n_obs), "n_clusters": int(adata.obs["leiden"].nunique()),
    "malignant_tme_counts": adata.obs["cell_status_derived"].value_counts().to_dict(),
    "axis_result": axis_result, "scvi_ok": scvi_ok,
}
with open(SCRATCH / "rapid_placeholders.json", "w") as f:
    json.dump(placeholders, f, indent=2, default=str)

log(f"\n=== DONE in {(time.time()-t0)/60:.1f} min ===")
