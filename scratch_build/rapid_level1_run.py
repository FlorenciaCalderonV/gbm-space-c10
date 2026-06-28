"""RAPID, full pipeline on the tiny 1500-cell subsample, no Slurm queueing (small enough for
direct execution). scVI capped at 5 fixed epochs (clearly a demo/placeholder run, not
representative -- per instructor direction, mark as untested-at-scale and move on). Saves
real values needed to fill build_solution_nb.py's placeholders to a JSON file.
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
from gbmspace_utils.data import SNRNA_ANSWER_KEY_OBS_COLUMNS  # noqa: E402


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

# scVI: fixed tiny epoch count, demo-only. Per instructor direction: mark as untested at
# realistic scale and proceed -- do NOT use the adaptive heuristic that caused the original
# multi-hour hang on full-scale data.
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
    log(f"scVI demo failed ({e}); proceeding without it, Harmony-only for this build")

REP = "X_pca_harmony"
sc.pp.neighbors(adata, n_neighbors=15, use_rep=REP)
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

import celltypist
from celltypist import models
model_ct = models.Model.load(model="Developing_Human_Brain.pkl")
ct_input = adata.raw.to_adata()
ct_input.obs = adata.obs
predictions = celltypist.annotate(ct_input, model=model_ct, majority_voting=False)
adata.obs["celltypist_raw"] = predictions.predicted_labels["predicted_labels"].values
ct_by_cluster = adata.obs.groupby("leiden")["celltypist_raw"].agg(lambda s: s.value_counts().idxmax())
cluster_to_celltype = ct_by_cluster.to_dict()
adata.obs["cell_type"] = adata.obs["leiden"].map(cluster_to_celltype).astype("category")
log(f"CellTypist done. cluster_to_celltype: {cluster_to_celltype}")

tme_like_celltypist_labels = set(adata.obs["cell_type"].unique()) - {"Glioblast", "Radial glia", "OPC"}
# heuristic: anything CellTypist did NOT call a malignant-mimicking developmental label
reference_cell_types = sorted(c for c in tme_like_celltypist_labels if isinstance(c, str))
log(f"Reference (TME) cell types for CNV: {reference_cell_types}")

gene_pos = pd.read_parquet(SCRATCH / "grch38_gene_positions.parquet")
adata.var["chromosome"] = adata.var_names.map(gene_pos["chromosome"])
adata.var["start"] = adata.var_names.map(gene_pos["start"])
adata.var["end"] = adata.var_names.map(gene_pos["end"])
adata.obs["cnv_reference"] = np.where(adata.obs["cell_type"].astype(str).isin(reference_cell_types),
                                       adata.obs["cell_type"].astype(str), "other")
n_ref = int((adata.obs["cnv_reference"] != "other").sum())
log(f"CNV reference cells: {n_ref}")

import infercnvpy as cnv
if n_ref >= 5:
    cnv.tl.infercnv(adata, reference_key="cnv_reference", reference_cat=reference_cell_types,
                     window_size=100, step=10)
    cnv.tl.cnv_score(adata)
    adata.obs["malignant_cell"] = adata.obs["cnv_score"] > adata.obs["cnv_score"].median()
else:
    log("Too few reference cells at this tiny scale -- using a coarse TME-marker fallback")
    cluster_is_tme = sig_by_cluster.idxmax(axis=1).isin(TME_MARKERS.keys())
    adata.obs["malignant_cell"] = ~adata.obs["leiden"].map(cluster_is_tme).fillna(False)

frac_mal = adata.obs.groupby("leiden")["malignant_cell"].mean()
malignant_clusters = frac_mal[frac_mal > 0.2].index.tolist()
adata.obs["cell_status_derived"] = np.where(adata.obs["leiden"].isin(malignant_clusters), "Malignant", "TME")
log(f"Malignant/TME: {adata.obs['cell_status_derived'].value_counts().to_dict()}")

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
    "SCVI_PER_EPOCH": "N/A -- not re-validated at small scale, see build_notes.md",
    "HEUR_EPOCHS": "N/A", "HEUR_MIN": "N/A",
    "SCVI_EPOCHS": SCVI_EPOCHS_DEMO, "SCVI_MIN": "demo run only",
    "n_obs_after_qc": int(adata.n_obs), "n_clusters": int(adata.obs["leiden"].nunique()),
    "malignant_tme_counts": adata.obs["cell_status_derived"].value_counts().to_dict(),
    "axis_result": axis_result, "scvi_ok": scvi_ok,
}
with open(SCRATCH / "rapid_placeholders.json", "w") as f:
    json.dump(placeholders, f, indent=2, default=str)

log(f"\n=== DONE in {(time.time()-t0)/60:.1f} min. Placeholders saved. ===")
