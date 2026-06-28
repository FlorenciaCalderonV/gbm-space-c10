"""Inspect the student h5ad + answer key to ground notebook parameters in reality."""
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

STUDENT = "/shared/projects/tp_2630_ubordeaux_neuromics_184418/projects/C10/data/snRNA_seq/level1_prepared/gbm_l1_snrna_AT10_AT14_raw.h5ad"
KEY = "/shared/projects/tp_2630_ubordeaux_neuromics_184418/projects/C10/lederer/answer_keys/snrna_AT10_AT14_answer_key.parquet"

print("=" * 70)
print("STUDENT FILE")
print("=" * 70)
adata = sc.read_h5ad(STUDENT)
print(adata)
print("\nShape:", adata.shape)
print("\n.obs columns:", list(adata.obs.columns))
print("\n.obs head:\n", adata.obs.head())
print("\n.var columns:", list(adata.var.columns))
print("\n.var head:\n", adata.var.head())
print("\ndonor_id:\n", adata.obs["donor_id"].value_counts())
print("\nsite_id:\n", adata.obs["site_id"].value_counts())
print("\nsample:\n", adata.obs["sample"].value_counts())

# Confirm X is raw integer counts
X = adata.X
print("\n.X dtype:", X.dtype, "type:", type(X))
sub = X[:200].toarray() if hasattr(X, "toarray") else np.asarray(X[:200])
print(".X[:200] min/max:", sub.min(), sub.max())
print(".X all integer?:", np.allclose(sub, np.round(sub)))
print(".X max value (sampled rows):", sub.max())

# .raw check
print("\n.raw is not None:", adata.raw is not None)
if adata.raw is not None:
    rsub = adata.raw.X[:50].toarray() if hasattr(adata.raw.X, "toarray") else np.asarray(adata.raw.X[:50])
    print(".raw.X[:50] integer?:", np.allclose(rsub, np.round(rsub)), "max:", rsub.max())
    print(".raw shape:", adata.raw.shape)

# .obsm / layers / uns
print("\n.obsm keys:", list(adata.obsm.keys()))
print(".layers keys:", list(adata.layers.keys()))
print(".uns keys:", list(adata.uns.keys()))

# MT genes
mt_mask = adata.var_names.str.startswith("MT-")
print("\nMT- genes count:", mt_mask.sum())
print("MT- gene names:", list(adata.var_names[mt_mask])[:20])
if "mt" in adata.var.columns:
    print("var['mt'] exists, sum:", adata.var["mt"].sum())

# Existing QC col stats
for c in ["n_genes_by_counts", "total_counts", "mt_frac", "doublet_scores"]:
    if c in adata.obs.columns:
        print(f"\n{c} describe:\n", adata.obs[c].describe())

# QC metrics computed fresh
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
print("\n--- Fresh QC quantiles (whole dataset) ---")
for c in ["total_counts", "n_genes_by_counts", "pct_counts_mt"]:
    q = adata.obs[c].quantile([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    print(f"\n{c}:\n", q.round(3))

# How many cells pass paper thresholds
keep = (adata.obs["n_genes_by_counts"] >= 500) & (adata.obs["total_counts"] >= 1000) & (adata.obs["pct_counts_mt"] <= 10)
print("\nCells passing (genes>=500 & UMI>=1000 & mt<=10%):", int(keep.sum()), "of", adata.n_obs)
print("Per donor passing:\n", adata.obs.loc[keep, "donor_id"].value_counts())

# Doublet scores threshold exploration
if "doublet_scores" in adata.obs.columns:
    ds = adata.obs["doublet_scores"]
    for thr in [0.2, 0.25, 0.3, 0.4]:
        print(f"doublet_scores > {thr}: {int((ds > thr).sum())}")

print("\n" + "=" * 70)
print("ANSWER KEY (instructor validation only)")
print("=" * 70)
key = pd.read_parquet(KEY)
print("Shape:", key.shape)
print("Columns:", list(key.columns))
print("\ncell_status:\n", key["cell_status"].value_counts())
print("\nannotation_coarse:\n", key["annotation_coarse"].value_counts())
print("\nN unique annotation_coarse:", key["annotation_coarse"].nunique())
print("\nannotation_granular n unique:", key["annotation_granular"].nunique())
print("\nneftel:\n", key["neftel"].value_counts() if "neftel" in key else "n/a")
print("\ncelltypist (top 15):\n", key["celltypist"].value_counts().head(15) if "celltypist" in key else "n/a")
print("\nphase:\n", key["phase"].value_counts() if "phase" in key else "n/a")
for c in ["CNV_signal_mean", "cnv_corr"]:
    if c in key:
        print(f"\n{c} describe:\n", key[c].describe())
# index alignment
print("\nIndex matches obs_names?:", key.index.equals(adata.obs_names) if len(key)==adata.n_obs else f"len mismatch {len(key)} vs {adata.n_obs}")
print("Index intersection size:", len(key.index.intersection(adata.obs_names)))
