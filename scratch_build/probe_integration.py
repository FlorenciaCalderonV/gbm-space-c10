"""Probe: Harmony timing, scVI per-epoch timing + heuristic, infercnvpy gene-position source.
Runs the real preprocessing up through PCA, then times integration on the FULL 118K dataset.
Saves a checkpoint .h5ad of the post-PCA object so the full pipeline can reuse it.
"""
import time
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, "/shared/projects/tp_2630_ubordeaux_neuromics_184418/projects/C10/lederer/gbm_space_proj/src")

STUDENT = "/shared/projects/tp_2630_ubordeaux_neuromics_184418/projects/C10/data/snRNA_seq/level1_prepared/gbm_l1_snrna_AT10_AT14_raw.h5ad"
WORK = Path("/shared/projects/tp_2630_ubordeaux_neuromics_184418/projects/C10/lederer/gbm_space_proj/scratch_build")

t0 = time.time()
print("Loading...", flush=True)
adata = sc.read_h5ad(STUDENT)
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

# QC: data already floored at paper thresholds; apply doublet + light upper bounds
n0 = adata.n_obs
adata = adata[adata.obs["doublet_scores"] < 0.25].copy()
print(f"After doublet<0.25: {adata.n_obs} (removed {n0-adata.n_obs})", flush=True)
sc.pp.filter_genes(adata, min_cells=3)
print(f"After gene filter min_cells=3: {adata.n_vars} genes", flush=True)

adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata
sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3", layer="counts", batch_key="donor_id")
print(f"HVGs: {int(adata.var['highly_variable'].sum())}", flush=True)

adata_hvg = adata[:, adata.var["highly_variable"]].copy()
sc.pp.scale(adata_hvg, max_value=10)
sc.tl.pca(adata_hvg, n_comps=50)
adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
print(f"PCA done. t={time.time()-t0:.0f}s", flush=True)

# Save checkpoint for reuse by full pipeline
adata.write_h5ad(WORK / "checkpoint_postpca.h5ad")
print(f"Saved checkpoint. n_obs={adata.n_obs}, n_vars={adata.n_vars}", flush=True)

# ---- Harmony timing ----
print("\n=== HARMONY ===", flush=True)
import scanpy.external as sce
tH = time.time()
sce.pp.harmony_integrate(adata, "donor_id", basis="X_pca", adjusted_basis="X_pca_harmony", max_iter_harmony=20)
print(f"Harmony wall-clock: {time.time()-tH:.1f}s", flush=True)
print("X_pca_harmony shape:", adata.obsm["X_pca_harmony"].shape, flush=True)

# ---- scVI heuristic + per-epoch probe ----
print("\n=== scVI ===", flush=True)
import scvi
import torch
torch.set_num_threads(8)
print("scvi version:", scvi.__version__, "torch threads:", torch.get_num_threads(), flush=True)
try:
    from scvi.model import SCVI
    heur = None
    try:
        from scvi.model._utils import get_max_epochs_heuristic
        heur = get_max_epochs_heuristic(adata.n_obs)
    except Exception as e:
        print("heuristic import failed:", e, flush=True)
    print("get_max_epochs_heuristic ->", heur, flush=True)

    scvi_ad = adata.copy()
    SCVI.setup_anndata(scvi_ad, layer="counts", batch_key="donor_id")
    model = SCVI(scvi_ad, n_latent=30)
    tP = time.time()
    model.train(max_epochs=3, early_stopping=False)
    per_epoch = (time.time()-tP)/3
    print(f"scVI per-epoch (3-epoch probe): {per_epoch:.1f}s/epoch", flush=True)
    print(f"  -> heuristic {heur} epochs would take ~{per_epoch*(heur or 0)/60:.1f} min", flush=True)
    print(f"  -> 50 epochs ~{per_epoch*50/60:.1f} min; 100 epochs ~{per_epoch*100/60:.1f} min", flush=True)
except Exception as e:
    import traceback; traceback.print_exc()

# ---- infercnvpy gene position source check ----
print("\n=== infercnvpy gene positions ===", flush=True)
try:
    import infercnvpy as cnv
    print("infercnvpy", cnv.__version__, flush=True)
    print("has io.genomic_position_from_gtf:", hasattr(cnv.io, "genomic_position_from_gtf"), flush=True)
    # Is there a built-in human gene position table available offline?
    import pybiomart
    print("pybiomart import OK (needs internet for query)", flush=True)
except Exception as e:
    import traceback; traceback.print_exc()

print(f"\nTOTAL probe time: {time.time()-t0:.0f}s", flush=True)
