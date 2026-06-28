"""Reuse post-PCA checkpoint. Fix Harmony shape issue, time both methods, scVI heuristic."""
import time
from pathlib import Path
import numpy as np
import scanpy as sc

WORK = Path("/shared/projects/tp_2630_ubordeaux_neuromics_184418/projects/C10/lederer/gbm_space_proj/scratch_build")
adata = sc.read_h5ad(WORK / "checkpoint_postpca.h5ad")
print("loaded checkpoint:", adata.shape, "obsm:", list(adata.obsm.keys()), flush=True)

# ---- Harmony via harmonypy directly (robust to return-shape quirk) ----
import harmonypy
print("\n=== HARMONY (direct harmonypy) ===", flush=True)
tH = time.time()
ho = harmonypy.run_harmony(adata.obsm["X_pca"], adata.obs, ["donor_id"], max_iter_harmony=20)
Z = ho.Z_corr
print("raw Z_corr type:", type(Z), "shape attr:", getattr(Z, "shape", "n/a"), flush=True)
Zc = np.asarray(Z)
print("np.asarray shape:", Zc.shape, "dtype:", Zc.dtype, flush=True)
# Z_corr is (n_pcs, n_cells); we want (n_cells, n_pcs)
Zt = Zc.T
print("transposed shape:", Zt.shape, flush=True)
adata.obsm["X_pca_harmony"] = Zt
print(f"Harmony OK. wall-clock {time.time()-tH:.1f}s, embedding {adata.obsm['X_pca_harmony'].shape}", flush=True)

# ---- scVI heuristic + per-epoch ----
print("\n=== scVI ===", flush=True)
import scvi, torch
torch.set_num_threads(8)
print("scvi", scvi.__version__, "torch threads", torch.get_num_threads(), flush=True)
from scvi.model import SCVI
heur = None
try:
    from scvi.model._utils import get_max_epochs_heuristic
    heur = get_max_epochs_heuristic(adata.n_obs)
except Exception as e:
    print("heuristic import err:", e, flush=True)
print("get_max_epochs_heuristic ->", heur, flush=True)

scvi_ad = adata.copy()
SCVI.setup_anndata(scvi_ad, layer="counts", batch_key="donor_id")
model = SCVI(scvi_ad, n_latent=30)
tP = time.time()
model.train(max_epochs=4, early_stopping=False)
per_epoch = (time.time()-tP)/4
print(f"scVI per-epoch (4-epoch probe): {per_epoch:.1f}s/epoch", flush=True)
print(f"  heuristic {heur} epochs ~ {per_epoch*(heur or 0)/60:.1f} min", flush=True)
print(f"  40 epochs ~ {per_epoch*40/60:.1f} min | 50 ~ {per_epoch*50/60:.1f} min", flush=True)

print("\nDONE", flush=True)
