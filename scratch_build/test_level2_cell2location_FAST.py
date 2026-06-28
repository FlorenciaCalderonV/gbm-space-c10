"""Drastically reduced cell2location mechanics test -- fewer cells, fewer categories, far
fewer epochs. Goal: prove the exact code path (TASK 5.1-5.4 in build_solution_nb2.py) runs
without errors and produces sane shapes, in a couple of minutes, not an hour+. Not meant to
produce biologically meaningful convergence -- just a fast correctness/mechanics check.
"""
import sys
import time
from pathlib import Path

import numpy as np
import scanpy as sc

ROOT = Path("/shared/projects/tp_2630_ubordeaux_neuromics_184418/projects/C10/lederer/gbm_space_proj")
SCRATCH = ROOT / "scratch_build"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("Loading + subsampling reference to 4000 cells")
    full = sc.read_h5ad(SCRATCH / "checkpoint_postpca.h5ad")
    np.random.seed(0)
    idx = np.random.choice(full.n_obs, size=4000, replace=False)
    adata_ref = full[idx].copy()
    del full

    sc.pp.neighbors(adata_ref, n_neighbors=15, n_pcs=30)
    sc.tl.leiden(adata_ref, resolution=0.2, flavor="igraph", n_iterations=2, key_added="cell_type")
    log(f"Stand-in cell_type: {adata_ref.obs['cell_type'].nunique()} categories, "
        f"counts: {adata_ref.obs['cell_type'].value_counts().to_dict()}")
    if "counts" not in adata_ref.layers:
        adata_ref.layers["counts"] = adata_ref.raw.X.copy() if adata_ref.raw is not None else adata_ref.X.copy()

    log("Loading Visium target (real, full spot count -- spots aren't the bottleneck)")
    adata = sc.read_h5ad(ROOT.parent.parent / "data" / "visium" / "level2_prepared" / "AT10-BRA-5-FO-1_2_student.h5ad")
    log(f"{adata.n_obs} spots x {adata.n_vars} genes")

    REF_EPOCHS, MAP_EPOCHS = 5, 20
    log(f"FAST TEST mode: reference {REF_EPOCHS} epochs, mapping {MAP_EPOCHS} epochs")

    from cell2location.utils.filtering import filter_genes
    from cell2location.models import RegressionModel, Cell2location

    shared = sorted(set(adata_ref.var_names) & set(adata.var_names))
    ref = adata_ref[:, shared].copy()
    vis = adata.copy()[:, shared].copy()
    log(f"Shared genes: {len(shared)}")

    selected = filter_genes(ref, cell_count_cutoff=15, cell_percentage_cutoff2=0.05, nonz_mean_cutoff=1.12)
    ref = ref[:, selected].copy()
    vis = vis[:, [g for g in selected if g in vis.var_names]].copy()
    log(f"Genes after filtering: ref {ref.shape}, vis {vis.shape}")

    t0 = time.time()
    RegressionModel.setup_anndata(ref, layer="counts", batch_key="donor_id", labels_key="cell_type")
    ref_model = RegressionModel(ref)
    ref_model.train(max_epochs=REF_EPOCHS, batch_size=4000)
    ref = ref_model.export_posterior(ref, sample_kwargs={"num_samples": 50, "batch_size": 4000})
    inf_aver = ref.varm["q05_per_cluster_mu_fg"]
    log(f"Reference signature done in {time.time()-t0:.1f}s -> {inf_aver.shape} "
        f"({(time.time()-t0)/REF_EPOCHS:.1f}s/epoch)")

    t0 = time.time()
    vis = vis[:, [g for g in inf_aver.index if g in vis.var_names]].copy()
    inf_aver_aligned = inf_aver.loc[vis.var_names]
    # vis was loaded fresh from the student Visium file and never normalized, so .X is
    # already raw counts here -- no layer="counts" needed (unlike `ref`, which came from a
    # checkpoint where .X had already been log-normalized).
    Cell2location.setup_anndata(vis, batch_key="sample_name" if "sample_name" in vis.obs else None)
    sp_model = Cell2location(vis, cell_state_df=inf_aver_aligned, N_cells_per_location=30, detection_alpha=200)
    sp_model.train(max_epochs=MAP_EPOCHS, batch_size=vis.n_obs)
    vis = sp_model.export_posterior(vis, sample_kwargs={"num_samples": 50, "batch_size": vis.n_obs})
    log(f"Spatial mapping done in {time.time()-t0:.1f}s ({(time.time()-t0)/MAP_EPOCHS:.1f}s/epoch)")

    abundance = vis.obsm.get("q05_cell_abundance_w_sf")
    if abundance is None:
        cols = [c for c in vis.obs.columns if c.startswith("q05")]
        abundance = vis.obs[cols]
        log("NOTE: abundance landed in .obs, not .obsm (anndata fallback path) -- still works, just a different location.")
    log(f"Abundance per spot: {abundance.shape}")
    log(abundance.describe().T[["mean", "std", "max"]].round(2).head(8).to_string())

    log("\n[SUCCESS] Full cell2location code path ran end-to-end without errors at reduced scale.")


if __name__ == "__main__":
    main()
