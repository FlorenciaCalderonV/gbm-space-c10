"""Validate the EXACT cell2location code path drafted in build_solution_nb2.py (TASK 5.1-5.4)
against real data, using a quick stand-in clustering (not Level 1's final cell-type names,
which aren't ready yet) as the reference label. This tests mechanics/shapes/timing now;
once Level 1 finishes we just re-point at the real annotated file with real cell_type names
-- the code itself doesn't need to change.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path("/shared/projects/tp_2630_ubordeaux_neuromics_184418/projects/C10/lederer/gbm_space_proj")
SCRATCH = ROOT / "scratch_build"


def banner(msg):
    print(f"\n{'='*70}\n{msg}\n{'='*70}", flush=True)


def main():
    banner("Quick stand-in reference: existing PCA checkpoint + fast Leiden")
    adata_ref = sc.read_h5ad(SCRATCH / "checkpoint_postpca.h5ad")
    sc.pp.neighbors(adata_ref, n_neighbors=15, n_pcs=30)
    sc.tl.leiden(adata_ref, resolution=0.3, flavor="igraph", n_iterations=2, key_added="cell_type")
    print(f"Stand-in 'cell_type' (Leiden r=0.3): {adata_ref.obs['cell_type'].nunique()} categories")
    print(adata_ref.obs["cell_type"].value_counts())
    # cell2location needs the raw-counts layer for the regression model
    if "counts" not in adata_ref.layers:
        adata_ref.layers["counts"] = adata_ref.raw.X.copy() if adata_ref.raw is not None else adata_ref.X.copy()

    banner("Load Visium target (real, student-facing)")
    adata = sc.read_h5ad(ROOT.parent.parent / "data" / "visium" / "level2_prepared" / "AT10-BRA-5-FO-1_2_student.h5ad")
    print(f"{adata.n_obs} spots x {adata.n_vars} genes")

    C2L_MODE = "FAST"
    REF_EPOCHS = {"FAST": 20, "FULL": 400}[C2L_MODE]
    MAP_EPOCHS = {"FAST": 300, "FULL": 6000}[C2L_MODE]
    print(f"Mode={C2L_MODE}: reference {REF_EPOCHS} epochs, mapping {MAP_EPOCHS} epochs")

    from cell2location.utils.filtering import filter_genes
    from cell2location.models import RegressionModel, Cell2location

    shared = sorted(set(adata_ref.var_names) & set(adata.var_names))
    ref = adata_ref[:, shared].copy()
    vis = adata.copy()[:, shared].copy()
    print(f"Shared genes: {len(shared)}")

    selected = filter_genes(ref, cell_count_cutoff=15, cell_percentage_cutoff2=0.05, nonz_mean_cutoff=1.12)
    ref = ref[:, selected].copy()
    vis = vis[:, [g for g in selected if g in vis.var_names]].copy()
    print(f"Genes after filtering: ref {ref.shape}, vis {vis.shape}")

    banner(f"TASK 5.2 -- reference signature model ({REF_EPOCHS} epochs)")
    t0 = time.time()
    RegressionModel.setup_anndata(ref, layer="counts", batch_key="donor_id", labels_key="cell_type")
    ref_model = RegressionModel(ref)
    ref_model.train(max_epochs=REF_EPOCHS, batch_size=10000)
    ref = ref_model.export_posterior(ref, sample_kwargs={"num_samples": 100, "batch_size": 10000})
    inf_aver = ref.varm["q05_per_cluster_mu_fg"]
    t_ref = time.time() - t0
    print(f"[TIME] reference signature: {t_ref/60:.1f} min -> inf_aver shape {inf_aver.shape}")

    banner(f"TASK 5.3 -- spatial mapping model ({MAP_EPOCHS} epochs)")
    t0 = time.time()
    vis = vis[:, [g for g in inf_aver.index if g in vis.var_names]].copy()
    inf_aver_aligned = inf_aver.loc[vis.var_names]
    print(f"After alignment: vis {vis.shape}, inf_aver_aligned {inf_aver_aligned.shape}")

    Cell2location.setup_anndata(vis, batch_key="sample_name" if "sample_name" in vis.obs else None)
    sp_model = Cell2location(vis, cell_state_df=inf_aver_aligned, N_cells_per_location=30, detection_alpha=200)
    sp_model.train(max_epochs=MAP_EPOCHS, batch_size=vis.n_obs)
    vis = sp_model.export_posterior(vis, sample_kwargs={"num_samples": 100, "batch_size": vis.n_obs})
    t_map = time.time() - t0
    print(f"[TIME] spatial mapping: {t_map/60:.1f} min")

    banner("TASK 5.3 (cont) -- locate the abundance output, exactly as the notebook code expects")
    print("vis.obsm keys:", list(vis.obsm.keys()))
    print("vis.obs columns with 'q05':", [c for c in vis.obs.columns if "q05" in c][:5], "...")
    if "q05_cell_abundance_w_sf" in vis.obsm:
        abundance = vis.obsm["q05_cell_abundance_w_sf"]
        print(f"FOUND in obsm: shape {abundance.shape}")
    else:
        cols = [c for c in vis.obs.columns if c.startswith("q05")]
        abundance = vis.obs[cols]
        print(f"FOUND in obs (not obsm!) -- {len(cols)} columns. NOTEBOOK CODE NEEDS A FIX for this path.")
    print(abundance.describe().T[["mean", "std", "max"]].round(2).head(10))

    print("\n[SUCCESS] Full cell2location code path (TASK 5.1-5.4 logic) ran end-to-end without errors.")


if __name__ == "__main__":
    main()
