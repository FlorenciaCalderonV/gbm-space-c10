"""Continue the Level 1 pipeline from checkpoint_postpca.h5ad through to the final
annotated save. Reuses the exact logic already drafted in build_solution_nb.py (sections
4-11), run here as a plain script first so we get REAL numbers/decisions to substitute
back into the notebook-build placeholders. Checkpoints after each expensive stage.
Run via srun, not the login node.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")

ROOT = Path("/shared/projects/tp_2630_ubordeaux_neuromics_184418/projects/C10/lederer/gbm_space_proj")
SCRATCH = ROOT / "scratch_build"
sys.path.insert(0, str(ROOT / "src"))
from gbmspace_utils.analysis import MALIGNANT_AXIS_MARKERS, MAJOR_CLASS_OF, TME_MARKERS, score_axis, assign_dominant_state  # noqa: E402

sc.settings.verbosity = 1


def banner(msg):
    print(f"\n{'='*70}\n{msg}\n{'='*70}", flush=True)


def main():
    t_start = time.time()

    # ---------- Stage A: integration ----------
    ckpt_a = SCRATCH / "checkpoint_A_integrated.h5ad"
    if ckpt_a.exists():
        banner("Stage A: loading existing integration checkpoint")
        adata = sc.read_h5ad(ckpt_a)
    else:
        banner("Stage A: integration (Harmony + scVI)")
        adata = sc.read_h5ad(SCRATCH / "checkpoint_postpca.h5ad")
        print(f"Loaded: {adata.shape}")

        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30, key_added="neighbors_uncorr")
        sc.tl.umap(adata, neighbors_key="neighbors_uncorr")
        adata.obsm["X_umap_uncorr"] = adata.obsm["X_umap"].copy()

        import harmonypy
        t0 = time.time()
        ho = harmonypy.run_harmony(adata.obsm["X_pca"], adata.obs, ["donor_id"], max_iter_harmony=20)
        Z = np.asarray(ho.Z_corr)
        if Z.shape[0] != adata.n_obs:
            Z = Z.T
        adata.obsm["X_pca_harmony"] = Z
        t_harmony = time.time() - t0
        print(f"[TIME] Harmony: {t_harmony:.1f}s -> {adata.obsm['X_pca_harmony'].shape}")

        import scvi
        import torch
        torch.set_num_threads(8)
        scvi.settings.seed = 0

        scvi_ad = adata.copy()
        scvi.model.SCVI.setup_anndata(scvi_ad, layer="counts", batch_key="donor_id")

        try:
            from scvi.model._utils import get_max_epochs_heuristic
            heuristic_epochs = get_max_epochs_heuristic(scvi_ad.n_obs)
        except Exception as e:
            print(f"(heuristic lookup failed: {e}, using 400 as reference default)")
            heuristic_epochs = 400
        print(f"[INFO] scvi-tools epoch heuristic for {scvi_ad.n_obs:,} cells: {heuristic_epochs}")

        PROBE_EPOCHS = 3
        t0 = time.time()
        probe_model = scvi.model.SCVI(scvi_ad, n_latent=30)
        probe_model.train(max_epochs=PROBE_EPOCHS, early_stopping=False)
        t_probe = time.time() - t0
        per_epoch = t_probe / PROBE_EPOCHS
        heur_min = heuristic_epochs * per_epoch / 60
        print(f"[TIME] scVI probe: {PROBE_EPOCHS} epochs in {t_probe:.1f}s -> {per_epoch:.1f}s/epoch")
        print(f"[INFO] Heuristic {heuristic_epochs} epochs would take ~{heur_min:.1f} min")

        # Cap so the FULL scVI step (probe already spent + remaining) stays well under an
        # hour on CPU -- target ~15 min remaining budget for the capped run.
        target_minutes = 15
        SCVI_EPOCHS = max(10, min(heuristic_epochs, int(target_minutes * 60 / per_epoch)))
        print(f"[DECISION] SCVI_MAX_EPOCHS = {SCVI_EPOCHS} (targeting ~{target_minutes} min)")

        t0 = time.time()
        model = scvi.model.SCVI(scvi_ad, n_latent=30)
        model.train(max_epochs=SCVI_EPOCHS, early_stopping=False)
        t_scvi = time.time() - t0
        scvi_min = t_scvi / 60
        print(f"[TIME] scVI full run: {SCVI_EPOCHS} epochs in {scvi_min:.1f} min")
        adata.obsm["X_scvi"] = model.get_latent_representation()

        # Save the real numbers for notebook placeholder substitution.
        timing = dict(
            SCVI_PER_EPOCH=round(per_epoch, 1), HEUR_EPOCHS=heuristic_epochs,
            HEUR_MIN=round(heur_min, 1), SCVI_EPOCHS=SCVI_EPOCHS, SCVI_MIN=round(scvi_min, 1),
            HARMONY_SEC=round(t_harmony, 1),
        )
        pd.Series(timing).to_json(SCRATCH / "scvi_timing.json")
        print(f"[SAVED] timing -> scvi_timing.json: {timing}")

        # Canonical comparison UMAPs + neighbor-purity metric
        for basis, key in [("X_pca_harmony", "umap_harmony"), ("X_scvi", "umap_scvi")]:
            sc.pp.neighbors(adata, n_neighbors=15, use_rep=basis, key_added=f"nn_{key}")
            sc.tl.umap(adata, neighbors_key=f"nn_{key}")
            adata.obsm[f"X_{key}"] = adata.obsm["X_umap"].copy()

        from sklearn.neighbors import NearestNeighbors

        def same_donor_neighbor_fraction(emb, labels, k=30):
            nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)
            idx = nn.kneighbors(emb, return_distance=False)[:, 1:]
            lab = labels.to_numpy()
            return (lab[idx] == lab[:, None]).mean(axis=1)

        ideal = (adata.obs["donor_id"].value_counts(normalize=True) ** 2).sum()
        print(f"\nExpected same-donor neighbor fraction under perfect mixing: {ideal:.3f}")
        for name, basis in [("Uncorrected", "X_pca"), ("Harmony", "X_pca_harmony"), ("scVI", "X_scvi")]:
            frac = same_donor_neighbor_fraction(adata.obsm[basis], adata.obs["donor_id"]).mean()
            print(f"  {name:12s}: mean same-donor neighbor fraction = {frac:.3f}")

        INTEGRATION_METHOD = "harmony"
        REP = {"harmony": "X_pca_harmony", "scvi": "X_scvi"}[INTEGRATION_METHOD]
        sc.pp.neighbors(adata, n_neighbors=15, use_rep=REP)
        sc.tl.umap(adata)
        adata.uns["integration_method_used_for_solution"] = INTEGRATION_METHOD

        adata.write_h5ad(ckpt_a)
        print(f"[CHECKPOINT] saved {ckpt_a}")

    # ---------- Stage B: clustering + annotation ----------
    ckpt_b = SCRATCH / "checkpoint_B_annotated.h5ad"
    if ckpt_b.exists():
        banner("Stage B: loading existing annotation checkpoint")
        adata = sc.read_h5ad(ckpt_b)
    else:
        banner("Stage B: clustering + annotation")
        for res in [0.3, 0.5, 1.0]:
            sc.tl.leiden(adata, resolution=res, key_added=f"leiden_r{res}", flavor="igraph", n_iterations=2)
            print(f"resolution {res}: {adata.obs[f'leiden_r{res}'].nunique()} clusters")
        adata.obs["leiden"] = adata.obs["leiden_r0.5"]
        print(f"\nUsing resolution 0.5: {adata.obs['leiden'].nunique()} clusters")
        print(adata.obs["leiden"].value_counts().sort_index())

        sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon", use_raw=True)
        top = pd.DataFrame(adata.uns["rank_genes_groups"]["names"]).head(8)
        print("\nTop-8 marker genes per cluster:")
        print(top.to_string())

        tme_present = {ct: [g for g in genes if g in adata.raw.var_names] for ct, genes in TME_MARKERS.items()}
        tme_present = {ct: g for ct, g in tme_present.items() if g}
        for ct, genes in tme_present.items():
            sc.tl.score_genes(adata, gene_list=genes, score_name=f"sig_{ct}", use_raw=True)
        sig_cols = [f"sig_{ct}" for ct in tme_present]
        sig_by_cluster = adata.obs.groupby("leiden")[sig_cols].mean()
        sig_by_cluster.columns = [c.replace("sig_", "") for c in sig_by_cluster.columns]

        banner("Running CellTypist (Developing_Human_Brain)")
        import celltypist
        from celltypist import models
        try:
            models.download_models(model=["Developing_Human_Brain.pkl"])
        except Exception as e:
            print(f"(download_models raised {e}, assuming already cached)")
        model_ct = models.Model.load(model="Developing_Human_Brain.pkl")
        ct_input = adata.raw.to_adata()
        ct_input.obs = adata.obs
        predictions = celltypist.annotate(ct_input, model=model_ct, majority_voting=False)
        adata.obs["celltypist_raw"] = predictions.predicted_labels["predicted_labels"].values
        ct_by_cluster = adata.obs.groupby("leiden")["celltypist_raw"].agg(lambda s: s.value_counts().idxmax())
        print("\nDominant CellTypist label per cluster:")
        print(ct_by_cluster.to_string())

        rgg = adata.uns["rank_genes_groups"]["names"]
        summary = []
        for cl in sorted(adata.obs["leiden"].cat.categories, key=int):
            top5 = ", ".join([rgg[i][int(cl)] for i in range(5)])
            best_sig = sig_by_cluster.loc[cl].idxmax()
            summary.append({"cluster": cl, "n": int((adata.obs["leiden"] == cl).sum()),
                             "top_DE": top5, "best_TME_sig": best_sig, "celltypist": ct_by_cluster[cl]})
        summary_df = pd.DataFrame(summary).set_index("cluster")
        pd.set_option("display.max_colwidth", 60)
        banner("CLUSTER SUMMARY TABLE (for manual cell_type call)")
        print(summary_df.to_string())
        summary_df.to_csv(SCRATCH / "cluster_summary.csv")
        print(f"\n[SAVED] {SCRATCH / 'cluster_summary.csv'}")

        adata.write_h5ad(ckpt_b)
        print(f"[CHECKPOINT] saved {ckpt_b} -- STOPPING HERE for manual cell_type review")
        return  # stop; the cluster->celltype call needs a human look at the table first

    banner("Stage B already complete -- re-run with cluster_to_celltype filled in to continue")
    print(f"Elapsed so far: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
