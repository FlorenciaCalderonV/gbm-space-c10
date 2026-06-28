"""One-time: create a small (1500-cell) reusable subsample of the Level 1 student data,
for fast iteration while building/testing notebook logic. Not for the final deliverable."""
import numpy as np
import scanpy as sc

backed = sc.read_h5ad(
    "/shared/projects/tp_2630_ubordeaux_neuromics_184418/projects/C10/data/snRNA_seq/level1_prepared/gbm_l1_snrna_AT10_AT14_raw.h5ad",
    backed="r",
)
np.random.seed(0)
idx = np.sort(np.random.choice(backed.n_obs, size=1500, replace=False))
small = backed[idx, :].to_memory()
out = "/shared/projects/tp_2630_ubordeaux_neuromics_184418/projects/C10/lederer/gbm_space_proj/scratch_build/tiny_snrna_1500.h5ad"
small.write_h5ad(out)
print(f"Wrote {out}: {small.n_obs} cells x {small.n_vars} genes, donors: {small.obs['donor_id'].value_counts().to_dict()}")
