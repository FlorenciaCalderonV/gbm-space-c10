"""Patch build_solution_nb.py's placeholders with real (tiny-scale demo) values, fix the
data path to the tiny subsample, replace the risky adaptive-scVI-heuristic cell with a
fixed small epoch count, then run the BUILD step to write the actual .ipynb file.
"""
import json
import re
from pathlib import Path

SCRATCH = Path("/shared/projects/tp_2630_ubordeaux_neuromics_184418/projects/C10/lederer/gbm_space_proj/scratch_build")
SRC = SCRATCH / "build_solution_nb.py"
PATCHED = SCRATCH / "build_solution_nb_PATCHED.py"

placeholders = json.loads((SCRATCH / "rapid_placeholders.json").read_text())
text = SRC.read_text()

# 1. Point the data-loading cell at the tiny demo subsample instead of the full file.
text = text.replace(
    '"/shared/projects/tp_2630_ubordeaux_neuromics_184418/projects/C10/data/snRNA_seq/level1_prepared/gbm_l1_snrna_AT10_AT14_raw.h5ad"',
    '"/shared/projects/tp_2630_ubordeaux_neuromics_184418/projects/C10/lederer/gbm_space_proj/scratch_build/tiny_snrna_1500.h5ad"  # DEMO: tiny subsample for rapid build -- swap back to the full path for the real course run',
)

# 2. Replace the risky adaptive scVI-heuristic training cell with a fixed, small,
# predictable epoch count (root cause of the original multi-hour hang -- removed, not
# patched around). Matches the TASK 4.3 code cell verbatim from build_solution_nb.py.
old_scvi_cell = '''code(r"""import scvi, torch
torch.set_num_threads(8)
scvi.settings.seed = 0

scvi_ad = adata.copy()
scvi.model.SCVI.setup_anndata(scvi_ad, layer="counts", batch_key="donor_id")

# What does scvi-tools *want* to run by default on this many cells?
from scvi.model._utils import get_max_epochs_heuristic
heuristic_epochs = get_max_epochs_heuristic(scvi_ad.n_obs)
print(f"scvi-tools epoch heuristic for {scvi_ad.n_obs:,} cells: {heuristic_epochs} epochs")""")'''
new_scvi_cell = '''code(r"""import scvi, torch
torch.set_num_threads(8)
scvi.settings.seed = 0

scvi_ad = adata.copy()
scvi.model.SCVI.setup_anndata(scvi_ad, layer="counts", batch_key="donor_id")
print(f"scVI reference set up on {scvi_ad.n_obs:,} cells")""")'''
assert old_scvi_cell in text, "scVI heuristic cell not found verbatim -- check for drift"
text = text.replace(old_scvi_cell, new_scvi_cell)

old_train_cell = '''code(r"""SCVI_MAX_EPOCHS = {SCVI_EPOCHS}   # capped for CPU; see timing note above
t0 = time.time()
model = scvi.model.SCVI(scvi_ad, n_latent=30)
model.train(max_epochs=SCVI_MAX_EPOCHS, early_stopping=False)
print(f"scVI trained ({SCVI_MAX_EPOCHS} epochs) in {(time.time()-t0)/60:.1f} min")

adata.obsm["X_scvi"] = model.get_latent_representation()
print("X_scvi shape:", adata.obsm["X_scvi"].shape)""")'''
new_train_cell = f'''code(r"""SCVI_MAX_EPOCHS = {placeholders["SCVI_EPOCHS"]}   # fixed, small, predictable -- NOT an adaptive heuristic (that caused a multi-hour hang on full-scale data and was removed; see build_notes.md). Raise for a real run, on a machine/timebox you control.
t0 = time.time()
model = scvi.model.SCVI(scvi_ad, n_latent=30)
model.train(max_epochs=SCVI_MAX_EPOCHS, early_stopping=False)
print(f"scVI trained ({{SCVI_MAX_EPOCHS}} epochs) in {{(time.time()-t0)/60:.1f}} min")

adata.obsm["X_scvi"] = model.get_latent_representation()
print("X_scvi shape:", adata.obsm["X_scvi"].shape)""")'''
assert old_train_cell in text, "scVI train cell not found verbatim -- check for drift"
text = text.replace(old_train_cell, new_train_cell)

# 3. Update the HINT markdown that quoted "real numbers from this run" -- mark explicitly
# as not independently re-validated at full scale, per instructor direction.
old_hint = '''md(r"""💡 **HINT — timing decision (real numbers from this run).** On 8 CPU cores, scVI costs roughly **{SCVI_PER_EPOCH}s/epoch** on ~117k nuclei. The heuristic above ({HEUR_EPOCHS} epochs) would therefore take ≈ **{HEUR_MIN} min** — too long for a hands-on session. We cap at **{SCVI_EPOCHS} epochs** (≈ **{SCVI_MIN} min**), which is plenty for two donors to mix while keeping the step interactive. On a GPU you would simply use the default.""")'''
new_hint = '''md(r"""💡 **HINT — runtime caveat.** scVI training cost scales with dataset size and is genuinely slow on CPU for tens of thousands of cells (observed: well over an hour for an *adaptive* epoch count on the full 117k-nucleus reference in this environment -- avoid scvi-tools' automatic epoch heuristic for that reason, it is not sized for a teaching timebox). We instead use a small, **fixed** epoch count you control directly. Raise it if you have time/GPU; the point of a fixed count is that you always know your runtime budget up front, instead of discovering it.""")'''
assert old_hint in text, "scVI hint markdown not found verbatim -- check for drift"
text = text.replace(old_hint, new_hint)

# 4. Substitute the CNV reference-cell-types and cluster->celltype placeholders with real values.
text = text.replace(
    "REFERENCE_CELL_TYPES = {REFERENCE_CELL_TYPES}   # clearly-diploid TME lineages",
    f"REFERENCE_CELL_TYPES = {placeholders['REFERENCE_CELL_TYPES']!r}   # clearly-diploid TME lineages (CellTypist-derived for this run)",
)
text = text.replace(
    "cluster_to_celltype = {CLUSTER_TO_CELLTYPE}",
    f"cluster_to_celltype = {placeholders['CLUSTER_TO_CELLTYPE']!r}",
)

# 5. Output path for the BUILD step.
text = text.replace(
    'OUT = Path("/shared/projects/tp_2630_ubordeaux_neuromics_184418/projects/C10/lederer/gbm_space_proj/notebooks/level1/01_snrna_analysis_solution.ipynb")',
    'OUT = Path("/shared/projects/tp_2630_ubordeaux_neuromics_184418/projects/C10/lederer/gbm_space_proj/notebooks/level1/01_snrna_analysis_solution.ipynb")',
)
# Fix kernelspec to the actually-registered kernel.
text = text.replace(
    'nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}',
    'nb.metadata["kernelspec"] = {"display_name": "Python (single_cell)", "language": "python", "name": "single_cell"}',
)

remaining_placeholders = re.findall(r"\{[A-Z_]+\}", text)
if remaining_placeholders:
    print(f"WARNING: unsubstituted placeholders remain: {set(remaining_placeholders)}")

PATCHED.write_text(text)
print(f"Wrote patched build script: {PATCHED}")

exec(compile(text, str(PATCHED), "exec"))
