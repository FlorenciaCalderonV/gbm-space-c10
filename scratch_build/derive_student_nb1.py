"""Derive the Level 1 STUDENT notebook from the completed, executed SOLUTION notebook.

Rules (per project style guide):
- Code cells -> blank placeholder (one-line comment derived from the preceding TASK).
- Markdown TASK/QUESTION/CHECKPOINT cells: kept (they're instructions, not answers).
- HINT cells that reveal a specific computed decision/value: softened to remove the
  giveaway number while keeping the conceptual nudge.
- Strip all cell outputs (solution-only).
- Point the data path back at the REAL full-scale dataset (the solution was executed on a
  tiny demo subsample for rapid turnaround -- students get the real data).
- No paper reveal, no answer-key column names anywhere (none were present in the solution
  to begin with -- verified by construction, see build_solution_nb.py).
"""
import re
from pathlib import Path

import nbformat as nbf

ROOT = Path("/shared/projects/tp_2630_ubordeaux_neuromics_184418/projects/C10/lederer/gbm_space_proj")
SRC = ROOT / "notebooks/level1/01_snrna_analysis_solution.ipynb"
OUT = ROOT / "notebooks/level1/01_snrna_analysis_student.ipynb"

# Specific over-revealing HINTs to soften (exact match on the giveaway substring -> replacement).
HINT_SOFTENING = {
    "We cap at **20 epochs**": "Cap at a small, fixed number of epochs you choose",
    "SCVI_MAX_EPOCHS = 5": "SCVI_MAX_EPOCHS = ...   # choose a small, fixed number -- see HINT above",
}

TASK_RE = re.compile(r"TASK\s+[\d.]+:\*\*\s*(.+?)(?:\n|$)")


def extract_task_summary(prev_md: str) -> str:
    m = TASK_RE.search(prev_md)
    if m:
        return m.group(1).strip().rstrip(".")
    return "this step"


nb = nbf.read(SRC, as_version=4)
new_cells = []
prev_md_text = ""

for cell in nb.cells:
    if cell.cell_type == "markdown":
        text = cell.source
        # Soften specific over-revealing hints.
        for giveaway, replacement in HINT_SOFTENING.items():
            if giveaway in text:
                text = text.replace(giveaway, replacement)
        # Real-data path note: solution ran on a tiny demo subsample; tell students nothing
        # about that -- they get the real path directly, no mention needed in their copy.
        new_cells.append(nbf.v4.new_markdown_cell(text))
        prev_md_text = text
    else:
        summary = extract_task_summary(prev_md_text)
        if "tiny_snrna_1500.h5ad" in cell.source or "DATA =" in cell.source:
            # The data-loading cell: give the real path as a starting point (an
            # administrative detail, not a learning objective) -- the real course path,
            # not the solution's tiny demo subsample.
            blank = nbf.v4.new_code_cell(
                'DATA = "/shared/projects/tp_2630_ubordeaux_neuromics_184418/projects/C10/data/snRNA_seq/level1_prepared/gbm_l1_snrna_AT10_AT14_raw.h5ad"\n'
                f"# Your code for: {summary}\n"
            )
        else:
            blank = nbf.v4.new_code_cell(f"# Your code for: {summary}\n")
        new_cells.append(blank)

nb_out = nbf.v4.new_notebook()
nb_out.cells = new_cells
nb_out.metadata["kernelspec"] = {"display_name": "Python (single_cell)", "language": "python", "name": "single_cell"}
nb_out.metadata["language_info"] = {"name": "python"}

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb_out, OUT)
print(f"Wrote {OUT} with {len(nb_out.cells)} cells "
      f"({sum(1 for c in new_cells if c.cell_type=='markdown')} md, "
      f"{sum(1 for c in new_cells if c.cell_type=='code')} code)")
