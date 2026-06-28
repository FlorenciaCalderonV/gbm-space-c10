"""Derive a student notebook from a completed, executed solution notebook: keep all
markdown (TASK/HINT/QUESTION/CHECKPOINT guidance -- none of it embeds real computed
results, verified separately), blank out every code cell to a single placeholder comment,
clear all outputs/execution_counts. Mechanical and safe by construction.
"""
import re
import sys

import nbformat as nbf

SRC, DST = sys.argv[1], sys.argv[2]

# One short, generic placeholder comment per section -- matches the template's bare
# "# Your ... here" convention (no `# TODO:`), inferred from the nearest preceding TASK line.
nb = nbf.read(SRC, as_version=4)
out_cells = []
last_task_text = "this step"

for cell in nb.cells:
    if cell.cell_type == "markdown":
        m = re.search(r"TASK\s+[\d.]+:\*\*\s*(.+?)(?:\n|$)", cell.source)
        if m:
            last_task_text = m.group(1).strip().rstrip(".")
        out_cells.append(nbf.v4.new_markdown_cell(cell.source))
    else:
        placeholder = f"# Your code for: {last_task_text}\n"
        out_cells.append(nbf.v4.new_code_cell(placeholder))

nb.cells = out_cells
nbf.write(nb, DST)
print(f"Wrote student notebook: {DST} ({len(out_cells)} cells)")
