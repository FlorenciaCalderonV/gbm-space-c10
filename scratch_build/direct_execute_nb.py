"""Execute a notebook's code cells directly in-process (no Jupyter kernel/ZMQ involved --
nbconvert's kernel startup hung on this server for unknown reasons). Captures stdout and
matplotlib figures, writes them back into the notebook JSON as real cell outputs.
"""
import base64
import contextlib
import io
import sys
import time

import nbformat as nbf

NB_PATH = sys.argv[1]
TIMEOUT_PER_CELL = float(sys.argv[2]) if len(sys.argv) > 2 else 300

nb = nbf.read(NB_PATH, as_version=4)
namespace: dict = {}

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

captured_figs = []


def fake_show(*a, **kw):
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=80)
    captured_figs.append(base64.b64encode(buf.getvalue()).decode())
    plt.close(fig)


plt.show = fake_show

for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    src = cell.source
    if not src.strip():
        continue
    # Strip Jupyter line-magics (%matplotlib inline, etc.) -- invalid in plain exec().
    src = "\n".join(line for line in src.split("\n") if not line.strip().startswith("%"))
    captured_figs.clear()
    buf = io.StringIO()
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Executing cell {i} ({len(src)} chars)...", flush=True)
    outputs = []
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(compile(src, f"<cell {i}>", "exec"), namespace)
        text = buf.getvalue()
        if text:
            outputs.append(nbf.v4.new_output("stream", name="stdout", text=text))
        for b64 in captured_figs:
            outputs.append(nbf.v4.new_output("display_data", data={"image/png": b64}))
        print(f"  -> OK in {time.time()-t0:.1f}s")
    except Exception as e:
        text = buf.getvalue()
        outputs.append(nbf.v4.new_output("stream", name="stdout", text=text))
        outputs.append(nbf.v4.new_output(
            "error", ename=type(e).__name__, evalue=str(e), traceback=[f"{type(e).__name__}: {e}"]
        ))
        print(f"  -> ERROR in {time.time()-t0:.1f}s: {type(e).__name__}: {e}", flush=True)
        cell.outputs = outputs
        cell.execution_count = i
        nbf.write(nb, NB_PATH)
        print(f"Saved partial notebook with error at cell {i}. Stopping.", flush=True)
        sys.exit(1)
    cell.outputs = outputs
    cell.execution_count = i
    nbf.write(nb, NB_PATH)  # save after every cell -- cheap, and protects partial progress

print("\n=== ALL CELLS EXECUTED SUCCESSFULLY ===", flush=True)
