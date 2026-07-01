# Setup guide — C10 (IFB Core cluster)

> ## ⚡ Fast path
> First **fork** this repo on GitHub into your own account (§1b). Then, once you can log in to the
> cluster (§1), clone **your fork over HTTPS** — the repo is public, so cloning needs **no SSH key
> and no token** — and run the setup script (on the **login node**):
> ```bash
> git clone https://github.com/<your-github-username>/gbm-space-c10.git ~/gbm-space-c10
> cd ~/gbm-space-c10
> git remote add upstream https://github.com/arl94/gbm-space-c10.git   # instructor's repo, for updates
> bash setup.sh
> ```
> `setup.sh` enables conda, verifies the `single_cell` environment, and registers the Jupyter
> kernel. Then open a **new terminal** and jump to §3 (VS Code). The sections below explain each
> step manually. **Setup only — run heavy compute via Slurm (§5), not the login node.**
>
> **Get instructor updates later:** `git pull upstream master`. Do your analysis in a *copy* of the
> student notebook (e.g. `01_..._student.ipynb` → `01_..._myname.ipynb`) so updates never clash with your edits.

Everything for this project runs on the **IFB Core cluster**. You will:
1. connect to the cluster,
2. activate the shared conda environment,
3. edit/run notebooks from **VS Code** with the **Jupyter** and **Claude** extensions,
4. run any heavy computation through **Slurm** (never on the login node).

Cluster facts you'll reuse:

| Thing | Value |
|---|---|
| Login host | `core.cluster.france-bioinformatique.fr` |
| Your Slurm account | `tp_2630_ubordeaux_neuromics_184418` |
| Project folder | `/shared/projects/tp_2630_ubordeaux_neuromics_184418/projects/C10` |
| CPU partitions | `fast` (default, ≤ 24 h), `long` (≤ 30 d) |
| GPU partition | `gpu` — **not enabled for this course account yet** (see note at the end) |
| Conda env | `single_cell` |

IFB user documentation: https://ifb-elixirfr.gitlab.io/cluster/doc/

---

## 1. Connect to the cluster

You need an IFB account with access to the project (ask your instructor if you can't log in).

From a terminal:
```bash
ssh <your-ifb-username>@core.cluster.france-bioinformatique.fr
```
You land on a **login node**. The login node is for editing, small commands, and submitting
jobs — **not** for heavy computation (see §5).

---

## 1b. Get the materials with git (fork + upstream)

You'll keep your **own copy** of the repo on GitHub — a *fork* — so you can freely edit and
complete the student notebooks, while still pulling in any updates the instructor makes to the
originals. (A fork is the right model here: you get your own writable copy without needing push
access to the instructor's repo, and everyone's work stays isolated.)

**Step 1 — fork (once, in your browser).** Sign in to GitHub, go to
<https://github.com/arl94/gbm-space-c10>, and click **Fork** (top-right) → *Create fork*. You now
own `https://github.com/<your-github-username>/gbm-space-c10`.

**Step 2 — clone YOUR fork onto the cluster over HTTPS** (login node). The repo is **public**, so
cloning needs **no SSH key and no token** — just use the `https://` URL:
```bash
git clone https://github.com/<your-github-username>/gbm-space-c10.git ~/gbm-space-c10
cd ~/gbm-space-c10
```
> ⚠️ Don't use the `git@github.com:…` (SSH) URL for cloning — without an SSH key that fails with
> *"Permission denied (publickey) … could not read from remote repository."* HTTPS avoids this.

**Step 3 — link the instructor's repo as `upstream`** (this is how you receive updates):
```bash
git remote add upstream https://github.com/arl94/gbm-space-c10.git
git remote -v      # 'origin' = your fork (you push here); 'upstream' = instructor (you pull from here)
```

**Get instructor updates anytime** — safe to run whenever notebooks change (public repo, no auth):
```bash
git pull upstream master
```
> **If `git pull upstream master` fails with `Permission denied (publickey)` / `could not read
> from remote repository`:** your `upstream` was added with the SSH URL (`git@github.com:…`), which
> needs a key. Point it at HTTPS instead (no key needed for a public repo), then pull again:
> ```bash
> git remote set-url upstream https://github.com/arl94/gbm-space-c10.git
> git remote -v            # upstream should now show https://
> git pull upstream master
> ```

To keep those pulls **conflict-free**, do your analysis in a *copy* of the student notebook so you
never edit the tracked file the instructor might also update:
```bash
cp notebooks/level1/01_snrna_analysis_student.ipynb notebooks/level1/01_myname.ipynb
```

**Saving your own progress back to your fork (optional).** Cloning/pulling a public repo needs no
credentials, but **pushing does**. So set up auth only when you first want to push:
```bash
git add notebooks/level1/01_myname.ipynb
git commit -m "my level 1 progress"
git push origin master        # asks for GitHub username + a Personal Access Token (not your password)
```
- **Easiest:** create a **Personal Access Token** (GitHub → Settings → Developer settings → Tokens)
  and paste it when `git push` prompts for a password.
- **Or SSH:** `ssh-keygen -t ed25519` on the cluster, add `~/.ssh/id_ed25519.pub` to GitHub →
  *Settings → SSH and GPG keys*, then switch your push URL with
  `git remote set-url origin git@github.com:<your-github-username>/gbm-space-c10.git`.

The big data **and the precomputed model files are not in the repo** — they stay on the shared
filesystem (paths are in the notebooks / `README.md` / `precomputed/README.md`).

---

## 2. The conda environment

A ready-made environment called **`single_cell`** already exists on the cluster with the full
stack (scanpy, anndata, scvi-tools, cell2location, celltypist, infercnvpy, squidpy, harmonypy,
liana, decoupler, torch, jupyter…). You do **not** need to build it — just enable conda and make
it discoverable.

> 🚀 **Shortcut:** the setup script does all of §2 (and copies the materials) for you — see the
> box at the top of this file. The manual steps are below in case you want to understand them or
> the script fails.

**Enable conda in your shell.** `module load conda` only puts `conda` on your `PATH`; it does
**not** enable `conda activate` (you'd get *"Run 'conda init' before 'conda activate'"*). Do this
**once** so conda works in every future shell:
```bash
/shared/software/miniconda/bin/conda init bash
conda config --set auto_activate_base false   # don't auto-enter 'base' on every login
```
Then **open a new terminal** (or `source ~/.bashrc`). From now on, `conda` is ready in any shell.

> One-off alternative (no `~/.bashrc` change): run
> `source /shared/software/miniconda/etc/profile.d/conda.sh` at the start of each session
> instead of `conda init`.

**Point conda at the shared course environments** (run once; edits your `~/.condarc`):
```bash
conda config --append envs_dirs /shared/projects/tp_2630_ubordeaux_neuromics_184418/envs
```

**Activate and check:**
```bash
conda activate single_cell
python -c "import scanpy, squidpy, cell2location, celltypist, infercnvpy, liana; print('env OK')"
```
If `conda activate single_cell` can't find it, activate by full path instead:
```bash
conda activate /shared/projects/tp_2630_ubordeaux_neuromics_184418/envs/single_cell
```

### (Optional) Build your own copy
Only if you need your own environment, from this folder (after `conda init`, in a fresh shell):
```bash
mamba env create -n single_cell_mine -f single_cell_environment.yml
conda activate single_cell_mine
```
> On a shared cluster, if you ever `pip install` into a conda env, always use
> `pip install --no-user` (a plain `pip install` can write into `~/.local` and quietly break
> other environments).

---

## 3. VS Code + connecting to the cluster

1. Install **VS Code** on your own laptop: https://code.visualstudio.com/
2. In VS Code, open the Extensions panel (`Ctrl/Cmd+Shift+X`) and install **"Remote - SSH"**
   (publisher: Microsoft).
3. Press `F1` → **"Remote-SSH: Connect to Host…"** → **"Add New SSH Host…"** and enter:
   ```
   ssh <your-ifb-username>@core.cluster.france-bioinformatique.fr
   ```
   Then connect. VS Code now runs *on the cluster* — its terminal, file explorer, and
   extensions all operate there.
4. **File → Open Folder…** and open your clone, `~/gbm-space-c10` (see the Fast-path box for the
   `git clone` command, and §2 for git access if you haven't cloned yet).

> All the remaining extensions (Python, Jupyter, Claude) must be installed **on the remote**
> (VS Code shows an "Install in SSH: …" button once you're connected).

---

## 4. Jupyter notebooks in VS Code

Install these extensions **in the remote** (VS Code shows an "Install in SSH: …" button):
- **Python** (Microsoft)
- **Jupyter** (Microsoft)

Then open a notebook, click **"Select Kernel"** (top-right) → **"Python Environments…"** → pick
**`single_cell`** (or the full-path env from §2), and run a cell to confirm it's alive. If
`single_cell` doesn't appear, run `python -m ipykernel install --user --name single_cell` once
in an activated shell, then reload VS Code.

> ⚠️ **Where does the kernel actually run?** It runs on **whatever machine VS Code is connected
> to**. If you followed §3 you are connected to the **login node** — so the kernel runs there,
> with *no more memory or CPU than the shared login node*, and heavy cells will be slow and may
> be killed. This is fine **only** for light editing and tiny tests. **To run the notebooks with
> real memory/compute, connect VS Code to a *compute node* — see §5.**

---

## 5. Running notebooks with real memory/compute — use Slurm, never the login node

The login node is shared by everyone and is **not** for real compute. To give your notebook the
memory and CPUs it needs, the kernel must run on a Slurm **compute node**. Pick one method:

### Method A (recommended) — connect VS Code directly to a compute node
This gives the full VS Code notebook experience (integrated kernel, terminal, debugger) with the
resources of a Slurm allocation.

1. **On the login node**, request an allocation and keep this terminal open:
   ```bash
   salloc --account=tp_2630_ubordeaux_neuromics_184418 --partition=fast \
          --cpus-per-task=8 --mem=64G --time=06:00:00
   squeue -u $USER      # note the node you were given, e.g. cpu-node-42
   ```
2. **On your laptop**, add to `~/.ssh/config` (once):
   ```
   Host ifb
       HostName core.cluster.france-bioinformatique.fr
       User <your-ifb-username>

   Host cpu-node-* gpu-node-*
       User <your-ifb-username>
       ProxyJump ifb
   ```
3. **In VS Code:** `F1` → "Remote-SSH: Connect to Host…" → type the node name (`cpu-node-42`).
   VS Code now runs *on the compute node*. Open the project folder, open the notebook, select the
   `single_cell` kernel (§4). The kernel now has your allocation's 8 CPUs / 64 GB.
4. When finished, close the VS Code remote window and type `exit` in the `salloc` terminal to
   release the node.

> The SSH-to-compute-node connection works **only while your `salloc` allocation is alive** (your
> session is attached to that job and bounded by its resources). If it ends, reconnect after a
> new `salloc`. If connecting to the node fails at your site, use Method B.

Ask for the resources you actually need and a **short** `--time` (shorter jobs start sooner):
bump `--mem` (e.g. `--mem=128G`) for the memory-heavy steps like inferCNV on the full dataset.

### Method B — batch execution for long, unattended runs
Best for running a whole notebook start-to-finish without babysitting it. Save as
`run_notebook.sh` (edit the path), then `sbatch` it:
```bash
#!/bin/bash
#SBATCH --account=tp_2630_ubordeaux_neuromics_184418
#SBATCH --partition=fast
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module load conda
conda activate single_cell
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/level1/01_snrna_analysis_student.ipynb
```
```bash
sbatch run_notebook.sh
squeue -u $USER            # is it running?
tail -f run_notebook_*.out # live output
```

### Method C — classic Jupyter Lab in the browser (alternative)
From inside an `salloc`/`srun` allocation on a compute node:
```bash
module load conda && conda activate single_cell
jupyter lab --no-browser --ip=0.0.0.0 --port=8888
```
then forward the port from your laptop and open the printed URL:
```bash
ssh -J <user>@core.cluster.france-bioinformatique.fr -L 8888:<node>:8888 <user>@<node>
```

---

## 6. Claude in VS Code (AI coding help)

1. With VS Code connected to the cluster, install the **Claude Code** extension (publisher:
   Anthropic) **in the remote**.
2. Open the Claude panel from the sidebar and **sign in** when prompted (a browser window opens
   for authentication).
3. Ask Claude for help right next to your code — explaining a function, debugging an error,
   drafting an analysis step.

> Use AI as a **learning accelerator, not a replacement for understanding**. If Claude
> suggests a function or parameter you don't recognize, pause and learn what it does before
> using it — that's the part that transfers to your own future projects.

---

## Note on GPUs
Two steps in the project are much faster on a GPU: **scVI** integration (Level 1) and
**cell2location** training (Level 2). The `gpu` partition is **not currently enabled** for this
course account. Until it is, either use the **Harmony** integration path in Level 1 (CPU, built
into the notebook) and the provided **precomputed cell2location results** for Level 2, or run
the GPU steps on CPU with reduced settings for a smaller test. Your instructor will let you
know when GPU access is available.

---

## Quick reference
```bash
ssh <user>@core.cluster.france-bioinformatique.fr        # connect
module load conda && conda activate single_cell          # environment
srun --account=tp_2630_ubordeaux_neuromics_184418 \
     --partition=fast --cpus-per-task=8 --mem=32G \
     --time=04:00:00 --pty bash                          # interactive compute node
sbatch run_notebook.sh                                   # batch job
squeue -u $USER                                          # my jobs
```
