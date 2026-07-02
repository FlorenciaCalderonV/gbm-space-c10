# Setup guide — C10 (IFB Core cluster)

**New here? Work through steps 1–8 below in order** — they explain each part of the setup. Once
you've done it once, there's a condensed **Fast path** at the very bottom you can reuse.

Everything for this project runs on the **IFB Core cluster**. You will:
1. set up passwordless login,
2. connect VS Code to the cluster,
3. clone the materials (from the VS Code terminal),
4. activate the shared conda environment,
5. run a notebook, and
6. run any heavy computation through **Slurm** (never on the login node).

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

You need an IFB account with access to the project (ask your instructor if you can't log in).

---

## 1. Passwordless login with an SSH key

Set this up **first**. It lets both a terminal and VS Code connect to the cluster without typing
your password every time.

**a. Make a key** (skip if `~/.ssh/id_ed25519` already exists). In a terminal **on your laptop**:
```bash
ssh-keygen -t ed25519          # press Enter through the prompts
```

**b. Copy the public key to the cluster** (asks for your password one last time):
```bash
# macOS / Linux:
ssh-copy-id <your-username>@core.cluster.france-bioinformatique.fr

# Windows PowerShell (if ssh-copy-id is missing):
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh <your-username>@core.cluster.france-bioinformatique.fr "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

**c. Add a host entry to your laptop's `~/.ssh/config`** (VS Code reads this too):
```
Host ifb
    HostName core.cluster.france-bioinformatique.fr
    User <your-username>
    IdentityFile ~/.ssh/id_ed25519
```

**d. Test — this should log in with no password:**
```bash
ssh ifb
```
> If it still asks for a password: on the cluster run `chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys`.
> If you gave the key a passphrase, you'll be asked for *that* (not your account password); run
> `ssh-add ~/.ssh/id_ed25519` once per session to cache it.

---

## 2. VS Code + connect to the cluster

1. Install **VS Code** on your laptop: https://code.visualstudio.com/
2. Open the Extensions panel (`Ctrl/Cmd+Shift+X`) and install **Remote - SSH** (Microsoft).
3. Press `F1` → **Remote-SSH: Connect to Host…** → pick **`ifb`** (the entry you added in step 1c).
   No password prompt. VS Code now runs *on the cluster* — its terminal, file explorer, and
   extensions all operate there.
4. Open the integrated terminal with **Terminal → New Terminal**. You'll run the next steps
   (clone, setup) right here, inside VS Code, on the cluster.

> The **Python**, **Jupyter**, and **Claude** extensions are installed later, **on the remote**
> (VS Code shows an "Install in SSH: …" button once you're connected).

---

## 3. Get the materials with git (fork + upstream)

Run these in the **VS Code integrated terminal** (step 2.4). You'll keep your **own copy** of the
repo on GitHub — a *fork* — so you can freely edit the notebooks while still pulling in updates the
instructor makes.

**Step 1 — fork (once, in your browser).** Sign in to GitHub, go to
<https://github.com/arl94/gbm-space-c10>, and click **Fork** (top-right) → *Create fork*. You now
own `https://github.com/<your-github-username>/gbm-space-c10`.

**Step 2 — clone YOUR fork over HTTPS.** The repo is **public**, so cloning needs **no SSH key and
no token** — just use the `https://` URL:
```bash
git clone https://github.com/<your-github-username>/gbm-space-c10.git ~/gbm-space-c10
cd ~/gbm-space-c10
```
> Don't use the `git@github.com:…` (SSH) URL for cloning — without an SSH key that fails with
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
> needs a key. Point it at HTTPS instead, then pull again:
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
credentials, but **pushing does**. Set up auth only when you first want to push:
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

## 4. The conda environment

A ready-made environment called **`single_cell`** already exists on the cluster with the full
stack (scanpy, anndata, scvi-tools, cell2location, celltypist, infercnvpy, squidpy, harmonypy,
liana, decoupler, torch, jupyter…). You do **not** need to build it — just enable conda and
activate it.

> **Shortcut:** from your clone, `bash setup.sh` does all of this section (enable conda, make the
> env discoverable, verify it, register the Jupyter kernel). The manual steps are below in case you
> want to understand them or the script fails.

**Enable conda in your shell.** `module load conda` only puts `conda` on your `PATH`; it does
**not** enable `conda activate` (you'd get *"Run 'conda init' before 'conda activate'"*). Do this
**once** so conda works in every future shell:
```bash
/shared/software/miniconda/bin/conda init bash
conda config --set auto_activate_base false   # don't auto-enter 'base' on every login
```
Then **open a new terminal** (or `source ~/.bashrc`). From now on, `conda` is ready in any shell.

> One-off alternative (no `~/.bashrc` change): run
> `source /shared/software/miniconda/etc/profile.d/conda.sh` at the start of each session.

**Point conda at the shared course environments** (run once; edits your `~/.condarc`):
```bash
conda config --append envs_dirs /shared/projects/tp_2630_ubordeaux_neuromics_184418/envs
```

**Activate and check:**
```bash
conda activate single_cell
python -c "import scanpy; print('environment OK - scanpy', scanpy.__version__)"
```
You should see `environment OK - scanpy 1.11.5`. If `conda activate single_cell` can't find it,
activate by full path instead:
```bash
conda activate /shared/projects/tp_2630_ubordeaux_neuromics_184418/envs/single_cell
```

---

## 5. Open a Jupyter notebook in VS Code and run a cell

1. Install these extensions **on the remote** (VS Code shows an "Install in SSH: …" button):
   **Python** (Microsoft) and **Jupyter** (Microsoft).
2. Create a notebook: **File → New File…** then choose **Jupyter Notebook** (or make a file ending
   in `.ipynb` and open it).
3. Top-right of the notebook, click **Select Kernel** → **Python Environments…** and pick
   **`single_cell`**. If it doesn't appear, run `python -m ipykernel install --user --name single_cell`
   once in an activated terminal, then reload VS Code.
4. In the first cell, type:
   ```python
   print("Hello world")
   ```
   Press **Shift+Enter** to run it. You should see `Hello world` printed below the cell.

> **Where does the kernel actually run?** On **whatever machine VS Code is connected to**. If you
> connected to the login node (step 2) the kernel runs there, with *no more memory or CPU than the
> shared login node* — fine only for light editing and tiny tests. **To run notebooks with real
> memory/compute, connect VS Code to a compute node — see step 7.**

---

## 6. (Optional) Create your own environment and add a package

`single_cell` is shared and read-only. When you need a package it doesn't have, make your **own**
environment. This example creates one and installs `scanpy` into it:
```bash
conda create -y -n myenv python=3.11 pip
conda activate myenv
pip install --no-user scanpy
python -c "import scanpy; print('my env has scanpy', scanpy.__version__)"
```
You should see `my env has scanpy 1.11.5`. Switch back to the course environment anytime with
`conda activate single_cell`. Clean up when done: `conda deactivate && conda env remove -y -n myenv`.

> Always use `pip install --no-user` on a shared cluster, so packages go into the active
> environment and not a shared `~/.local` (which can silently break other environments).

You can also rebuild the full course environment yourself from the pinned spec (slower):
`mamba env create -n single_cell_mine -f single_cell_environment.yml`.

---

## 7. Running notebooks with real memory/compute — use Slurm, never the login node

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
2. **On your laptop**, extend `~/.ssh/config` (once):
   ```
   Host ifb
       HostName core.cluster.france-bioinformatique.fr
       User <your-username>
       IdentityFile ~/.ssh/id_ed25519

   Host cpu-node-* gpu-node-*
       User <your-username>
       ProxyJump ifb
   ```
3. **In VS Code:** `F1` → "Remote-SSH: Connect to Host…" → type the node name (`cpu-node-42`).
   VS Code now runs *on the compute node*. Open the project folder, open the notebook, select the
   `single_cell` kernel (step 5). The kernel now has your allocation's 8 CPUs / 64 GB.
4. When finished, close the VS Code remote window and type `exit` in the `salloc` terminal to
   release the node.

> The SSH-to-compute-node connection works **only while your `salloc` allocation is alive** (your
> session is attached to that job and bounded by its resources). If it ends, reconnect after a
> new `salloc`. If connecting to the node fails at your site, use Method B.

Ask for the resources you actually need and a **short** `--time` (shorter jobs start sooner):
bump `--mem` (e.g. `--mem=128G`) for the memory-heavy steps like inferCNV on the full dataset.

### Method A2 — grab a node, then `ssh` into it from the cluster
If you'd rather stay in a plain terminal than reconnect VS Code, you can log directly onto the
compute node you were allocated (this works because the cluster attaches your SSH session to your
running job):
1. From the **login node**, request a node and keep this terminal open:
   ```bash
   salloc --account=tp_2630_ubordeaux_neuromics_184418 --partition=fast \
          --cpus-per-task=8 --mem=64G --time=06:00:00
   squeue -u $USER      # note the node, e.g. cpu-node-42
   ```
2. From **another login-node terminal**, SSH onto that node, activate the env, and work there:
   ```bash
   ssh cpu-node-42
   conda activate single_cell
   jupyter lab --no-browser --ip=0.0.0.0 --port=8888   # or run your scripts / nbconvert
   ```
   Your shell (and anything it launches) now runs on the compute node with the allocation's
   resources. To use a notebook in the browser, forward the port as in Method C. This only works
   while the `salloc` job is alive.

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

## 8. Claude in VS Code (AI coding help)

1. With VS Code connected to the cluster, install the **Claude Code** extension (publisher:
   Anthropic) **on the remote**.
2. Open the Claude panel from the sidebar and **sign in** when prompted (a browser window opens
   for authentication).
3. Ask Claude for help right next to your code — explaining a function, debugging an error,
   drafting an analysis step.

> Use AI as a **learning accelerator, not a replacement for understanding**. If Claude suggests a
> function or parameter you don't recognize, pause and learn what it does before using it — that's
> the part that transfers to your own future projects.

---

## Note on GPUs
Two steps in the project are much faster on a GPU: **scVI** integration (Level 1) and
**cell2location** training (Level 2). The `gpu` partition is **not currently enabled** for this
course account. Until it is, either use the **Harmony** integration path in Level 1 (CPU, built
into the notebook) and the provided **precomputed cell2location results** for Level 2, or run the
GPU steps on CPU with reduced settings for a smaller test. Your instructor will let you know when
GPU access is available.

---

## Quick reference
```bash
ssh ifb                                                  # connect (passwordless, from ~/.ssh/config)
conda activate single_cell                               # environment (after 'conda init')
srun --account=tp_2630_ubordeaux_neuromics_184418 \
     --partition=fast --cpus-per-task=8 --mem=32G \
     --time=04:00:00 --pty bash                          # interactive compute node
sbatch run_notebook.sh                                   # batch job
squeue -u $USER                                          # my jobs
git pull upstream master                                 # get instructor updates
```

---

## Fast path (once you've done the full setup once)

The order matters: get **VS Code + passwordless login** working **first** (steps 1–2), then do
everything else — including cloning — from the **terminal inside VS Code**.

1. Set up a passwordless SSH key and connect VS Code to the cluster (steps 1–2).
2. Open the VS Code integrated terminal (**Terminal → New Terminal**) and run:
   ```bash
   git clone https://github.com/<your-github-username>/gbm-space-c10.git ~/gbm-space-c10
   cd ~/gbm-space-c10
   git remote add upstream https://github.com/arl94/gbm-space-c10.git   # instructor repo, for updates
   bash setup.sh
   ```
`setup.sh` enables conda, verifies the `single_cell` environment, and registers the Jupyter
kernel. The repo is **public**, so cloning needs **no SSH key and no token**. **Setup only — run
heavy compute via Slurm (step 7), not the login node.**
