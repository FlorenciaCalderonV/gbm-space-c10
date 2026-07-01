# IFB cluster quickstart — VS Code, passwordless login, conda

A short, project-agnostic setup for working on the **IFB Core cluster** from **VS Code**. Do this
once; it applies to any course project. About 15 minutes.

Cluster login host: `core.cluster.france-bioinformatique.fr` (you need an IFB account).

---

## 1. Install VS Code + the Remote-SSH extension

1. Install **VS Code** on your laptop: <https://code.visualstudio.com/>
2. Open the Extensions panel (`Ctrl/Cmd+Shift+X`) and install **Remote - SSH** (Microsoft).

(The **Python** and **Jupyter** extensions are installed later, on the cluster, in step 5.)

---

## 2. Passwordless login with an SSH key

Instead of typing your password every time, register your laptop with an SSH key.

**a. Make a key** (skip if `~/.ssh/id_ed25519` already exists). In a laptop terminal:
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

**c. Tell your SSH config to use the key.** Open `~/.ssh/config` (in VS Code: `F1` then
"Remote-SSH: Open SSH Configuration File") and add:
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
> If you gave the key a passphrase, you will be asked for *that* (not your account password); run
> `ssh-add ~/.ssh/id_ed25519` once per session to cache it.

---

## 3. Connect VS Code to the cluster

1. `F1` then **Remote-SSH: Connect to Host...** and pick **`ifb`**. No password prompt.
2. **File then Open Folder...** to open your working directory on the cluster.

---

## 4. Activate the shared conda environment

A ready-made environment called **`single_cell`** already exists on the cluster with the
scientific Python stack. You do not need to build it — just enable conda and activate it.

**a. Enable conda once** (in a cluster terminal; the VS Code terminal is fine):
```bash
/shared/software/miniconda/bin/conda init bash
```
Then open a **new terminal** (or run `source ~/.bashrc`) so `conda` is available.

**b. Make the shared environments discoverable** (run once; edits your `~/.condarc`):
```bash
conda config --append envs_dirs /shared/projects/tp_2630_ubordeaux_neuromics_184418/envs
```

**c. Activate `single_cell` and verify:**
```bash
conda activate single_cell
python -c "import scanpy; print('environment OK - scanpy', scanpy.__version__)"
```
You should see `environment OK - scanpy 1.11.5`. If `conda activate single_cell` cannot find it,
activate by full path instead:
```bash
conda activate /shared/projects/tp_2630_ubordeaux_neuromics_184418/envs/single_cell
```

> Rule of thumb: the login node is for editing and light setup only. Anything heavy (training,
> large data, long jobs) must go through **Slurm** — e.g. grab a compute node with
> `srun --account=<your-project-account> --partition=fast --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash`.

---

## 5. Open a Jupyter notebook in VS Code and run a cell

1. With VS Code connected to the cluster (step 3), install these extensions **on the remote**
   (VS Code shows an **"Install in SSH: ..."** button for each): **Python** (Microsoft) and
   **Jupyter** (Microsoft).
2. Create a notebook: **File then New File...** then choose **Jupyter Notebook** (or make a file
   ending in `.ipynb` and open it).
3. Top-right of the notebook, click **Select Kernel** then **Python Environments...** and choose
   **`single_cell`**. (If it does not appear, run `python -m ipykernel install --user --name single_cell`
   once in an activated terminal, then reload VS Code.)
4. In the first cell, type:
   ```python
   print("Hello world")
   ```
   Press **Shift+Enter** to run it. You should see `Hello world` printed below the cell.

If that prints, your environment and Jupyter setup are working.

---

## 6. (Optional) Create your own environment and add a package

`single_cell` is shared and read-only. When you need a package it does not have, make your **own**
environment. This example creates one and installs `scanpy` into it:
```bash
conda create -y -n myenv python=3.11 pip
conda activate myenv
pip install --no-user scanpy
python -c "import scanpy; print('my env has scanpy', scanpy.__version__)"
```
You should see `my env has scanpy 1.11.5`. Switch back to the course environment anytime with
`conda activate single_cell`.

> Always use `pip install --no-user` on a shared cluster, so packages go into the active
> environment and not a shared `~/.local` (which can silently break other environments).

Clean up when you no longer need it:
```bash
conda deactivate
conda env remove -y -n myenv
```

---

## 7. Sanity check — run the squidpy IMC tutorial

A final check that the spatial-analysis stack (`squidpy`, `scanpy`) and data downloading all work.
In a notebook using the **`single_cell`** kernel, work through this short official tutorial:

<https://squidpy.readthedocs.io/en/stable/notebooks/tutorials/tutorial_imc.html>

A minimal version you can paste cell-by-cell:
```python
import squidpy as sq

adata = sq.datasets.imc()                       # downloads a small example dataset (~1.5 MB)
print(adata)

sq.pl.spatial_scatter(adata, shape=None, color="cell type", size=10)   # a spatial plot

sq.gr.spatial_neighbors(adata)                  # build the spatial graph
sq.gr.nhood_enrichment(adata, cluster_key="cell type")   # neighborhood enrichment test
sq.pl.nhood_enrichment(adata, cluster_key="cell type")   # heatmap
```
If the spatial scatter and the enrichment heatmap both render, your environment is fully working.
This runs in about a minute on the login node (the neighborhood-enrichment permutation test is the
slowest part). If you want to run the whole tutorial, that is fine too — it is small.

Once this works, you are ready to start a project.
