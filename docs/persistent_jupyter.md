# Keep your notebooks running when you close your laptop

This guide shows how to run Jupyter notebooks on the IFB Core cluster so that
**your work keeps running on the compute node even after you disconnect** (close
your laptop, lose Wi-Fi, quit VS Code). You reconnect later and land back on the
same running kernels, with all your variables and in-progress cells intact.

It relies on one helper script, `start_jupyter.sh`, in the project root.

---

## Why the naive approach loses your work

The tempting workflow is: open VS Code, use Remote-SSH to connect to a compute
node, open a `.ipynb`, and run cells. This works while you are connected, but it
has a fatal flaw for long analyses:

- The Jupyter **kernel** (the Python process holding your data in memory) is
  started **by, and tied to, your VS Code / SSH session**.
- When your laptop sleeps or you close VS Code, that session ends, and the kernel
  is killed along with it.
- Everything in memory is gone. A cell that was still running (integration,
  cell2location, inferCNV) is interrupted and you start over.

The problem is that the kernel's lifetime is bound to your laptop's connection.

## How the fix works

We move the notebook server **inside the Slurm job**, so it no longer depends on
your laptop at all:

1. `start_jupyter.sh` submits a Slurm job on the `fast` partition. **The job
   itself** starts a JupyterLab server on the compute node.
2. The server and all its kernels live inside that Slurm allocation. Your laptop
   is now just a *viewer* that connects to it over an SSH tunnel.
3. When your laptop disconnects, only the viewer goes away. The Slurm job keeps
   running, so the server and every running cell keep going.
4. To get back in, you re-open the same tunnel and the same URL. You reconnect to
   the **same live server and kernels** — nothing was lost.

Persistence lasts **as long as the Slurm job is alive** (until its time limit, or
until you `scancel` it). It is not forever — it is "for the life of your job",
which you control (default 8 hours, override with `--time`).

---

## One-time facts about the cluster

- **Login host:** `core.cluster.france-bioinformatique.fr`. This is where you run
  `start_jupyter.sh`.
- The **compute node is reached *through* the login node.** You never connect to a
  compute node directly from the internet; the SSH tunnel uses the login node as a
  jump host (`-J`, "ProxyJump"). This already works on IFB **while you hold a job
  on that node** — which you do, because your Jupyter job is running there.
- Everything runs in the read-only `single_cell` conda environment, which already
  has JupyterLab installed. You do not install anything.
- In the commands below, replace `<user>` with your IFB username (e.g. the
  `tp######` you log in with) and `<node>` / `<port>` / `<token>` with the values
  the script prints for you.

---

## Method 1 — Browser (recommended, simplest)

### Step 1: start the server (on the login node)

Log in to the cluster and run the helper from the project root:

```bash
cd /shared/projects/tp_2630_ubordeaux_neuromics_184418/projects/C10/lederer/gbm_space_proj
./start_jupyter.sh
```

You can override the resources; every flag also has an environment-variable form:

```bash
./start_jupyter.sh --cpus 16 --mem 128G --time 12:00:00 --port 8888 --dir ~/C10
# or:  CPUS=16 MEM=128G TIME=12:00:00 ./start_jupyter.sh
```

Defaults: partition `fast`, account `tp_2630_ubordeaux_neuromics_184418`,
8 CPUs, 64 GB, 8 hours, port 8888, notebook root `$HOME`.

The script submits the job and prints a job ID and next steps.

### Step 2: wait for it to start, then read the connection info

```bash
squeue -u $USER          # wait until STATE shows 'R' (running)
cat ~/jupyter_connect.txt # the job writes node, port, token, tunnel and URL here
```

If `~/jupyter_connect.txt` is not there yet, the job has not started — wait a few
seconds and try again. (JupyterLab takes ~30-60 s to boot the first time.)

The file looks like this (your node, port and token will differ):

```
NODE  : cpu-node-104
PORT  : 8899
TOKEN : 4e68cde5b06dd853fdf3476879104dd18e9dced9da552cca

STEP 1 — On your LAPTOP, open a terminal and run this (keep it open):

  ssh -N -L 8899:cpu-node-104:8899 -J tp185005@core.cluster.france-bioinformatique.fr tp185005@cpu-node-104

STEP 2 — In your browser on the laptop, open:

  http://localhost:8899/lab?token=4e68cde5b06dd853fdf3476879104dd18e9dced9da552cca
```

### Step 3: open the SSH tunnel from your LAPTOP

On your **laptop** (not the cluster), open a terminal and paste the tunnel command
from the file. Its general shape is:

```bash
ssh -N -L <port>:<node>:<port> -J <user>@core.cluster.france-bioinformatique.fr <user>@<node>
```

What this does:

- `-L <port>:<node>:<port>` forwards `localhost:<port>` on your laptop to
  `<node>:<port>` on the cluster.
- `-J <user>@core.cluster...` jumps through the login node to reach the compute
  node (which is not directly reachable).
- `-N` means "just forward, do not open a shell."

**Leave this terminal open.** It is the tunnel. On Windows, use the same command
in PowerShell, Git Bash, or WSL.

### Step 4: open JupyterLab in your browser

Open the URL from the file, which points at your own laptop:

```
http://localhost:<port>/lab?token=<token>
```

You now have JupyterLab running with full compute-node CPU and memory. Open a
notebook and work normally.

### Step 5: close your laptop, come back later

Close the lid, walk away, lose your connection — it does not matter. The Slurm job,
the server, and every running cell keep going on the compute node.

To reconnect: **re-run the same Step 3 tunnel command** and **reopen the same
Step 4 URL**. You are back on the exact same session with all your variables and
any still-running cells. (If your laptop changed networks, that is fine; the node
and token are unchanged as long as the job is alive.)

---

## Method 2 — VS Code (connect to the persistent server)

If you prefer VS Code, do not let it start its own kernel on the node. Instead,
point it at the persistent server from Method 1 so the kernel lives in the Slurm
job, not in VS Code.

1. Do Steps 1-3 of Method 1 (start the job, read `~/jupyter_connect.txt`, open the
   SSH tunnel from your laptop). Keep the tunnel terminal open.
2. In VS Code, open the command palette (`Ctrl/Cmd+Shift+P`) and run
   **"Jupyter: Specify Jupyter Server for Connections..."**
   (equivalently, open a notebook, click the kernel picker in the top right, and
   choose **"Existing Jupyter Server..."**).
3. Choose **"Existing"** and paste the server URL **including the token**:

   ```
   http://localhost:<port>/?token=<token>
   ```

4. Open your `.ipynb` and pick a kernel from that server. Cells now run inside the
   persistent server on the compute node.

Because the kernel lives in the Slurm job, **closing VS Code or your laptop does
not kill it.** Reconnect later by reopening the tunnel and re-selecting the same
existing server.

---

## Managing the job

```bash
squeue -u $USER        # is my Jupyter job still running? on which node?
scancel <jobid>        # stop the server and free the resources when you are done
```

The job ID is printed when you launch, and is also in the connection file and in
the job log under `~/jupyter_logs/jupyter_<jobid>.log`.

Remember:

- **Persistence lasts only while the Slurm job is alive.** When the job ends
  (time limit reached, or you `scancel` it), the server and all kernels stop and
  in-memory state is lost. Save your work (notebooks, `.h5ad` outputs) to disk as
  you go, just as you normally would.
- Please `scancel` your job when you finish for the day so you are not holding CPUs
  and memory that other students need.

---

## Quick troubleshooting

- **`~/jupyter_connect.txt` is missing** — the job has not started yet (check
  `squeue -u $USER`) or JupyterLab is still booting. Wait and re-read it.
- **Browser cannot connect / "connection refused"** — the tunnel terminal on your
  laptop is not open, or you used the wrong node/port. Re-check `~/jupyter_connect.txt`
  (the node can differ each time you launch) and re-run the tunnel command.
- **"Invalid credentials" / token page** — copy the full URL including
  `?token=...` from the connection file.
- **Port already in use** — another server is using that port on the node. Relaunch
  with a different port, e.g. `./start_jupyter.sh --port 8890`, and use the new port
  in the tunnel and URL.
