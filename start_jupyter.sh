#!/usr/bin/env bash
#
# start_jupyter.sh — launch a PERSISTENT JupyterLab server inside a Slurm job.
#
# Run this ON THE LOGIN NODE (core.cluster.france-bioinformatique.fr).
# It submits a Slurm job on the `fast` partition that activates the `single_cell`
# conda env and starts `jupyter lab` bound to the compute node. Because the server
# lives inside the Slurm job (NOT inside your SSH / VS Code session), it keeps
# running when you close your laptop. Reconnect any time while the job is alive.
#
# Usage:
#   ./start_jupyter.sh [--cpus N] [--mem 64G] [--time HH:MM:SS] [--port 8888]
#                      [--account ACC] [--partition fast] [--dir /path/to/notebooks]
#
# Every flag also has an environment-variable equivalent, e.g.:
#   CPUS=16 MEM=128G TIME=12:00:00 PORT=9999 ./start_jupyter.sh
#
# After it prints "Submitted", wait ~10-60 s for the job to start, then read the
# connection file it writes:   cat ~/jupyter_connect.txt
# ---------------------------------------------------------------------------

set -euo pipefail

# ------------------------- defaults (override via env or flags) -------------
ACCOUNT="${ACCOUNT:-tp_2630_ubordeaux_neuromics_184418}"
PARTITION="${PARTITION:-fast}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
TIME="${TIME:-08:00:00}"
PORT="${PORT:-8888}"
NOTEBOOK_DIR="${NOTEBOOK_DIR:-$HOME}"

LOGIN_HOST="core.cluster.france-bioinformatique.fr"
CONDA_SH="/shared/software/miniconda/etc/profile.d/conda.sh"
ENV_PATH="/shared/projects/tp_2630_ubordeaux_neuromics_184418/envs/single_cell"

# Logs and the connection-info file MUST live on the SHARED filesystem so the
# compute node can write them and the login node can read them. $HOME is shared
# on IFB Core; node-local /tmp is NOT (a job there cannot be reached this way).
LOG_DIR="${LOG_DIR:-$HOME/jupyter_logs}"
CONNECT_FILE="${CONNECT_FILE:-$HOME/jupyter_connect.txt}"

# ------------------------------- flag parsing -------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpus)      CPUS="$2"; shift 2 ;;
    --mem)       MEM="$2"; shift 2 ;;
    --time)      TIME="$2"; shift 2 ;;
    --port)      PORT="$2"; shift 2 ;;
    --account)   ACCOUNT="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --dir)       NOTEBOOK_DIR="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//' | sed '/^!/d'
      exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Try: $0 --help" >&2
      exit 1 ;;
  esac
done

# ------------------------------- sanity checks ------------------------------
command -v sbatch >/dev/null 2>&1 || {
  echo "ERROR: sbatch not found. Are you on the IFB login node?" >&2; exit 1; }
[[ -f "$CONDA_SH" ]] || {
  echo "ERROR: conda.sh not found at $CONDA_SH" >&2; exit 1; }
[[ -d "$ENV_PATH" ]] || {
  echo "ERROR: single_cell env not found at $ENV_PATH" >&2; exit 1; }

mkdir -p "$LOG_DIR"

# A token protects your server (anyone who reaches the port needs it). Generate a
# fresh random one each launch. We create it on the login node so this file could
# print it too, but the job writes the final connection info (it knows the node).
TOKEN="$(python3 -c 'import secrets; print(secrets.token_hex(24))' 2>/dev/null \
         || openssl rand -hex 24 2>/dev/null \
         || head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')"

JOB_OUT="$LOG_DIR/jupyter_%j.log"

# --------------------------- submit the Slurm job ---------------------------
# The heredoc below is the job script. It runs ON THE COMPUTE NODE. We pass the
# port/token/paths through the environment via --export so nothing is hard-coded.
JOBID="$(
ACCOUNT="$ACCOUNT" PARTITION="$PARTITION" \
sbatch --parsable \
  --job-name="jupyter" \
  --account="$ACCOUNT" \
  --partition="$PARTITION" \
  --cpus-per-task="$CPUS" \
  --mem="$MEM" \
  --time="$TIME" \
  --output="$JOB_OUT" \
  --error="$JOB_OUT" \
  --export=ALL,PORT="$PORT",JTOKEN="$TOKEN",CONNECT_FILE="$CONNECT_FILE",NOTEBOOK_DIR="$NOTEBOOK_DIR",CONDA_SH="$CONDA_SH",ENV_PATH="$ENV_PATH",LOGIN_HOST="$LOGIN_HOST" \
  <<'JOBEOF'
#!/usr/bin/env bash
set -euo pipefail

NODE="$(hostname -s)"
USER_NAME="${USER:-$(whoami)}"

# --- activate the read-only single_cell env (activate by full path; the env is
#     not on the default envs_dirs on IFB) ---
source "$CONDA_SH"
conda activate "$ENV_PATH"

TUNNEL="ssh -N -L ${PORT}:${NODE}:${PORT} -J ${USER_NAME}@${LOGIN_HOST} ${USER_NAME}@${NODE}"
URL="http://localhost:${PORT}/lab?token=${JTOKEN}"

# --- write the connection info to a predictable file on the shared FS ---
cat > "$CONNECT_FILE" <<INFO
============================================================
 Persistent JupyterLab server — connection info
 (job ${SLURM_JOB_ID:-?}, written $(date '+%Y-%m-%d %H:%M:%S'))
============================================================
NODE  : ${NODE}
PORT  : ${PORT}
TOKEN : ${JTOKEN}

STEP 1 — On your LAPTOP, open a terminal and run this (keep it open):

  ${TUNNEL}

STEP 2 — In your browser on the laptop, open:

  ${URL}

Close your laptop any time. Your notebook and its running cells stay alive on
the compute node while this Slurm job runs. To get back in: re-run the STEP 1
tunnel command and reopen the STEP 2 URL — you land on the SAME live kernels.

VS Code alternative (instead of the browser): command palette ->
"Jupyter: Specify Jupyter Server for Connections..." -> "Existing" -> paste:
  http://localhost:${PORT}/?token=${JTOKEN}
(the tunnel from STEP 1 must be running).

Check the job : squeue -u ${USER_NAME}
Stop  the job: scancel ${SLURM_JOB_ID:-<jobid>}
============================================================
INFO

# --- also echo it into the job log so it is visible there too ---
echo "===================== JUPYTER CONNECTION INFO ====================="
cat "$CONNECT_FILE"
echo "==================================================================="
echo "Starting jupyter lab on ${NODE}:${PORT} ..."

# --- launch the server; runs in the foreground so the job stays alive ---
cd "$NOTEBOOK_DIR"
exec jupyter lab \
  --no-browser \
  --ip=0.0.0.0 \
  --port="${PORT}" \
  --ServerApp.token="${JTOKEN}" \
  --ServerApp.root_dir="${NOTEBOOK_DIR}" \
  --ServerApp.allow_origin='*'
JOBEOF
)"

# ------------------------------ user instructions ---------------------------
LOG_FILE="$LOG_DIR/jupyter_${JOBID}.log"

cat <<MSG

Submitted persistent JupyterLab job: ${JOBID}
  partition=${PARTITION}  cpus=${CPUS}  mem=${MEM}  time=${TIME}  port=${PORT}
  job log       : ${LOG_FILE}
  connection file: ${CONNECT_FILE}

NEXT STEPS
----------
1. Wait for the job to start (usually 10-60 s). Check it with:
     squeue -u \$USER
   When state shows 'R' (running), continue.

2. Read the connection info the job wrote (node, port, token, tunnel, URL):
     cat ${CONNECT_FILE}
   (If it is not there yet, the job has not started — wait and retry.)

3. Follow the two steps printed in that file:
     - On your LAPTOP: run the 'ssh -N -L ...' tunnel command (leave it open).
     - In your browser : open the http://localhost:${PORT}/lab?token=... URL.
   Or in VS Code: "Jupyter: Specify Jupyter Server for Connections..." ->
   "Existing" -> paste the http://localhost:${PORT}/?token=... URL.

RECONNECT AFTER CLOSING YOUR LAPTOP
-----------------------------------
The server and your running kernels stay alive as long as this Slurm job runs.
Just re-run the SAME tunnel command and reopen the SAME URL. Nothing is lost.

STOP IT WHEN DONE
-----------------
  scancel ${JOBID}
Persistence lasts only while the job is alive (max ${TIME}, or until you cancel).

MSG
