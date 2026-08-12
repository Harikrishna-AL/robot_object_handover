# Setting up bimanual handover training on Capella (ZIH/TU Dresden)

This is a from-scratch setup guide: cloning the repo, getting the Isaac Lab
container built and running on Capella, and launching a training job. It
does **not** cover the training loop itself (reward shaping, hyperparameters,
what the metrics mean) — just the infrastructure to get a job running.

Everywhere you see a path containing `eske352h` or an email address, that's
a personal path/contact baked into this branch during initial setup — you
need to replace it with your own before anything will work for you.

## 1. Prerequisites

- A ZIH/TU Dresden HPC account with access to the Capella partition (H100
  nodes). If you don't have one yet, that's a separate request through ZIH —
  not covered here.
- SSH key registered with ZIH so `ssh <your-zih-login>@login1.capella.hpc.tu-dresden.de`
  works without a password prompt.
- A [Weights & Biases](https://wandb.ai) account (free tier is fine) if you
  want live training metrics. Not strictly required — training runs fine
  without it, you'd just lose the live dashboard.

## 2. Clone the repo and check out this branch

```bash
git clone <repo-url> robot_object_handover
cd robot_object_handover
git checkout feature/cluster-capella-setup
```

## 3. SSH config (recommended, not required)

Add an alias so every command below can just say `capella` instead of the
full hostname:

```
# ~/.ssh/config
Host capella
    HostName login1.capella.hpc.tu-dresden.de
    User <your-zih-login>
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

Test it: `ssh capella echo ok`

## 4. Allocate a ZIH workspace

Capella doesn't give you a persistent home-directory quota big enough for
Isaac Sim + a training repo. You need a "workspace" (ZIH's term for a
scratch allocation with an expiry date):

```bash
ssh capella
ws_list                        # confirm you don't already have one
ws_allocate isaaclab 90        # allocate for 90 days (max varies; extend later with ws_extend)
ws_find isaaclab                # prints the real filesystem path, e.g. /data/cat/ws/<login>-isaaclab
```

Note the printed path — that's your `<workspace>` for every step below.
Everywhere this guide (and the tracked scripts) say `/data/cat/ws/eske352h-isaaclab`,
substitute your own `<workspace>` path.

## 5. Sync the repo to your workspace

From your local machine:

```bash
rsync -avh --exclude='.git' /path/to/robot_object_handover/ capella:<workspace>/isaaclab/
```

(You'll re-run this — or `scp` individual files — any time you change code
and want it reflected in a new training job. The job-submission scripts also
copy the repo to node-local scratch at launch time, so a stale sync here
just means stale code in the *next* job, not a live problem.)

## 6. Fix the hardcoded paths

Three files have a personal path or contact baked in from initial setup —
edit each on your synced copy (either locally before syncing, or directly on
Capella with `ssh capella` + a text editor):

**`docker/cluster/.env.cluster`** — update every `/data/cat/ws/eske352h-isaaclab`
to your `<workspace>` path, and `CLUSTER_LOGIN` if you're not using the `capella`
SSH alias from step 3:

```
CLUSTER_ISAAC_SIM_CACHE_DIR=<workspace>/docker-isaac-sim
CLUSTER_ISAACLAB_DIR=<workspace>/isaaclab
CLUSTER_LOGIN=capella
CLUSTER_SIF_PATH=<workspace>/
```

**`docker/cluster/isaac-lab-base.def`** — the `%files` section near the top
hardcodes the repo's build-time source path:

```
%files
    /data/cat/ws/eske352h-isaaclab/isaaclab /workspace/isaaclab
```

Change `/data/cat/ws/eske352h-isaaclab/isaaclab` to `<workspace>/isaaclab`.

**`docker/cluster/submit_job_slurm.sh`** — has `--mail-user=eshaanethics@gmail.com`;
change to your own email (or delete the `#SBATCH --mail-type`/`--mail-user`
lines if you don't want job-completion emails).

## 7. Build the Apptainer image — directly on a Capella compute node

Capella doesn't support Docker (no daemon, no privileged builds on login
nodes), so the image is built with Apptainer directly on a compute node,
**not** via `cluster_interface.sh push` (that script assumes a local Docker
install and was not used to produce any of the working images on this
branch — treat it as unmaintained/aspirational for this project).

Grab an interactive compute node allocation and build there:

```bash
ssh capella
srun --partition=capella --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 --pty bash
cd <workspace>
apptainer build --fakeroot isaac-lab-base.sif isaaclab/docker/cluster/isaac-lab-base.def
```

This pulls `nvcr.io/nvidia/isaac-sim:4.2.0`, installs Isaac Lab's
dependencies, `stable-baselines3` + `wandb` (needed by this project's
`train.py`, not part of the stock Isaac Lab install), and pre-creates the
directories the training job will later bind-mount into. Expect this to
take a while (image pull + full dependency install) — comfortably fits
in a 2-hour interactive allocation.

When it finishes you should have `<workspace>/isaac-lab-base.sif`. Exit the
interactive session (`exit`) once it's done — you don't need the allocation
anymore.

## 8. wandb (optional, for live metrics)

If you want live training metrics on wandb.ai instead of just local
tensorboard logs, put your API key in a file **that never gets committed or
synced from your local repo** — create it directly on Capella:

```bash
ssh capella
echo 'export WANDB_API_KEY=<your-key-from-wandb.ai/authorize>' > <workspace>/.env.wandb
chmod 600 <workspace>/.env.wandb
```

`run_singularity.sh` sources this file automatically if it exists and runs
training with `WANDB_MODE=online`. If you skip this step, training still
runs fine — wandb calls will just fail at startup unless you also edit
`run_singularity.sh` to set `WANDB_MODE=offline` instead (in which case
metrics land in local tensorboard event files only, and you'd `wandb sync`
manually later from a machine that does have your API key).

## 9. Launch a training job

Jobs are submitted via `sbatch` with a small wrapper script. There's a
generic one at `docker/cluster/submit_job_slurm.sh`, but the more direct and
predictable path — used for every real run on this branch — is to write a
one-off SLURM script that calls `run_singularity.sh` yourself. Minimal
example:

```bash
#!/bin/bash
#SBATCH --partition=capella
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --time=14:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --mem-per-cpu=16192
#SBATCH --job-name="my-training-run"
#SBATCH --output=<workspace>/my_run_%j.log

CLUSTER_ISAACLAB_DIR=<workspace>/isaaclab

bash "$CLUSTER_ISAACLAB_DIR/docker/cluster/run_singularity.sh" \
    "$CLUSTER_ISAACLAB_DIR" \
    isaac-lab-base \
    --task Isaac-Bimanual-Direct-reach-v0 \
    --num_envs 2500 \
    --headless \
    --enable_cameras \
    --video --video_length 400 --video_interval 2500 \
    --max_iterations 1250 \
    --experience /workspace/isaaclab/_isaac_sim/apps/omni.isaac.sim.python.kit
```

Save it (e.g. `my_run.sh`) on Capella under `<workspace>/`, then:

```bash
ssh capella "cd <workspace> && sbatch my_run.sh"
```

`--num_envs` and `--max_iterations` are the two values you'll most likely
want to change for a first smoke test — try `--num_envs 32` and
`--max_iterations 10` first to confirm the whole pipeline runs end-to-end
before committing a full-size job. `--video_interval` is in raw environment
steps, not iterations — with N parallel envs it only advances by N per
rollout step, so scale it down for smaller `--num_envs` or it may never
fire a second recording (see the comment in `run_singularity.sh` for the
exact math if you hit this).

## 10. Check on a running job

```bash
ssh capella squeue -u <your-zih-login>          # is it running / queued / what node
ssh capella tail -f <workspace>/my_run_<jobid>.log   # live stdout/stderr
```

`run_singularity.sh` keeps all working files on the compute node's local
scratch (`$TMPDIR`) for the job's lifetime — this was a deliberate fix for
an Apptainer bug where bind-mounting a path on the shared Weka filesystem
hangs indefinitely. It syncs checkpoints/logs/video out to
`<workspace>/isaaclab/bimanual_train_logs/<jobid>/` and
`<workspace>/isaaclab/logs/<jobid>/` periodically while training runs, and
once more when the job exits (normally or via SLURM's time-limit signal) —
so a killed or crashed job doesn't lose everything since the last sync.

## 11. Common failure modes

- **Job submitted, but `squeue` shows `PD` (pending) for a long time**:
  normal — you're waiting for a free H100 node. Priority/queue depth varies.
- **`wandb.errors.errors.UsageError: No API key configured`**: `.env.wandb`
  is missing on Capella, or `WANDB_MODE=online` is set without it — see
  step 8.
- **Job hangs with high GPU/CPU usage but no log output for several
  minutes**: likely just Isaac Sim's normal startup + scene construction
  time, which scales with `--num_envs` (thousands of envs can take a couple
  of minutes before you see any `train.py` output). Not necessarily stuck —
  check whether GPU utilization is actually nonzero before assuming failure.
- **A job hangs indefinitely with no error, no progress, high system load**:
  if you've modified `run_singularity.sh` and reintroduced a bind-mount
  whose *source* path is on the shared Weka filesystem (anything under
  `<workspace>/...` used directly as a `-B` source), that will hang
  Apptainer indefinitely on this cluster. Keep all bind-mount sources on
  local `$TMPDIR` for the container's lifetime; sync to Weka with plain
  `cp` before/after, never through Apptainer.
