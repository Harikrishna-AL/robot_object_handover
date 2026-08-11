#!/usr/bin/env bash

echo "(run_singularity.py): Called on compute node from current isaaclab directory $1 with container profile $2 and arguments ${@:3}"

#==
# Main
#==


# get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# load variables to set the Isaac Lab path on the cluster
source $SCRIPT_DIR/.env.cluster
source $SCRIPT_DIR/../.env.base

# WANDB_API_KEY, if present, lives in $CLUSTER_SIF_PATH/.env.wandb -- created
# directly on the cluster (never committed, never synced from the repo) so
# WANDB_MODE=online can actually authenticate. Without this, wandb.init()
# fails at process start (before the sim/env is even built) with
# "No API key configured", which is what happened to job 3877003.
if [ -f "$CLUSTER_SIF_PATH/.env.wandb" ]; then
    source "$CLUSTER_SIF_PATH/.env.wandb"
fi

# Log/checkpoint dirs live on node-local $TMPDIR for the entire run, NOT on
# the shared Weka filesystem. Diagnosed 2026-08-10: `singularity exec -B
# <weka-path>:...` hangs indefinitely (futex_do_wait, no error) whenever the
# bind SOURCE resides on Weka -- confirmed via a fully synthetic reproduction
# (public alpine.sif + throwaway Weka dir, no project files involved) that
# any Apptainer bind-mount with a Weka-backed source hangs on this cluster,
# while the identical bind with both source and target on local /tmp, and
# plain (non-Apptainer) reads/writes of the same Weka paths, are both fast.
# See apptainer_hang_findings.md. Fix: keep every bind-mount source local for
# the whole container lifetime, then copy results to Weka with a plain `cp`
# (not Apptainer) after the container has exited.
JOB_LOGS_DIR="$TMPDIR/local_logs/${SLURM_JOB_ID:-standalone}"
mkdir -p "$JOB_LOGS_DIR"
touch "$JOB_LOGS_DIR/.keep"

# train.py hardcodes its checkpoint/log output under the repo's own
# .../bimanual_handover/train/logs directory rather than /workspace/isaaclab/logs.
# Kept local (see note above) so the bind-mount source is never on Weka;
# copied out to persistent storage after the container exits, below.
BIMANUAL_TRAIN_LOGS_DIR="$TMPDIR/local_bimanual_train_logs/${SLURM_JOB_ID:-standalone}"
mkdir -p "$BIMANUAL_TRAIN_LOGS_DIR"

# Persistent (Weka) destinations that results get copied to via plain `cp`
# after the container exits -- never touched by Apptainer directly.
PERSIST_JOB_LOGS_DIR="$CLUSTER_ISAACLAB_DIR/logs/${SLURM_JOB_ID:-standalone}"
PERSIST_BIMANUAL_TRAIN_LOGS_DIR="$CLUSTER_ISAACLAB_DIR/bimanual_train_logs/${SLURM_JOB_ID:-standalone}"
mkdir -p "$PERSIST_JOB_LOGS_DIR" "$PERSIST_BIMANUAL_TRAIN_LOGS_DIR"

# Shader/PSO/compute caches, local-only (never Weka). These were dropped
# earlier today while debugging bind-mount count; without them every run
# recompiles GPU shaders from scratch, which plausibly explains a long
# silent pause after Isaac Sim's "app ready" with no log output but heavy
# GPU/CPU use. Restoring as node-local dirs (fresh each job, but persists
# for the lifetime of a single run, and never touches Weka).
LOCAL_CACHE_DIR="$TMPDIR/local_cache"
mkdir -p "$LOCAL_CACHE_DIR/kit" "$LOCAL_CACHE_DIR/glcache" "$LOCAL_CACHE_DIR/computecache" "$LOCAL_CACHE_DIR/ov" "$LOCAL_CACHE_DIR/pip"

# copy the temporary isaaclab directory with the latest changes to the compute node
cp -r $1 $TMPDIR
# Get the directory name
dir_name=$(basename "$1")

# the bind-mount below (-B $TMPDIR/$dir_name:/workspace/isaaclab) replaces whatever
# was baked into the image at /workspace/isaaclab, including the _isaac_sim symlink
# created at build time. Recreate it here so it exists in the mounted copy too.
ln -sf ${DOCKER_ISAACSIM_ROOT_PATH} $TMPDIR/$dir_name/_isaac_sim

# copy the pre-built .sif to the compute node (built directly with `apptainer build`,
# not via the docker push/tar workflow, so there is no .tar to extract here)
cp $CLUSTER_SIF_PATH/$2.sif $TMPDIR/$2.sif

# Background sync: copy checkpoints/logs/video from local $TMPDIR to Weka
# periodically WHILE training runs, not just after it exits. A time-limit
# kill, OOM, or node failure skips any code after `singularity exec`
# below, which would otherwise lose every checkpoint since the run started.
# Plain `cp`, never Apptainer -- safe per the Weka-bind hang note above.
sync_results() {
    cp -r "$JOB_LOGS_DIR/." "$PERSIST_JOB_LOGS_DIR/" 2>/dev/null
    cp -r "$BIMANUAL_TRAIN_LOGS_DIR/." "$PERSIST_BIMANUAL_TRAIN_LOGS_DIR/" 2>/dev/null
}
( while true; do sleep 120; sync_results; done ) &
SYNC_PID=$!
# Also catch SLURM's termination signal (SIGTERM, sent before SIGKILL on
# timeout/scancel) and normal EXIT, and force one last sync either way.
trap 'kill $SYNC_PID 2>/dev/null; sync_results' EXIT TERM

# execute command in singularity container
# NOTE: ISAACLAB_PATH is normally set in `isaaclab.sh` but we directly call the isaac-sim python because we sync the entire
# Isaac Lab directory to the compute node and remote the symbolic link to isaac-sim
#
# All bind SOURCES below are on local $TMPDIR, never on Weka (see the
# Weka-hang note above). Shader/GL/compute caches restored (previously
# dropped while debugging bind-mount count) to avoid a from-scratch shader
# recompile on every single run. --writable is not usable/needed here:
# isaac-lab-base.sif has no writable overlay partition, and none of these
# targets need one since their sources are plain local directories.
singularity exec \
    -B $TMPDIR/$dir_name:/workspace/isaaclab:rw \
    -B $JOB_LOGS_DIR:/workspace/isaaclab/logs:rw \
    -B $BIMANUAL_TRAIN_LOGS_DIR:/workspace/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/bimanual_handover/train/logs:rw \
    -B $LOCAL_CACHE_DIR/kit:${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache:rw \
    -B $LOCAL_CACHE_DIR/glcache:${DOCKER_USER_HOME}/.cache/nvidia/GLCache:rw \
    -B $LOCAL_CACHE_DIR/computecache:${DOCKER_USER_HOME}/.nv/ComputeCache:rw \
    -B $LOCAL_CACHE_DIR/ov:${DOCKER_USER_HOME}/.cache/ov:rw \
    -B $LOCAL_CACHE_DIR/pip:${DOCKER_USER_HOME}/.cache/pip:rw \
    --nv --containall $TMPDIR/$2.sif \
    bash -c 'export ISAACLAB_PATH=/workspace/isaaclab && export WANDB_MODE=online && export WANDB_DIR=/workspace/isaaclab/logs && export WANDB_API_KEY='"${WANDB_API_KEY}"' && cd /workspace/isaaclab && /isaac-sim/python.sh '"${CLUSTER_PYTHON_EXECUTABLE}"' "$@"' -- "${@:3}"
# WANDB_MODE=online (2026-08-10): re-verified that Capella compute nodes
# DO have outbound internet -- confirmed by reaching api.wandb.ai/graphql
# directly from inside a running job via srun --overlap. The original
# "compute nodes have no internet" assumption from earlier project setup
# no longer holds (or never held on this specific node pool). If wandb
# sync hangs/fails on a future run, fall back to WANDB_MODE=offline and
# `wandb sync` manually from the login node afterward.

# Container exited normally -- stop the background sync loop and do one
# final, complete sync (the EXIT trap also covers this, but do it explicitly
# here too so normal completion doesn't wait on the trap's own sleep/kill).
kill $SYNC_PID 2>/dev/null
sync_results

# if defined, remove the temporary isaaclab directory pushed when the job was submitted
if $REMOVE_CODE_COPY_AFTER_JOB; then
    rm -rf $1
fi

echo "(run_singularity.py): Return"
