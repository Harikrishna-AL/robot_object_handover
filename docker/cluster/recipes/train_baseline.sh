#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --mem=64G
#SBATCH --job-name=baseline
#SBATCH --output=%x_%j.log
#
# Training recipe for Isaac-Bimanual-Direct-reach-v0 (Frau-Alfaro et al., arXiv:2506.16822)
# on any SLURM cluster with an NVIDIA GPU and Apptainer/Singularity. Setup: README.md next
# to this file. Cluster-specific flags (partition, account, node excludes) go on the sbatch
# command line, which overrides the header above. Submit from the repo root:
#
#   sbatch -p <partition> -A <account> docker/cluster/recipes/train_baseline.sh
#
# Environment knobs (all optional):
#   ISAACLAB_TREE  checkout of this branch           (default: the submit directory)
#   SIF_PATH       container image                   (default: $HOME/isaac-lab-base.sif)
#   OUT_DIR        where checkpoints/logs persist    (default: $HOME/runs)
#   TASK, ENVS     task id and env count             (default: paper task, 1024 envs)
#   SEED           PPO/env seed                      (default: sb3_ppo_cfg.yaml's seed)
#   EXPERIENCE     Kit app file; empty = headless profile (default). The full profile
#                  (/isaac-sim/apps/omni.isaac.sim.python.kit) segfaults on newer drivers.
#   SMOKE_ITERS    e.g. 20 for a quick plumbing test; NEVER for a real run (see below)
#
# ---------------------------------------------------------------------------
# THE IMPORTANT PART: do NOT pass --max_iterations for a real run.
# ---------------------------------------------------------------------------
# train.py overrides the config's n_timesteps whenever that flag is present:
#
#     if args_cli.max_iterations is not None:
#         agent_cfg["n_timesteps"] = args_cli.max_iterations * n_steps * num_envs
#
# sb3_ppo_cfg.yaml asks for n_timesteps: 6e8. Running with
# "--max_iterations 1250 --num_envs 256" silently trains for
# 1250*16*256 = 5.12M steps, i.e. 0.85% of the intended budget -- and the
# baseline does not leave the floor until ~10M steps, so it looks completely
# broken (flat reward ~-12, zero successes) while being perfectly fine.
#
# Measured curve at 1024 envs (job 4211348):
#     5M steps   reward  -8
#    11M steps   reward +10    <- first successes
#    50M steps   reward +146
#   150M steps   reward +1050
#   300M steps   reward +1190  (still climbing at 24h limit)
#
# ~3300-3500 steps/s at 1024 envs without video on one H100 => 6e8 steps is ~48h.
# No --video: rendering roughly halves throughput; render from a checkpoint afterwards.
set -euo pipefail

TASK="${TASK:-Isaac-Bimanual-Direct-reach-v0}"
ENVS="${ENVS:-1024}"
TREE="${ISAACLAB_TREE:-${SLURM_SUBMIT_DIR:-$PWD}}"
SIF="${SIF_PATH:-$HOME/isaac-lab-base.sif}"
EXPERIENCE="${EXPERIENCE:-}"
SCRATCH="${TMPDIR:-/tmp}"
TASK_REL=source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/bimanual_handover
TRAINPY=$TASK_REL/train/train.py

if command -v apptainer >/dev/null; then RUNTIME=apptainer
elif command -v singularity >/dev/null; then RUNTIME=singularity
else echo "FATAL: neither apptainer nor singularity on PATH (try: module load apptainer)"; exit 1; fi

echo "NODE=$(hostname) TASK=$TASK ENVS=$ENVS SEED=${SEED:-cfg} TREE=$TREE SIF=$SIF RUNTIME=$RUNTIME START=$(date)"
test -f "$TREE/$TRAINPY" || { echo "FATAL: $TREE is not a checkout of this branch (submit from the repo root or set ISAACLAB_TREE)"; exit 1; }
test -f "$SIF" || { echo "FATAL: image missing: $SIF (see docker/cluster/recipes/README.md, or set SIF_PATH)"; exit 1; }
nvidia-smi -L || echo "WARN: nvidia-smi failed on the host; continuing"

# Stage on node-local scratch with job-scoped names ($TMPDIR can be shared between jobs).
JOB="${SLURM_JOB_ID:-$$}"
W="$SCRATCH/isaaclab_baseline_$JOB"
mkdir -p "$W/tree" "$W/cache/kit" "$W/cache/glcache" "$W/cache/computecache" "$W/cache/ov"
tar -C "$TREE" --exclude=.git --exclude=logs --exclude=outputs --exclude=wandb -cf - . | tar -C "$W/tree" -xf -
ln -sfn /isaac-sim "$W/tree/_isaac_sim"
LOGDIR="$W/tree/$TASK_REL/train/logs"
mkdir -p "$LOGDIR" "$W/tree/logs"
# Copy the image to local disk if there is room (faster, avoids network-filesystem hangs).
if [ "$(df --output=avail -B1G "$SCRATCH" | tail -1 | tr -d ' ')" -gt 20 ]; then
  cp -f "$SIF" "$W/image.sif"; SIFRUN="$W/image.sif"
else
  echo "low scratch space, running the image in place"; SIFRUN="$SIF"
fi

OUT="${OUT_DIR:-$HOME/runs}/${SLURM_JOB_NAME:-baseline}_$JOB"
mkdir -p "$OUT"
echo "OUT=$OUT"

# Rescue checkpoints every 15 min so a job killed at its time limit loses at most that.
# Keep the newest few, sorted by the STEP NUMBER in the filename (copying resets mtimes).
prune() {
  find "$OUT" -name 'model_*_steps.zip' 2>/dev/null \
    | sed 's/.*model_\([0-9]*\)_steps\.zip/\1 &/' \
    | sort -rn | tail -n +6 | cut -d' ' -f2- | xargs -r rm -f
}
( while true; do sleep 900; cp -r "$LOGDIR/." "$OUT/" 2>/dev/null || true; prune; done ) &
SYNC_PID=$!
cleanup() { cp -r "$LOGDIR/." "$OUT/" 2>/dev/null || true; prune; kill $SYNC_PID 2>/dev/null || true; rm -rf "$W"; }
trap cleanup EXIT

# wandb: online only if a key is provided (env, or $HOME/.env.wandb with WANDB_API_KEY=...).
WANDB_API_KEY="${WANDB_API_KEY:-}"
[ -f "$HOME/.env.wandb" ] && source "$HOME/.env.wandb"
WANDB_MODE=offline; [ -n "$WANDB_API_KEY" ] && WANDB_MODE=online

EXTRA=""
[ -n "${SEED:-}" ] && EXTRA="$EXTRA --seed $SEED"
[ -n "$EXPERIENCE" ] && EXTRA="$EXTRA --experience $EXPERIENCE"
[ -n "${SMOKE_ITERS:-}" ] && EXTRA="$EXTRA --max_iterations $SMOKE_ITERS" && echo "SMOKE TEST: $SMOKE_ITERS iterations only"

RC=0
stdbuf -oL -eL $RUNTIME exec --nv --containall \
  -B "$W/tree":/workspace/isaaclab:rw \
  -B "$W/cache/kit":/isaac-sim/kit/cache:rw \
  -B "$W/cache/glcache":/root/.cache/nvidia/GLCache:rw \
  -B "$W/cache/computecache":/root/.nv/ComputeCache:rw \
  -B "$W/cache/ov":/root/.cache/ov:rw \
  "$SIFRUN" \
  bash -c "export ISAACLAB_PATH=/workspace/isaaclab OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONUNBUFFERED=1 && \
           export WANDB_MODE=$WANDB_MODE WANDB_DIR=/workspace/isaaclab/$TASK_REL/train/logs WANDB_API_KEY='${WANDB_API_KEY}' && \
           export WANDB_NAME='${SLURM_JOB_NAME:-baseline}_$JOB' && \
           cd /workspace/isaaclab && /isaac-sim/python.sh $TRAINPY \
             --task $TASK --num_envs $ENVS --headless $EXTRA" || RC=$?
echo "TRAIN_EXIT=$RC PERSISTED_TO=$OUT END=$(date)"
exit $RC
