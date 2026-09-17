#!/bin/bash
#SBATCH --partition=capella
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --mem-per-cpu=13000
#SBATCH --job-name=baseline
#SBATCH --output=%x_%j.log
#
# Working training recipe for Isaac-Bimanual-Direct-reach-v0 on Capella.
#
# ---------------------------------------------------------------------------
# THE IMPORTANT PART: do NOT pass --max_iterations.
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
#     5M steps   reward  -8    success 0
#    11M steps   reward +10    success 1.4e-4   <- first successes
#    50M steps   reward +146
#   150M steps   reward +1050
#   300M steps   reward +1190  success 3.2e-3   (still climbing at 24h limit)
#
# Other settings that matter:
#   --num_envs 1024   matches the paper (Frau-Alfaro et al., arXiv:2506.16822)
#   no --video        rendering roughly halves throughput; render separately
#                     from a checkpoint afterwards
#   ~3500 steps/s at 1024 envs without video => 6e8 steps is about 48h
set -euo pipefail

TASK="${TASK:-Isaac-Bimanual-Direct-reach-v0}"
ENVS="${ENVS:-1024}"
TREE="${ISAACLAB_TREE:-$HOME/isaaclab}"
SIF="${SIF_PATH:-$HOME/isaac-lab-base.sif}"
TRAINPY=source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/bimanual_handover/train/train.py

echo "NODE=$(hostname) TASK=$TASK ENVS=$ENVS TREE=$TREE SIF=$SIF"
test -d "$TREE" || { echo "FATAL: tree missing: $TREE"; exit 1; }
test -f "$SIF"  || { echo "FATAL: sif missing: $SIF (build with docker/cluster/isaac-lab-base.def)"; exit 1; }

# Everything is staged on node-local $TMPDIR. Names are job-scoped because
# $TMPDIR is shared between jobs on the same node here -- an unscoped filename
# collides with a concurrent job and the copy fails.
dir_name="isaaclab_${SLURM_JOB_ID:-$$}"
cp -r "$TREE" "$TMPDIR/$dir_name"
ln -sf /isaac-sim "$TMPDIR/$dir_name/_isaac_sim"
LOGDIR="$TMPDIR/$dir_name/${TRAINPY%/train.py}/logs"
mkdir -p "$LOGDIR"
SIFJOB="$TMPDIR/isaac-lab-base_${SLURM_JOB_ID:-$$}.sif"
cp -f "$SIF" "$SIFJOB"

CACHE="$TMPDIR/cache_${SLURM_JOB_ID:-$$}"
mkdir -p "$CACHE/kit" "$CACHE/glcache" "$CACHE/computecache" "$CACHE/ov"

OUT="${OUT_DIR:-$HOME/runs}/${SLURM_JOB_ID:-standalone}"
mkdir -p "$OUT"

# Rescue checkpoints periodically: a 48h job killed at the time limit would
# otherwise lose everything on node-local scratch. Keep the newest few, sorted
# by the STEP NUMBER in the filename -- sorting by mtime does not work because
# the copy below resets every file's mtime on each pass.
prune() {
  find "$OUT" -name 'model_*_steps.zip' 2>/dev/null \
    | sed 's/.*model_\([0-9]*\)_steps\.zip/\1 &/' \
    | sort -rn | tail -n +6 | cut -d' ' -f2- | xargs -r rm -f
}
( while true; do sleep 900; cp -r "$LOGDIR/." "$OUT/" 2>/dev/null || true; prune; done ) &
SYNC_PID=$!
trap 'cp -r "$LOGDIR/." "$OUT/" 2>/dev/null || true; kill $SYNC_PID 2>/dev/null || true' EXIT TERM

# Optional: put WANDB_API_KEY in $HOME/.env.wandb to enable online logging.
WANDB_API_KEY=""
[ -f "$HOME/.env.wandb" ] && source "$HOME/.env.wandb"

stdbuf -oL -eL apptainer exec --nv --containall \
  -B "$TMPDIR/$dir_name":/workspace/isaaclab:rw \
  -B "$CACHE/kit":/isaac-sim/kit/cache:rw \
  -B "$CACHE/glcache":/root/.cache/nvidia/GLCache:rw \
  -B "$CACHE/computecache":/root/.nv/ComputeCache:rw \
  -B "$CACHE/ov":/root/.cache/ov:rw \
  "$SIFJOB" \
  bash -c "export ISAACLAB_PATH=/workspace/isaaclab && export WANDB_MODE=online && \
           export WANDB_DIR=/workspace/isaaclab/logs && export WANDB_API_KEY='${WANDB_API_KEY}' && \
           export WANDB_NAME='${SLURM_JOB_NAME:-baseline}_${SLURM_JOB_ID:-0}' && \
           cd /workspace/isaaclab && /isaac-sim/python.sh $TRAINPY \
             --task $TASK --num_envs $ENVS --headless \
             --experience /isaac-sim/apps/omni.isaac.sim.python.kit" || true

cp -r "$LOGDIR/." "$OUT/" 2>/dev/null || true
prune
echo "PERSISTED_TO=$OUT"
