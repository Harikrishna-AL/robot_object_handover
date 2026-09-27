#!/bin/bash
#SBATCH --partition=capella
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --exclude=c145
#SBATCH --job-name=eval-per-role
#SBATCH --output=/home/hapi039h/eval_per_role_%j.log
#
# Verifies per-episode giver/taker randomisation: balance, grip in BOTH roles,
# the context scalar agreeing with the actual role, and the object starting in
# the giver's hand. Run BEFORE committing a long training run to the role swap.
#
# set -euo pipefail is deliberate. The first version of this script had no error
# handling: on 2026-09-11 every setup step failed (workspace not mounted on the
# compute node), the container never even started, and the script still printed
# its DONE line and exited 0 -- SLURM reported COMPLETED for a job that did
# nothing. A job that claims success after failing is worse than one that crashes.
set -euo pipefail

# Everything is sourced from /home. Diagnosed 2026-09-11: the workspace
# filesystems (/data/cat AND /data/horse) are not mounted on Capella compute
# nodes -- probed three nodes (c3, c156 x2), none could see either, while /home
# was visible on all three. /home is NFS and always available.
TREE=/home/hapi039h/isaaclab
SIF=/home/hapi039h/isaac-lab-base.sif
TASKDIR=source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/bimanual_handover

# Point CKPT at a SNAPSHOT, not at a live icl_runs path. long_run.sh's rescue
# loop prune() keeps only the 4 newest checkpoints, so a path that exists at
# submit time can be gone by the time the job starts -- job 4307112 failed
# exactly that way. Copy it to /home/hapi039h/eval_ckpts/ first.
CKPT="${CKPT:?set CKPT to a snapshotted checkpoint, e.g. /home/hapi039h/eval_ckpts/roleswap_model_X_steps.zip}"
test -f "$CKPT" || { echo "FATAL: checkpoint missing: $CKPT"; exit 1; }
EXTRA_ARGS=""
[ "${FIXED_ROLES:-0}" = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --fixed_roles"
[ "${STOCHASTIC:-0}" = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --stochastic"
echo "NODE=$(hostname)  CKPT=$CKPT  EXTRA_ARGS=[$EXTRA_ARGS]"
test -d "$TREE" || { echo "FATAL: tree missing: $TREE"; exit 1; }
test -f "$SIF"  || { echo "FATAL: sif missing: $SIF";  exit 1; }

dir_name="isaaclab_${SLURM_JOB_ID:-$$}"
cp -r "$TREE" "$TMPDIR/$dir_name"
ln -sf /isaac-sim "$TMPDIR/$dir_name/_isaac_sim"
cp "$SIF" "$TMPDIR/isaac-lab-base.sif"
cp "$CKPT" "$TMPDIR/$dir_name/ckpt.zip"

CACHE="$TMPDIR/local_cache"
mkdir -p "$CACHE/kit" "$CACHE/glcache" "$CACHE/computecache" "$CACHE/ov"

stdbuf -oL -eL apptainer exec --nv --containall \
  -B "$TMPDIR/$dir_name":/workspace/isaaclab:rw \
  -B "$CACHE/kit":/isaac-sim/kit/cache:rw \
  -B "$CACHE/glcache":/root/.cache/nvidia/GLCache:rw \
  -B "$CACHE/computecache":/root/.nv/ComputeCache:rw \
  -B "$CACHE/ov":/root/.cache/ov:rw \
  "$TMPDIR/isaac-lab-base.sif" \
  bash -c "export ISAACLAB_PATH=/workspace/isaaclab && cd /workspace/isaaclab && \
           /isaac-sim/python.sh $TASKDIR/diag/eval_per_role.py --num_envs 64 --steps 4000 --checkpoint /workspace/isaaclab/ckpt.zip $EXTRA_ARGS"
echo EVAL_PER_ROLE_WRAPPER_DONE
