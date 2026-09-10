#!/bin/bash
#SBATCH --partition=capella
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --exclude=c145
#SBATCH --job-name=diag-curriculum
#SBATCH --output=/home/hapi039h/diag_curriculum_%j.log
#
# Verifies the reverse-curriculum reset placement. Run this BEFORE committing
# a 6-hour training run to the curriculum.

WS=/data/cat/ws/hapi039h-isaac
TASKDIR=source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/bimanual_handover

rsync -a --delete --exclude wandb --exclude logs --exclude outputs --exclude __pycache__ --exclude _isaac_sim \
  /home/hapi039h/isaaclab/source/ "$WS/isaaclab/source/"

dir_name="isaaclab_${SLURM_JOB_ID:-$$}"
cp -r "$WS/isaaclab" "$TMPDIR/$dir_name"
ln -sf /isaac-sim "$TMPDIR/$dir_name/_isaac_sim"
cp "$WS/isaac-lab-base.sif" "$TMPDIR/isaac-lab-base.sif"

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
           /isaac-sim/python.sh $TASKDIR/diag/diag_curriculum.py --num_envs 16"
echo DIAG_CURRICULUM_DONE
