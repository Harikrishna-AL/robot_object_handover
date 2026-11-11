#!/bin/bash
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:45:00
#SBATCH --mem=48G
#SBATCH --exclude=c130,c145,c155,c79,c100,c111
#SBATCH --job-name=handover-v2-video
#SBATCH --output=/home/hapi039h/codex_handover_20261014/ckpt_render_v2_%j.log
set -euo pipefail
TASK_ROOT=/home/hapi039h/codex_handover_20261014
TASK_REL=source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/bimanual_handover
SCRATCH_WORK="${TMPDIR:-/tmp}/codex_ckpt_render_v2_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH_WORK/tree" "$SCRATCH_WORK/cache/kit" "$SCRATCH_WORK/cache/gl" "$SCRATCH_WORK/cache/compute" "$SCRATCH_WORK/cache/ov"
mkdir -p "$TASK_ROOT/results/checkpoint_v2_33280000"
tar -C "$TASK_ROOT/tree" --exclude=.git --exclude=outputs --exclude=wandb --exclude=logs -cf - . | tar -C "$SCRATCH_WORK/tree" -xf -
cp "$TASK_ROOT/candidate/bimanual_bidirectional_env.py" "$SCRATCH_WORK/tree/$TASK_REL/bimanual_bidirectional_env.py"
cp "$TASK_ROOT/candidate/bimanual_bidirectional_env_cfg.py" "$SCRATCH_WORK/tree/$TASK_REL/bimanual_bidirectional_env_cfg.py"
ln -sfn /isaac-sim "$SCRATCH_WORK/tree/_isaac_sim"
apptainer exec --nv --containall \
  -B "$SCRATCH_WORK/tree":/workspace/isaaclab:rw \
  -B "$TASK_ROOT":/diagnostics:rw \
  -B "$TASK_ROOT/results":/results:rw \
  -B "$SCRATCH_WORK/cache/kit":/isaac-sim/kit/cache:rw \
  -B "$SCRATCH_WORK/cache/gl":/root/.cache/nvidia/GLCache:rw \
  -B "$SCRATCH_WORK/cache/compute":/root/.nv/ComputeCache:rw \
  -B "$SCRATCH_WORK/cache/ov":/root/.cache/ov:rw \
  /home/hapi039h/isaac-lab-base.sif \
  bash -lc 'export ISAACLAB_PATH=/workspace/isaaclab; cd /workspace/isaaclab; /isaac-sim/python.sh /diagnostics/render_checkpoint_v2.py --checkpoint /diagnostics/training_outputs_v2/logs/sb3/Isaac-Bimanual-Bidirectional-v0/2026-10-14_23-21-58/model_33280000_steps.zip --output /results/checkpoint_v2_33280000'
