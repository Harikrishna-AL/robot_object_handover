#!/bin/bash
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --exclude=c130,c145,c155,c79,c100,c111
#SBATCH --job-name=handover-arm-v4-var-eval
#SBATCH --output=/home/hapi039h/codex_handover_20261014/arm_v4_variation_render_%j.log
set -euo pipefail

TASK_ROOT=/home/hapi039h/codex_handover_20261014
TASK_REL=source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/bimanual_handover
SCRATCH_WORK="${TMPDIR:-/tmp}/codex_arm_v4_variation_${SLURM_JOB_ID}"
RESULT_DIR="$TASK_ROOT/results/arm_v4_position_variations_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH_WORK/tree" "$SCRATCH_WORK/cache/kit" "$SCRATCH_WORK/cache/gl" "$SCRATCH_WORK/cache/compute" "$SCRATCH_WORK/cache/ov" "$RESULT_DIR"
CHECKPOINT="$TASK_ROOT/results/preserved_checkpoints/model_313921536_arm_v4.zip"
test -s "$CHECKPOINT"
printf '%s\n' "$CHECKPOINT" > "$RESULT_DIR/checkpoint_used.txt"
CHECKPOINT_IN_CONTAINER="/diagnostics/${CHECKPOINT#"$TASK_ROOT"/}"

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
  bash -lc "export ISAACLAB_PATH=/workspace/isaaclab; cd /workspace/isaaclab; /isaac-sim/python.sh /diagnostics/render_position_variations.py --checkpoint '$CHECKPOINT_IN_CONTAINER' --output '/results/arm_v4_position_variations_${SLURM_JOB_ID}'"
