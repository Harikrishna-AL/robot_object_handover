#!/bin/bash
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --exclude=c130,c145,c155,c79,c100,c111
#SBATCH --array=0-2
#SBATCH --job-name=handover-arm-heldout
#SBATCH --output=/home/hapi039h/codex_handover_20261014/arm_heldout_%A_%a.log
set -euo pipefail

TASK_ROOT=/home/hapi039h/codex_handover_20261014
TASK_REL=source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/bimanual_handover
SCRATCH_WORK="${TMPDIR:-/tmp}/codex_arm_heldout_${SLURM_JOB_ID}"
RESULT_DIR="$TASK_ROOT/results/arm_heldout_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SCRATCH_WORK/tree" "$SCRATCH_WORK/cache/kit" "$SCRATCH_WORK/cache/gl" "$SCRATCH_WORK/cache/compute" "$SCRATCH_WORK/cache/ov" "$RESULT_DIR"

CHECKPOINTS=(
  "$TASK_ROOT/results/preserved_checkpoints/model_273920000_steps.zip"
  "$TASK_ROOT/results/preserved_checkpoints/model_600002560_final_v3.zip"
  "$TASK_ROOT/results/preserved_checkpoints/model_313921536_arm_v4.zip"
)
CHECKPOINT="${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}"
test -s "$CHECKPOINT"
printf '%s\n' "$CHECKPOINT" > "$RESULT_DIR/checkpoint_used.txt"
CHECKPOINT_IN_CONTAINER="/diagnostics/${CHECKPOINT#"$TASK_ROOT"/}"

tar -C "$TASK_ROOT/tree" --exclude=.git --exclude=outputs --exclude=wandb --exclude=logs -cf - . | tar -C "$SCRATCH_WORK/tree" -xf -
cp "$TASK_ROOT/arm_randomization_v4/bimanual_bidirectional_env.py" "$SCRATCH_WORK/tree/$TASK_REL/bimanual_bidirectional_env.py"
cp "$TASK_ROOT/arm_randomization_v4/bimanual_bidirectional_env_cfg.py" "$SCRATCH_WORK/tree/$TASK_REL/bimanual_bidirectional_env_cfg.py"
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
  bash -lc "export ISAACLAB_PATH=/workspace/isaaclab; cd /workspace/isaaclab; /isaac-sim/python.sh /diagnostics/arm_randomization_v4/eval_randomized_arm_starts.py --checkpoint '$CHECKPOINT_IN_CONTAINER' --output '/results/arm_heldout_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}' --episodes 128"
