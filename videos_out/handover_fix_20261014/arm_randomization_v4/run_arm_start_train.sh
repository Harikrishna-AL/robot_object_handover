#!/bin/bash
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G
#SBATCH --exclude=c130,c145,c155,c79,c100,c111
#SBATCH --job-name=handover-arm-start-v4
#SBATCH --output=/home/hapi039h/codex_handover_20261014/arm_start_train_%j.log
set -euo pipefail

TASK_ROOT=/home/hapi039h/codex_handover_20261014
TASK_REL=source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/bimanual_handover
SCRATCH_WORK="${TMPDIR:-/tmp}/codex_arm_start_v4_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH_WORK/tree" "$SCRATCH_WORK/cache/kit" "$SCRATCH_WORK/cache/gl" "$SCRATCH_WORK/cache/compute" "$SCRATCH_WORK/cache/ov"
mkdir -p "$TASK_ROOT/training_outputs_v4_arm/wandb"
tar -C "$TASK_ROOT/tree" --exclude=.git --exclude=outputs --exclude=wandb --exclude=logs -cf - . | tar -C "$SCRATCH_WORK/tree" -xf -
cp "$TASK_ROOT/arm_randomization_v4/bimanual_bidirectional_env.py" "$SCRATCH_WORK/tree/$TASK_REL/bimanual_bidirectional_env.py"
cp "$TASK_ROOT/arm_randomization_v4/bimanual_bidirectional_env_cfg.py" "$SCRATCH_WORK/tree/$TASK_REL/bimanual_bidirectional_env_cfg.py"
cp "$TASK_ROOT/arm_randomization_v4/train_resume.py" "$SCRATCH_WORK/tree/$TASK_REL/train/train_resume.py"
ln -sfn /isaac-sim "$SCRATCH_WORK/tree/_isaac_sim"

apptainer exec --nv --containall \
  -B "$SCRATCH_WORK/tree":/workspace/isaaclab:rw \
  -B "$TASK_ROOT/training_outputs_v4_arm":/training_outputs_v4_arm:rw \
  -B "$TASK_ROOT/results/preserved_checkpoints":/preserved_checkpoints:ro \
  -B "$SCRATCH_WORK/cache/kit":/isaac-sim/kit/cache:rw \
  -B "$SCRATCH_WORK/cache/gl":/root/.cache/nvidia/GLCache:rw \
  -B "$SCRATCH_WORK/cache/compute":/root/.nv/ComputeCache:rw \
  -B "$SCRATCH_WORK/cache/ov":/root/.cache/ov:rw \
  /home/hapi039h/isaac-lab-base.sif \
  bash -lc '
    export ISAACLAB_PATH=/workspace/isaaclab
    export BIMANUAL_LOG_ROOT=/training_outputs_v4_arm
    export WANDB_MODE=offline
    export WANDB_DIR=/training_outputs_v4_arm/wandb
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS=8
    cd /workspace/isaaclab
    /isaac-sim/python.sh source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/bimanual_handover/train/train_resume.py \
      --task Isaac-Bimanual-Bidirectional-v0 \
      --num_envs 512 \
      --headless \
      --device cuda:0 \
      --resume_checkpoint /preserved_checkpoints/model_273920000_steps.zip \
      --resume_learning_rate 1e-5 \
      --resume_curriculum_level 4 \
      --arm_start_randomization \
      --arm_start_nominal_fraction 0.25 \
      --arm_start_shoulder_range_rad 0.03 \
      --arm_start_elbow_range_rad 0.025 \
      --additional_timesteps 40000000 \
      --checkpoint_freq 2000 \
      --experience /isaac-sim/apps/omni.isaac.sim.python.kit
  '
