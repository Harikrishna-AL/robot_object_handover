#!/bin/bash
#SBATCH --partition=capella
#SBATCH --account=p_lasr_students
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=1-12:00:00
#SBATCH --mem=64G
#SBATCH --exclude=c145,c155,c79,c100,c111,c81,c133,c154,c46,c130
#SBATCH --job-name=xarm-finetune
#SBATCH --output=/home/hapi039h/xarm_runs/xarm_finetune_%j.log
# Fine-tune the GEN3 arm-randomization v4 policy (313.9M steps; 108/128 on wider
# arm starts) on the xArm7 task. Same recipe as Codex's run_arm_start_train.sh
# except: xArm task, Isaac Lab headless profile (Isaac Sim's full profile segfaults
# on driver 615.71), project account, and LEVEL / LR / STEPS are overridable:
#   sbatch -J xarm-curr --export=ALL,LEVEL=0,LR=3e-5,STEPS=100000000 run_xarm_finetune.sh
# LEVEL < 4 keeps the reverse curriculum on (train_resume only disables it at level 4).
set -euo pipefail
TREE=/home/hapi039h/xarm_handover
SIF=/home/hapi039h/isaac-lab-base.sif
LEVEL="${LEVEL:-4}"; LR="${LR:-1e-5}"; STEPS="${STEPS:-100000000}"
CKPT="${CKPT:-videos_out/handover_fix_20261014/arm_randomization_v4/model_313921536_arm_v4.zip}"
OUT=/home/hapi039h/xarm_runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
TASK_REL=source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/bimanual_handover
mkdir -p "$OUT/wandb"
echo "NODE=$(hostname) LEVEL=$LEVEL LR=$LR STEPS=$STEPS CKPT=$CKPT OUT=$OUT"
echo "TREE_COMMIT=$(git -C "$TREE" rev-parse --short HEAD)"

W="$TMPDIR/xarm_${SLURM_JOB_ID}"; mkdir -p "$W/tree" "$W/cache/kit" "$W/cache/gl" "$W/cache/compute" "$W/cache/ov"
tar -C "$TREE" --exclude=.git --exclude=outputs --exclude=wandb --exclude=logs -cf - . | tar -C "$W/tree" -xf -
ln -sfn /isaac-sim "$W/tree/_isaac_sim"
if [ "$(df --output=avail -B1G "$TMPDIR" | tail -1 | tr -d " ")" -gt 15 ]; then
  cp "$SIF" "$W/isaac-lab-base.sif"; SIFRUN="$W/isaac-lab-base.sif"
else
  echo "low local disk, running image from /home"; SIFRUN="$SIF"
fi

apptainer exec --nv --containall \
  -B "$W/tree":/workspace/isaaclab:rw \
  -B "$OUT":/out:rw \
  -B "$W/cache/kit":/isaac-sim/kit/cache:rw \
  -B "$W/cache/gl":/root/.cache/nvidia/GLCache:rw \
  -B "$W/cache/compute":/root/.nv/ComputeCache:rw \
  -B "$W/cache/ov":/root/.cache/ov:rw \
  "$SIFRUN" bash -lc "
    export ISAACLAB_PATH=/workspace/isaaclab BIMANUAL_LOG_ROOT=/out WANDB_MODE=offline WANDB_DIR=/out/wandb
    export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
    cd /workspace/isaaclab
    /isaac-sim/python.sh $TASK_REL/train/train_resume.py \
      --task Isaac-Bimanual-Bidirectional-xArm-v0 \
      --num_envs 512 --headless --device cuda:0 \
      --resume_checkpoint /workspace/isaaclab/$CKPT \
      --resume_learning_rate $LR \
      --resume_curriculum_level $LEVEL \
      --arm_start_randomization \
      --arm_start_nominal_fraction 0.25 \
      --arm_start_shoulder_range_rad 0.03 \
      --arm_start_elbow_range_rad 0.025 \
      --additional_timesteps $STEPS \
      --checkpoint_freq 2000
  "
echo XARM_FINETUNE_DONE
