#!/bin/bash
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=48G
#SBATCH --exclude=c145,c155,c79,c100,c111
#SBATCH --job-name=handover-preview
#SBATCH --output=/home/hapi039h/codex_handover_20261014/preview_%j.log
set -euo pipefail
ROOT=/home/hapi039h/codex_handover_20261014
TMP_WORK="$TMPDIR/codex_handover_${SLURM_JOB_ID}"
mkdir -p "$TMP_WORK"
if ! test -d "$TMP_WORK/tree/source/extensions/omni.isaac.lab_tasks"; then
 mkdir -p "$TMP_WORK/tree"
 tar -C "$ROOT/tree" --exclude=.git --exclude=outputs --exclude=wandb -cf - . | tar -C "$TMP_WORK/tree" -xf -
fi
ln -sfn /isaac-sim "$TMP_WORK/tree/_isaac_sim"
SIFJOB=/home/hapi039h/isaac-lab-base.sif
mkdir -p "$TMP_WORK/cache/kit" "$TMP_WORK/cache/gl" "$TMP_WORK/cache/compute" "$TMP_WORK/cache/ov"
apptainer exec --nv --containall \
 -B "$TMP_WORK/tree":/workspace/isaaclab:rw \
 -B "$ROOT":/diagnostics:rw \
 -B "$ROOT/results":/results:rw \
 -B "$TMP_WORK/cache/kit":/isaac-sim/kit/cache:rw \
 -B "$TMP_WORK/cache/gl":/root/.cache/nvidia/GLCache:rw \
 -B "$TMP_WORK/cache/compute":/root/.nv/ComputeCache:rw \
 -B "$TMP_WORK/cache/ov":/root/.cache/ov:rw \
 "$SIFJOB" bash -c 'export ISAACLAB_PATH=/workspace/isaaclab; cd /workspace/isaaclab; bash /diagnostics/render_next.sh'
