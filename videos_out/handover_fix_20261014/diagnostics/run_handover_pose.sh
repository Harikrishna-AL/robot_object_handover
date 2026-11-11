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
#SBATCH --job-name=handover-pose
#SBATCH --output=/home/hapi039h/codex_handover_20261014/pose_%j.log
set -euo pipefail
ROOT=/home/hapi039h/codex_handover_20261014
mkdir -p "$TMPDIR/tree"
tar -C "$ROOT/tree" --exclude=.git --exclude=outputs --exclude=wandb -cf - . | tar -C "$TMPDIR/tree" -xf -
ln -sfn /isaac-sim "$TMPDIR/tree/_isaac_sim"
cp /home/hapi039h/isaac-lab-base.sif "$TMPDIR/sim.sif"
mkdir -p "$TMPDIR/cache/kit" "$TMPDIR/cache/gl" "$TMPDIR/cache/compute" "$TMPDIR/cache/ov"
apptainer exec --nv --containall \
 -B "$TMPDIR/tree":/workspace/isaaclab:rw \
 -B "$ROOT":/diagnostics:rw \
 -B "$ROOT/results":/results:rw \
 -B "$TMPDIR/cache/kit":/isaac-sim/kit/cache:rw \
 -B "$TMPDIR/cache/gl":/root/.cache/nvidia/GLCache:rw \
 -B "$TMPDIR/cache/compute":/root/.nv/ComputeCache:rw \
 -B "$TMPDIR/cache/ov":/root/.cache/ov:rw \
 "$TMPDIR/sim.sif" bash -c 'export ISAACLAB_PATH=/workspace/isaaclab; cd /workspace/isaaclab; /isaac-sim/python.sh /diagnostics/diag_handover_pose.py --num_envs 32 --out /results'
