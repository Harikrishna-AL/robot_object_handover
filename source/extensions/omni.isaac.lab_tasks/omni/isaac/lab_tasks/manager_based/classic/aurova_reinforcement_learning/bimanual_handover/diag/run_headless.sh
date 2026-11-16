#!/bin/bash
#SBATCH --partition=capella
#SBATCH --account=p_lasr_students
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --exclude=c145,c155,c79,c100,c111,c81,c133,c154,c46
#SBATCH --job-name=headless
#SBATCH --output=/home/hapi039h/headless_%j.log
# Generic runner: copies TREE to node-local disk and runs CMD inside the Isaac Lab container.
# Uses Isaac Lab's headless profile (default when --experience is not given): after the
# Nov 2026 driver update (580.159 -> 615.71) Isaac Sim's full profile segfaults while the
# viewport initialises Hydra (jobs 4589288, 4589368, probe 4589489 case C).
set -euo pipefail
TREE="${TREE:-/home/hapi039h/xarm_handover}"; SIF=/home/hapi039h/isaac-lab-base.sif
CMD="${CMD:?set CMD}"
echo "NODE=$(hostname) TREE=$TREE"; echo "CMD=$CMD"
dir_name="isaaclab_${SLURM_JOB_ID}"; cp -r "$TREE" "$TMPDIR/$dir_name"; ln -sf /isaac-sim "$TMPDIR/$dir_name/_isaac_sim"
# Node-local copy is faster, but some nodes' /tmp is too small for the 7.4 GB image
# (job 4589718 on c46: "No space left on device"). Fall back to running it from /home.
if [ "$(df --output=avail -B1G "$TMPDIR" | tail -1 | tr -d " ")" -gt 15 ]; then
  cp "$SIF" "$TMPDIR/isaac-lab-base.sif"; SIFRUN="$TMPDIR/isaac-lab-base.sif"
else
  echo "low local disk, running image from /home"; SIFRUN="$SIF"
fi
CACHE="$TMPDIR/local_cache"; mkdir -p "$CACHE/kit" "$CACHE/glcache" "$CACHE/computecache" "$CACHE/ov"
stdbuf -oL -eL apptainer exec --nv --containall -B "$TMPDIR/$dir_name":/workspace/isaaclab:rw \
  -B "$CACHE/kit":/isaac-sim/kit/cache:rw -B "$CACHE/glcache":/root/.cache/nvidia/GLCache:rw \
  -B "$CACHE/computecache":/root/.nv/ComputeCache:rw -B "$CACHE/ov":/root/.cache/ov:rw \
  "$SIFRUN" bash -c "export ISAACLAB_PATH=/workspace/isaaclab WANDB_MODE=offline && cd /workspace/isaaclab && $CMD"
echo HEADLESS_DONE
