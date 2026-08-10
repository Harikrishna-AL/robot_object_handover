#!/usr/bin/env bash

# apptainer/singularity are installed system-wide on Capella (/usr/bin/apptainer),
# no `module load` needed.

# create job script with compute demands
### MODIFY HERE FOR YOUR JOB ###
cat <<EOT > job.sh
#!/bin/bash

#SBATCH --partition=capella
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=4048
#SBATCH --mail-type=END
#SBATCH --mail-user=eshaanethics@gmail.com
#SBATCH --job-name="training-$(date +"%Y-%m-%dT%H:%M")"

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script
bash "$1/docker/cluster/run_singularity.sh" "$1" "$2" "${@:3}"
EOT

sbatch < job.sh
rm job.sh
