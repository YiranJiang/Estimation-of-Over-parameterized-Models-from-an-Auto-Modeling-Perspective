#!/bin/bash

#SBATCH --time=00:20:00
#SBATCH --array=1-150

module load R

# Calculate a and b based on the task ID
a=$(( (SLURM_ARRAY_TASK_ID - 1) / 50 + 1 ))
b=$(( (SLURM_ARRAY_TASK_ID - 1) % 50 + 1 ))

Rscript mnm-simulation.R $a $b
