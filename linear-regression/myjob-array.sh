#!/bin/bash

#SBATCH --array=1-900
#SBATCH --time=00:15:00
#SBATCH --output=/dev/null

# Generic output file - individual job outputs will be handled within the script




# Define the arrays for num_nonzero_beta and sigma
alpha_values=(0.3 0.6 0.9)
tau_values=(0.3 1 3)

# Calculate alpha, tau, and k based on SLURM_ARRAY_TASK_ID
index=$((SLURM_ARRAY_TASK_ID - 1))

alpha_index=$(((index / 100) % 3))
alpha=${alpha_values[$alpha_index]}

tau_index=$(((index / 300) % 3))
tau=${tau_values[$tau_index]}

k=$((index % 100 + 1))

# Load the R module (if required)
module load R

# Create a unique filename for the output
output_file="./output/output_alpha_${alpha}_k_${k}_tau_${tau}.txt"

# Run the R script with the calculated parameters and redirect output
Rscript linear_regression.R $alpha $k $tau > $output_file
