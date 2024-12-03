Reproduction Steps

1. Run Simulation Experiments:

Rscript mnm-simulation.R $a $b

 - a: Index of the experiment (1-3)
 - b: Index of the repetitions (1-50), each containing 10 datasets


Alternative -- SLURM Job Script (modify the file if required):

sbatch myjob-array.sh

2. Summary:

Rscript summary.R