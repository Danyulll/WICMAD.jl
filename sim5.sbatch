#!/bin/bash

#SBATCH --job-name=wicmad_sim5
#SBATCH --account=def-dsteph
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --mem=180G                 # adjust as needed (or use --mem=0 to grab all memory)
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.krasnov@mail.mcgill.ca

module load julia/1.11.3

# Threading + headless plotting + avoid oversubscription
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export JULIA_EXCLUSIVE=1
export OMP_NUM_THREADS=1
export GKSwstype=100

# (Optional, recommended on Narval / to avoid HOME quota)
# export JULIA_DEPOT_PATH="/project/def-dsteph/${USER}/julia:${JULIA_DEPOT_PATH}"

# Run: scripts self-activate the project
srun --cpu-bind=cores julia --project ./simulation/sim5.jl
