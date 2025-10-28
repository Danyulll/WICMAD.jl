#!/bin/bash
#SBATCH --job-name=wicmad_simulation-jl
#SBATCH --account=def-dsteph
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.krasnov@mail.mcgill.ca

set -euo pipefail

# Load Julia (adjust if your cluster uses a different module path/name)
module load julia/1.11.3

# Optional: avoid thread oversubscription (we’re using processes, not threads)
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export JULIA_NUM_THREADS=1

# Give workers longer than the default 60s to start
export JULIA_WORKER_TIMEOUT=180

# Run from the submit directory
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Print job information
echo "Job ID:           $SLURM_JOB_ID"
echo "Job Name:         $SLURM_JOB_NAME"
echo "Nodes:            $SLURM_NODELIST"
echo "Tasks (ntasks):   ${SLURM_NTASKS:-1}"
echo "CPUs per task:    ${SLURM_CPUS_PER_TASK:-1}"
echo "Mem per CPU:      ${SLURM_MEM_PER_CPU:-unknown}"
echo "Start time:       $(date)"

echo
echo "Creating hostfile via srun..."
srun hostname -s > hostfile
sleep 2
echo "Hostfile lines: $(wc -l < hostfile) (expected ${SLURM_NTASKS:-1})"
echo

echo "Starting WICMAD simulation study with machine file..."
# Important: do NOT call addprocs() in the Julia script for Option B
julia --project=. --machine-file ./hostfile simulation_study.jl

echo "Done at: $(date)"
