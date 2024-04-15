#!/bin/bash -l
#SBATCH --partition=general
#SBATCH --qos=low
#SBATCH --account=peterp
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32GB
#SBATCH --ntasks=1
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=8
#SBATCH --threads-per-core=1
#SBATCH --array=1-30
#SBATCH --hint=nomultithread
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nh287@njit.edu
#SBATCH -o /project/peterp/nh287/logs/%x_%j.out
#SBATCH -e /project/peterp/nh287/logs/%x_%j.out
#SBATCH --open-mode=append

DO=/project/peterp/nh287/data/cond_num/
export DO

python3 -u /home/nh287/scattering/tests/cond_num_tests/exec.py 1 ${VF} ${SLURM_ARRAY_TASK_ID} ${CFG} ${DO}
mv /project/peterp/nh287/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out ${DO}/${CTYPE}volfrac_${VF}/Nmode_${SLURM_ARRA_TASK_ID}/seed_1
