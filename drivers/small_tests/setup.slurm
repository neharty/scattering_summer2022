#!/bin/bash -l
#SBATCH --partition=general
#SBATCH --qos=low
#SBATCH --account=peterp
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --ntasks=1
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=4
#SBATCH --threads-per-core=1
#SBATCH --array=1-3
#SBATCH --hint=nomultithread
#SBATCH --nodes=1
#SBATCH --time=10-0
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=neharty.slurm@gmail.com
#SBATCH -o /project/peterp/nh287/test_logs/%x_%j.out
#SBATCH -e /project/peterp/nh287/test_logs/%x_%j.out
#SBATCH --open-mode=append

DO=/home/project/peterp/nh287/data/small_tests/${CFG_TYPE}/${SLURM_JOB_ARRAY_ID}
export DO

python3 -u /home/nh287/scattering/drivers/small_tests/exec_test.py ${SLURM_ARRAY_JOB_ID} ${CFG} ${DO}
