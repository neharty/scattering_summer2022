#!/bin/bash -l
#SBATCH --partition=general
#SBATCH --qos=low
#SBATCH --account=peterp
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=6GB
#SBATCH --ntasks=1
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=4
#SBATCH --threads-per-core=1
#SBATCH --array=1-30
#SBATCH --hint=nomultithread
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nh287@njit.edu
#SBATCH -o /project/peterp/nh287/logs/%x_%A_%a.out
#SBATCH -e /project/peterp/nh287/logs/%x_%A_%a.out
#SBATCH --open-mode=append

DO=/project/peterp/nh287/data/condnum/dmin_${dmin}/vf_${VF}/
export DO

mkdir -p ${DO}

python3 -u /home/nh287/scattering/drivers/condnum/cond_vs_Nmode_cluster.py 1234 ${VF} ${SLURM_ARRAY_TASK_ID} ${MT} ${CFG} ${DO}
mv /project/peterp/nh287/logs/${MT}_${VF}_${dmin}_condnum_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out ${DO}${MT}_${VF}_${dmin}_condnum_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out
