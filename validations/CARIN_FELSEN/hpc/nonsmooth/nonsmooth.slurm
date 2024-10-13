#!/bin/bash -l
#SBATCH --partition=general
#SBATCH --qos=low
#SBATCH --account=peterp
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16GB
#SBATCH --ntasks=1
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=8
#SBATCH --threads-per-core=1
#SBATCH --hint=nomultithread
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nh287@njit.edu
#SBATCH -o /project/peterp/nh287/logs/%x_%A_%a.out
#SBATCH -e /project/peterp/nh287/logs/%x_%A_%a.out
#SBATCH --open-mode=append

python3 /home/nh287/scattering/validations/CARIN_FELSEN/hpc/nonsmooth/cf_nonsmooth_hpc.py ${bc} ${Np} ${Nm} /project/peterp/nh287/data/cf/
