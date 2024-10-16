#!/bin/bash -l
#SBATCH --partition=general
#SBATCH --qos=low
#SBATCH --account=peterp
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1GB
#SBATCH --ntasks=1
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=8
#SBATCH --threads-per-core=1
#SBATCH --array=0-8192%256
#SBATCH --hint=nomultithread
#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nh287@njit.edu
#SBATCH -o /project/peterp/nh287/logs/%x_%A_%a.out
#SBATCH -e /project/peterp/nh287/logs/%x_%A_%a.out
#SBATCH --open-mode=append

python3 -u /home/nh287/scattering/tmatrix/scripts/wideband/make_tmatrix_1k.py ${alp} ${bc} ${rad} ${Np} ${Nm} /home/nh287/scattering/tmatrix/kfiles/kfile_nonsmooth_Npoints${Np}_Nmodes_tmat${Nm}_T${T}_Nt${Nt}.dat ${SLURM_ARRAY_TASK_ID} strip /project/peterp/nh287/tmatrix/data/
