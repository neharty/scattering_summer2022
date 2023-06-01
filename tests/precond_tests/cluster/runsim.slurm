#!/bin/bash -l
#SBATCH --job-name=precond_comparison
#SBATCH --partition=dms-cpu
#SBATCH --mem=24GB
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=6
#SBATCH --threads-per-core=1
#SBATCH --nodes=1
#SBATCH --time=1-0
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=neharty.slurm@gmail.com
#SBATCH -o /home/n/nh287/scattering/tests/precond_tests/logs/%j.out
#SBATCH -e /home/n/nh287/scattering/tests/precond_tests/logs/%j.out
#SBATCH --open-mode=append

python3 -u /home/n/nh287/scattering/tests/precond_tests/cluster/precond_compare.py ${PRC}
