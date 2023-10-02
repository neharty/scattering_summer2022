#!/bin/bash -l
#SBATCH --partition=dms-cpu
#SBATCH --mem=16GB
#SBATCH --ntasks=1
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=6
#SBATCH --threads-per-core=1
#SBATCH --hint=nomultithread
#SBATCH --nodes=1
#SBATCH --time=1-0
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=neharty.slurm@gmail.com
#SBATCH -o /home/n/nh287/scattering/tests/cqm_tests/logs/%j.out
#SBATCH -e /home/n/nh287/scattering/tests/cqm_tests/logs/%j.out
#SBATCH --open-mode=append

python3 -u /home/n/nh287/scattering/tests/cqm_tests/convergence_tests/3x3/conv.py 3 ${BC} ${NT} ${WV} ${TI} ${NM} ${DO}
