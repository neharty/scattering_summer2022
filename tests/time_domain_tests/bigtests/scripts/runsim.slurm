#!/bin/bash -l
#SBATCH --partition=dms-cpu
#SBATCH --mem=16GB
#SBATCH --ntasks=1
#SBATCH --sockets-per-node=2
#SBATCH --cores-per-socket=6
#SBATCH --threads-per-core=1
#SBATCH --nodes=1
#SBATCH --time=1-0
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nh287@njit.edu
#SBATCH -o /home/n/nh287/scattering/tests/time_domain_tests/bigtests/logs/%j.out
#SBATCH -e /home/n/nh287/scattering/tests/time_domain_tests/bigtests/logs/%j.out
#SBATCH --open-mode=append

python3 -u /home/n/nh287/scattering/tests/time_domain_tests/bigtests/scripts/runsim.py ${WVF} ${SIZE} ${ORDER} ${SPACING} ${BC} 12
