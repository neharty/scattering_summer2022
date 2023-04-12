#!/bin/bash -l
#SBATCH --partition=public 
#SBATCH --mem=16GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --time=1-0
#SBATCH --mail-type = BEGIN, END, FAIL
#SBATCH --mail-user=nh287@njit.edu

python3 -u /home/n/nh287/scattering/tests/time_domain_tests/bigtests/scripts/runsim.py ${WVF} ${SIZE} ${ORDER} ${SPACING} ${BC}
