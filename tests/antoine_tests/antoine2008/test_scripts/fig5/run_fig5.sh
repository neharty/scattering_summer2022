#!/bin/bash -l
#SBATCH --job-name=make_fig5
#SBATCH --partition=public 
#SBATCH --mem-per-cpu=4GB
#SBATCH --nodes=1
#SBATCH --time=24:00:00 
#SBATCH --mail-type=end
#SBATCH --mail-user=nh287@njit.edu
#SBATCH --output=%j-%x.out
#SBATCH --error=%j-%x.err

python3 -u /home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/test_scripts/fig5/fig5.py
