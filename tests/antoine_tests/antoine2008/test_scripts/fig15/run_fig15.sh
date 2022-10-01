#!/bin/bash -l
#SBATCH --job-name=run_fig15
#SBATCH --partition=public 
#SBATCH --mem-per-cpu=8GB
#SBATCH --nodes=1
#SBATCH --time=2-0 
#SBATCH --mail-type=end
#SBATCH --mail-user=nh287@njit.edu
#SBATCH --output=/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/logs/fig15/%x-%j.out
#SBATCH --error=/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/logs/fig15/%x-%j.err

python3 -u /home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/test_scripts/fig15/fig15.py
