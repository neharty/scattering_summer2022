#!/bin/bash -l
#SBATCH --job-name=run_tds_fig18
#SBATCH --partition=public 
#SBATCH --mem-per-cpu=4GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=7-0 
#SBATCH --mail-type=end
#SBATCH --mail-user=nh287@njit.edu
#SBATCH --output=/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/logs/fig18/%x-%j.out
#SBATCH --error=/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/logs/fig18/%x-%j.err

python3 -u /home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/test_scripts/fig18/fig18_dirichlet_tds.py
