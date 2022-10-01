#!/bin/bash -l
#SBATCH --job-name=fig7tests
#SBATCH --partition=public 
#SBATCH --mem-per-cpu=8GB
#SBATCH --nodes=1
#SBATCH --time=1-0
#SBATCH --mail-type=end
#SBATCH --mail-user=nh287@njit.edu

python3 test_scripts/fig7/fig7_gmres50.py

#python3 make_figs/make_fig7_gmres50.py

