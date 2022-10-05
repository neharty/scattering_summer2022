#!/bin/bash -l
#SBATCH --job-name=run_fig5
#SBATCH --partition=public 
#SBATCH --mem=8GB
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --time=1-0
#SBATCH --mail-type=end
#SBATCH --mail-user=nh287@njit.edu
#SBATCH --output=/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/logs/fig5/%x-%j.out
#SBATCH --error=/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/logs/fig5/%x-%j.err

python3 -u /home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/test_scripts/fig5/fig5_dirichlet.py
python3 -u /home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/test_scripts/fig5/fig5_neumann.py
python3 -u /home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/test_scripts/fig5/fig5_impedance_small.py
python3 -u /home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/test_scripts/fig5/fig5_impedance_large.py
