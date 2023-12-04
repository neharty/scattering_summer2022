#!/bin/bash -l
#SBATCH --partition=public
#SBATCH --mem=8GB
#SBATCH --ntasks=1
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=4
#SBATCH --threads-per-core=1
#SBATCH --hint=nomultithread
#SBATCH --nodes=1
#SBATCH --time=7-0
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=neharty.slurm@gmail.com
#SBATCH -o /home/n/nh287/scattering/tests/time_domain_tests/average_fl_waveform/logs/%j.out
#SBATCH -e /home/n/nh287/scattering/tests/time_domain_tests/average_fl_waveform/logs/%j.out
#SBATCH --open-mode=append

python3 -u /home/n/nh287/scattering/tests/time_domain_tests/average_fl_waveform/exec.py ${NT} ${WV} ${CP} ${DO}
