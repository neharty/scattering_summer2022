#!/bin/bash -l
#SBATCH --job-name=avg_${BC}_${RM}_${VF}_${RC}
#SBATCH --partition=public
#SBATCH --mem=2GB
#SBATCH --ntasks=1
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=2
#SBATCH --threads-per-core=1
#SBATCH --array=1-200
#SBATCH --hint=nomultithread
#SBATCH --nodes=1
#SBATCH --time=1-0
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=neharty.slurm@gmail.com
#SBATCH -o /home/n/nh287/scattering/tests/time_domain_tests/average_fl_waveform/logs/%j.out
#SBATCH -e /home/n/nh287/scattering/tests/time_domain_tests/average_fl_waveform/logs/%j.out
#SBATCH --open-mode=append

DO=/home/n/nh287/scattering/tests/time_domain_tests/average_fl_waveform/cyls/${BC}/${RM}/${VF}/${RC}/${SLURM_ARRAY_TASK_ID}/
export DO

python3 -u /home/n/nh287/scattering/tests/time_domain_tests/average_fl_waveform/cyls/cyl_gen.py ${SLURM_ARRAY_TASK_ID} ${BC} ${RM} ${RC} ${VF} ${DO}
