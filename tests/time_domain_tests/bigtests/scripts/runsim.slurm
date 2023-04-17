#!/bin/bash -l
#SBATCH --partition=dms-cpu
#SBATCH --mem=4GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --nodes=1
#SBATCH --time=0-5
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nh287@njit.edu
#SBATCH -o /home/n/nh287/scattering/tests/time_domain_tests/bigtests/logs/%j.out
#SBATCH -e /home/n/nh287/scattering/tests/time_domain_tests/bigtests/logs/%j.out
#SBATCH --open-mode=append

#OUTDIR=/home/n/nh287/scattering/tests/time_domain_tests/bigtests/logs/${SLURM_JOB_ID}
#mkdir -p "$OUTDIR"
#logfile="${OUTDIR}/out_wvf${WVF}_s${SIZE}_o${ORDER}_sp${SPACING}_bc${BC}.out"

python3 -u /home/n/nh287/scattering/tests/time_domain_tests/bigtests/scripts/runsim.py ${WVF} ${SIZE} ${ORDER} ${SPACING} ${BC}
