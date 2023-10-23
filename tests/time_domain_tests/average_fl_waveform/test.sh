WV=g
NC=500
NT=10
SD=666
DO=/home/n/nh287/scattering/data/time_domain_tests/average_fl_waveform/${WV}/${NC}/${NT}/${SD}/
export WV NC NT SD DO
sbatch --job-name=avg_${WV}_${NC}_${NT}_${SD} setup.slurm

