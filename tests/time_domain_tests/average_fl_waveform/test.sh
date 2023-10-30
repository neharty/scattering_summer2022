BC=d
RM=10
RC=0.05
WV=g
VF=0.1
NT=12
SD=1
CP=/home/n/nh287/scattering/tests/time_domain_tests/average_fl_waveform/cyls/${BC}/${RM}/${VF}/${RC}/${SD}/
DO=/home/n/nh287/scattering/data/time_domain_tests/average_fl_waveform/${WV}/${BC}/${NT}/${RM}/${VF}/${RC}/${SD}/
export BC RM RC WV NC NT SD CP
sbatch --job-name=epc_${WV}_${NC}_${NT}_${SD} setup.slurm

