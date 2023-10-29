${SD} ${BC} ${RM} ${RC} ${VF} ${DO} 
SD=666
BC=d
RM=10
RC=0.05
VF=0.1
DO=/home/n/nh287/scattering/data/time_domain_tests/average_fl_waveform/${BC}/${RM}/${VF}/${RC}/${SD}/
export SD BC RM RC VF DO
sbatch --job-name=avg_${BC}_${RM}_${VF}_${RC}_${SD} setup.slurm

