SD=666
BC=i
RM=10
RC=0.05
VF=0.1
DO=/home/n/nh287/scattering/tests/time_domain_tests/average_fl_waveform/cyls/${BC}/${RM}/${VF}/${RC}/${SD}/
export SD BC RM RC VF DO
sbatch --job-name=cyls_${BC}_${RM}_${VF}_${RC}_${SD} setup.slurm

