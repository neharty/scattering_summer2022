RM=10
RC=0.05
WV=g
for BC in d n i
do
for VF in 0.01
do
for NT in 12
do
for SD in 170
do
        CP=/home/n/nh287/scattering/tests/time_domain_tests/average_fl_waveform/cyls/${BC}/${RM}/${VF}/${RC}/${SD}/
        DO=/home/n/nh287/scattering/data/time_domain_tests/average_fl_waveform/waveform_${WV}/bc_${BC}/samp_${NT}/R_${RM}/vf_${VF}/rc_${RC}/seed_${SD}/
        export BC RM RC WV NT SD CP DO
        sbatch --job-name=epc_${SD}_${BC}_${WV}_${NT} setup.slurm
done
done
done
done
