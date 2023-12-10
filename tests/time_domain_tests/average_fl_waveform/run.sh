RM=10
RC=0.2
WV=g
for BC in d
do
for VF in 0.3
do
for NT in 14
do
for SD in {1..100}
do
        CP=/home/n/nh287/scattering/tests/time_domain_tests/average_fl_waveform/cyls/${BC}/${RM}/${VF}/${RC}/${SD}/
        DO=/home/n/nh287/scattering/data/time_domain_tests/average_fl_waveform/waveform_${WV}/bc_${BC}/samp_${NT}/R_${RM}/vf_${VF}/rc_${RC}/seed_${SD}/
        export BC RM RC WV NT SD CP DO
        sbatch --job-name=avg_${SD}_${BC}_${VF}_${NT} setup.slurm
done
done
done
done
