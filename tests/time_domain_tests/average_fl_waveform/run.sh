BC=i
RM=10
RC=0.05
WV=g
for VF in 0.01
do
for NT in 12
do
for SD in {1..200}
do
        CP=/home/n/nh287/scattering/tests/time_domain_tests/average_fl_waveform/cyls/${BC}/${RM}/${VF}/${RC}/${SD}/
        DO=/home/n/nh287/scattering/data/time_domain_tests/average_fl_waveform/waveform_${WV}/bc_${BC}/samp_${NT}/R_${RM}/vf_${VF}/rc_${RC}/seed_${SD}/
        export BC RM RC WV NT SD CP DO
        sbatch --job-name=epc_${WV}_${BC}_${NT}_${SD} setup.slurm
done
done
done
