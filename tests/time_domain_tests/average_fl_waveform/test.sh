BC=d
RM=10
RC=0.2
WV=g
NT=14
for VF in 0.3
do
for SD in {900..905}
do
    CP=/home/n/nh287/scattering/tests/time_domain_tests/average_fl_waveform/cyls/${BC}/${RM}/${VF}/${RC}/${SD}/
    DO=/home/n/nh287/scattering/data/time_domain_tests/average_fl_waveform/waveform_${WV}/bc_${BC}/samp_${NT}/R_${RM}/vf_${VF}/rc_${RC}/seed_${SD}/
    export BC RM RC WV NC NT SD CP DO
    sbatch --job-name=${SD}_${VF}_${BC}_${NT} setup.slurm
done
done
