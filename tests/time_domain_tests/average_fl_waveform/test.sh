BC=n
RM=10
RC=0.05
WV=g
VF=0.05
NT=12
for SD in {900..905}
do
    CP=/home/n/nh287/scattering/tests/time_domain_tests/average_fl_waveform/cyls/${BC}/${RM}/${VF}/${RC}/${SD}/
    DO=/home/n/nh287/scattering/data/time_domain_tests/average_fl_waveform/waveform_${WV}/bc_${BC}/samp_${NT}/R_${RM}/vf_${VF}/rc_${RC}/seed_${SD}/
    export BC RM RC WV NC NT SD CP DO
    sbatch --job-name=avg_${WV}_${BC}_${NT}_${SD} setup.slurm
done

