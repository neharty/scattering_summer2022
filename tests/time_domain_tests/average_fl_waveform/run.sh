for WV in g
do
for NC in 500
do
for NT in 10
do
for SD in {1..30}
do
        DO=/home/n/nh287/scattering/data/time_domain_tests/average_fl_waveform/${WV}/${NC}/${NT}/${SD}
        export WV NC NT SD
        sbatch --job-name=epc_${WV}_${NC}_${NT}_${SD} setup.slurm
done
done
done
done

