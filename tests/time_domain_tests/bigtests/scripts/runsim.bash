for WVF in g dg ; do
    for SIZE in 0.01 0.2 0.5; do
        for ORDER in o r; do
            for SPACING in 3 7; do
                for BC in d n is il m; do
                    echo "${WVF}, ${SIZE}, ${ORDER}, ${SPACING}, ${BC}"
                    export WVF SIZE ORDER SPACING BC 
                    #
                    sbatch -o/home/n/nh287/scattering/tests/time_domain_tests/bigtests/logs/out_wvf${WVF}_s${SIZE}_o${ORDER}_sp${SPACING}_bc${BC}.out \
                    -e/home/n/nh287/scattering/tests/time_domain_tests/bigtests/logs/out_wvf${WVF}_s${SIZE}_o${ORDER}_sp${SPACING}_bc${BC}.err \
                    --job-name=tds_sim_wvf${WVF}_s${SIZE}_o${ORDER}_sp${SPACING}_bc${BC} \
                    runsim.slurm
                    #
                    sleep 1
                done
            done
        done
    done
done

