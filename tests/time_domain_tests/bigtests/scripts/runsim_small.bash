: '
for WVF in g dg ; do
    for SIZE in 0.01 0.2 0.5; do
        for ORDER in o r; do
            for SPACING in 3 7; do
                for BC in d n is il m; do
                    echo "${WVF}, ${SIZE}, ${ORDER}, ${SPACING}, ${BC}"
                    export WVF SIZE ORDER SPACING BC 
                    #
                    sbatch --job-name=tds_sim_wvf${WVF}_s${SIZE}_o${ORDER}_sp${SPACING}_bc${BC} \
                    runsim.slurm
                    #
                done
            done
        done
    done
done
'

WVF=dg
SIZE=0.2
ORDER=r
SPACING=3
BC=m
echo "${WVF}, ${SIZE}, ${ORDER}, ${SPACING}, ${BC}"
export WVF SIZE ORDER SPACING BC
sbatch --job-name=tds_sim_wvf${WVF}_s${SIZE}_o${ORDER}_sp${SPACING}_bc${BC} runsim.slurm

