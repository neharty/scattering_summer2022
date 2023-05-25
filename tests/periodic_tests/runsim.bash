for inc in 
for bc in d n do
    echo "${WVF}, ${SIZE}, ${ORDER}, ${SPACING}, ${BC}"
    export WVF SIZE ORDER SPACING BC 
    #
    sbatch --job-name=pt_${WVF}_s${SIZE}_o${ORDER}_sp${SPACING}_bc${BC} \
    runsim.slurm
    #
done
done

