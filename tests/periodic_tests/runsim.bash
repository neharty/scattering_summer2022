NX=10
NY=5
BC=d


for INC in 6 4 3 2 
do 
    echo "${NX}, ${NY}, ${INC}, ${BC}"
    export NX NY BC INC
    #
    sbatch --job-name=pt_Nx${NX}_Ny${NY}_inc${INC}_bc${BC} \
    runsim.slurm
    #
done
