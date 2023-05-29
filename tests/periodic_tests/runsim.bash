NX=10
NY=5
MAX=$(($NX*$NY-1))

for INC in 6 4 3 2 
do 
for BC in d n 
do
for NR in $(seq 0 $MAX) 
do
    echo "${NX}, ${NY}, ${INC}, ${BC}, ${NR}"
    export NX NY BC NR INC
    #
    sbatch --job-name=pt_Nx${NX}_Ny${NY}_inc${INC}_bc${BC}_Nr${NR} \
    runsim.slurm
    #
done
done
done
