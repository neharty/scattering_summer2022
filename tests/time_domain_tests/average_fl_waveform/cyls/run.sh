for BC in d
do
for RM in 10
do
for RC in 0.05
do
for VF in 0.05 0.1 0.15 0.20
do
    export BC RM RC VF DO
    sbatch --job-name=avg_${BC}_${RM}_${VF}_${RC} setup.slurm
done
done
done
done
