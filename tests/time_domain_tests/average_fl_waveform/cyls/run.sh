for BC in d n i
do
for RM in 10
do
for RC in 0.2
do
for VF in 0.05 0.15 0.3
do
    export BC RM RC VF DO
    sbatch --job-name=cyls_${BC}_${RM}_${VF}_${RC} setup.slurm
done
done
done
done
