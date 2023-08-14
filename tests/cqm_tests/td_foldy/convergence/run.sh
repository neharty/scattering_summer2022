for NS in 8 9 10 11 12 13 14 15
do
for SC in 1 2 3 4 
do
    export NS SC
    sbatch --job-id=${NS}_${SC} driver.slurm
done
done