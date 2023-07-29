for BC in d n i m
do
#for N in 8 9 10 11 12 13
for N in 14 15 16
do  
    export BC N
    sbatch --job-name=${BC}_${N} runsim.slurm
done
done

