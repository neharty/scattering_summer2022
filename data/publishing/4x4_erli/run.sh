for tint in bdf2 irk3 irk5
do
for width in 0.5 1 5
do
for bcs in d n
do
for inc in 1 2 3 4 5
do
for Nmode in {1..10}
do
for Nt in {5..15}
do
    export tint width bcs inc Nmode Nt
    sbatch --job-name=${tint}_${width}_${bcs}_${inc}_${Nmode}_${Nt} setup.slurm
done
done
done
done
done
done
