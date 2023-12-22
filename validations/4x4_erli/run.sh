for tint in irk5
do
for bcs in d n dndn ndnd
do
for inc in 1 2
do
for Nmode in {1..10}
do
for Nt in {5..14}
do
    export tint width bcs inc Nmode Nt
    sbatch --job-name=${tint}_${width}_${bcs}_${inc}_${Nmode}_${Nt} setup.slurm
done
done
done
done
done
