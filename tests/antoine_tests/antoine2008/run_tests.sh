for n in 5 6 7 9 10 11 12 13 14 15 16
do
    sbatch make_figs/fig$n/make_fig$n.sh 
done
wait
