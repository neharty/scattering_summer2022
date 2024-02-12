Nt=12
#random array
for Nmode in {1..10}
do
for rho in 0.05 0.10 0.15 0.20
do
for GMTOL in 1e-4 1e-6 1e-8 1e-10
do
    python3 cond_num_iters.py $Nt $Nmode $GMTOL 0 rand $rho /home/nick/scattering/data/gmres_tests/cq/ran/rho_${rho}/
done
done
done
