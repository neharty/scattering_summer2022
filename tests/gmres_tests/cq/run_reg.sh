Nt=12

#regular array
for Nmode in {11..20}
do
for Ncyl in 6
do
for GMTOL in 1e-4 1e-6 1e-8 1e-10
do
    python3 cond_num_iters.py $Nt $Nmode $GMTOL $Ncyl reg 0 /home/nick/scattering/data/gmres_tests/cq/reg/Ncyl_${Ncyl}/
done
done
done
