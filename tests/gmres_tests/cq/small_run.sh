Nt=12
Nmode=10
GMTOL=1e-6
Ncyl=3
#regular array
python3 cond_num_iters.py $Nt $Nmode $GMTOL $Ncyl reg 0 /home/nick/scattering/data/gmres_tests/cq/reg/Ncyl_${Ncyl}/

#random array
rho=0.10
python3 cond_num_iters.py $Nt $Nmode $GMTOL 0 rand $rho /home/nick/scattering/data/gmres_tests/cq/ran/rho_${rho}/
