Nt=12
#random array
for Nmode in 9 10
do
for rho in 0.05 0.10 0.15 0.20
do
for GMTOL in 1e-4
do
    python3 cq_wavenums_scmat.py $Nt $Nmode $GMTOL 0 rand $rho /home/nick/scattering/data/mat_properties_tests/cq/ran/rho_${rho}/
done
done
done
