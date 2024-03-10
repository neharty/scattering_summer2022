Nt=12
#random array
for rho in 0.05 0.10 0.15 0.20
do
    python3 cq_wavenums_scmat.py $Nt $GMTOL 0 rand $rho /home/nick/scattering/data/mat_properties_tests/cq/ran/rho_${rho}/
done
