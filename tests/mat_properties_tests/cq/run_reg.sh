Nt=12

#regular array
for Nmode in {7..10}
do
for Ncyl in {4..10}
do
for GMTOL in 1e-4
do
    python3 cq_wavenums_scmat.py $Nt $Nmode $GMTOL $Ncyl reg 0 /home/nick/scattering/data/mat_properties_tests/cq/reg/Ncyl_${Ncyl}/
done
done
done
