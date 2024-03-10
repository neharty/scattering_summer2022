Nt=12

#regular array
for Ncyl in {4..10}
do
    python3 cq_wavenums_scmat.py $Nt $Ncyl reg 0 /home/nick/scattering/data/mat_properties_tests/cq/reg/Ncyl_${Ncyl}/
done
