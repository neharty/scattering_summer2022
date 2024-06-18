for bc in n d
do
for shape in fat_starfish
do
    python3 /home/nick/scattering/tmatrix/scripts/run_generate_tmatrix.py 6 ${shape} /home/nick/scattering/tests/tmatrix_tests/bdf2k_T80_Nt15.dat ${bc} 1 64 20 120 0 /home/nick/scattering/tmatrix/data/
done
done
