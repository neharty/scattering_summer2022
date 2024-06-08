for bc in n
do
for shape in fat_starfish cyl
do
    python3 /home/nick/scattering/tmatrix/erli_writeup_data/run_generate_tmatrix.py 6 ${shape} /home/nick/scattering/tests/tmatrix_tests/bdf2k_T40_Nt15.dat ${bc} 1 64 20 120 /home/nick/scattering/tmatrix/erli_writeup_data/
done
done
