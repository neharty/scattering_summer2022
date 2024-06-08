for bc in n d
do
for shape in fat_starfish
do
    python3 /home/nick/scattering/tmatrix/scripts/run_generate_tmatrix.py 6 ${shape} /home/nick/scattering/tests/tmatrix_tests/bdf2k_T40_Nt15.dat ${bc} 1 64 20 120 1.5707963267948966 /home/nick/scattering/tmatrix/rotation_pid2/
done
done
