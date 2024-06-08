for bc in n d
do
for shape in fat_starfish cyl
do
    for i in 0 1 2
    do
        python3 /home/nick/scattering/tmatrix/scripts/run_generate_tmatrix.py 6 ${shape} /home/nick/scattering/tests/tmatrix_tests/irk5_${i}_T40_Nt14.dat ${bc} 1 256 20 200 /home/nick/scattering/tmatrix/data/
    done
done
done
