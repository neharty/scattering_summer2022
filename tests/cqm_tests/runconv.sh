for TI in bdf2 irk3 irk5
do
for BC in d n 
do
for WV in dg g 
do
for NT in 10 11 12 13 14 15
do
    for NM in 4 8 16 32
    do
        DO=/home/n/nh287/scattering/data/cqm/convergence/${TI}/${WV}/${BC}/${NT}/${NM}
        export TI BC WV NT NM DO
        sbatch --job-name=${BC}_${WV}_${TI}_${NT}_${NM} convdriver.slurm
    done
done
done
done
done
