for TI in bdf2 irk3 irk5
do
for BC in d n i m r
do
for WV in dg g
do
for NT in 15
do
    for NM in 32 
    do
        DO=/home/n/nh287/scattering/data/cqm/convergence/5x10/${TI}/${WV}/${BC}/${NT}/${NM}
        export TI BC WV NT NM DO
        sbatch --job-name=5x10_${BC}_${WV}_${TI}_${NT}_${NM} convdriver.slurm
    done
done
done
done
done
