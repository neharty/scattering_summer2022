for TI in bdf2 irk3 irk5
for BC in d n 
do
for WV in dg g 
do
export BC WV
#    for NT in 8 9 10 11 12 13 14 15
#    do
#        NM=8
#        DO=/home/n/nh287/scattering/data/cqm/convergence/${WV}/${BC}/${NT}/${NM}
#        export NT NM DO
#        sbatch --job-name=${BC}_${WV}_${NT}_${NM} convdriver.slurm
#    done
    
#    for NM in 8 16 32 64 
#    do
#        NT=12
#        DO=/home/n/nh287/scattering/data/cqm/convergence/${WV}/${BC}/${NT}/${NM}
#        export NT NM DO
#        sbatch --job-name=${BC}_${WV}_${NT}_${NM} convdriver.slurm
#    done

for NT in 8 9 10 11 12 13 14 15
do
    for NM in 4 8 16 32 64
    do
        DO=/home/n/nh287/scattering/data/cqm/convergence/${TI}/${WV}/${BC}/${NT}/${NM}
        export NT NM DO
        sbatch --job-name=${BC}_${WV}_${TI}_${NT}_${NM} convdriver.slurm
    done
done
done
done
done
