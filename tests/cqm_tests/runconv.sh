for BC in d n do
for WV in dg g do
export BC WV
    for NT in 8 9 10 11 12 13 14 15 do
        NM=32
        DO=/home/n/nh/scattering/data/cqm/convergence/${WV}/${BC}/${NT}/${NM}
        export NT NM DO
        sbatch --job-name=${BC}_${WV}_${NT}_NM runsim.sh
    done
    for NM in 8 16 32 64 do
        NT=15
        DO=/home/n/nh/scattering/data/cqm/convergence/${WV}/${BC}/${NT}/${NM}
        export NT NM DO
        sbatch --job-name=${BC}_${WV}_${NM}_NT runsim.sh
    done
done
done