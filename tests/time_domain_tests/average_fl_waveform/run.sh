for EPS in 1 1e-1 1e-2 1e-3
do
for WV in g
do
for NT in {10..30}
do
        DO=/home/n/nh287/scattering/data/cqm/impedance/eps_converge/${WV}/${EPS}/${NT}/
        export WV EPS NT DO
        sbatch --job-name=epc_${WV}_${EPS}_${NT} convdriver.slurm
done
done
done

