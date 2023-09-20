TI=irk5
BC=d
WV=dg
NT=10
NM=16
DO=/home/n/nh287/scattering/data/cqm/convergence/${TI}/${WV}/${BC}/${NT}/${NM}
export TI BC WV NT NM DO
sbatch --job-name=${BC}_${WV}_${TI}_${NT}_${NM} convdriver.slurm

