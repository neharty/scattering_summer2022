TI=bdf2
BC=r
WV=dg
NT=10
NM=16
DO=/home/n/nh287/scattering/data/cqm/convergence/5x10/${TI}/${WV}/${BC}/${NT}/${NM}
export TI BC WV NT NM DO
sbatch --job-name=5x10_${BC}_${WV}_${TI}_${NT}_${NM} convdriver.slurm

