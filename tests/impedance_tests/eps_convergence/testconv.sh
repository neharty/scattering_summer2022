WV=g
EPS=1e-1
NT=25
DO=/home/n/nh287/scattering/data/cqm/impedance/eps_converge/${WV}/${EPS}/${NT}/
export WV EPS NT DO
sbatch --job-name=epc_${WV}_${EPS}_${NT} convdriver.slurm

