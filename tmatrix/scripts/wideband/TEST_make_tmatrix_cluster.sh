alp=2
bc=d
rad=1
Np=64
Nm=10
T=40
Nt=15
dout=/home/nh287/scattering/tmatrix/data/
export alp bc rad Np Nm T Nt dout
sbatch --job-name=TMAT_${rad}_${T}_${bc} /home/nh287/scattering/tmatrix/scripts/setup.slurm
