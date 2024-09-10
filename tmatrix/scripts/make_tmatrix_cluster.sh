alp=6
Np=256
Nm=30
Nt=15
dout=/home/nh287/scattering/tmatrix/data/
export alp Np Nm Nt dout

for rad in 1 0.5
do
for T in 40 80 120 240 360
do
for bc in d n
do
    export rad T bc	
    sbatch --job-name=TMAT_${rad}_${T}_${bc} /home/nh287/scattering/tmatrix/scripts/setup.slurm
done
