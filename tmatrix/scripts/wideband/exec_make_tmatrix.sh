alp=4
Nm=128
Nt=14
rad=0.5
T=40.0
fft_tol=1e-10
export alp bc rad Np Nm T Nt fft_tol

for Np in 64 128 256
do
for bc in d n
do
	export bc Np
	sbatch --job-name=Tmatrix_maker_${bc}_${Np} /home/nh287/scattering/tmatrix/scripts/wideband/make_tmatrix.slurm
done
done
