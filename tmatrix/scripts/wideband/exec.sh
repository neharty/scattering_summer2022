alp=6
Np=128
Nm=80
Nt=15
rad=0.5
T=40
fft_tol=1e-5

for bc in d n
do
	python3 /home/nh287/scattering/tmatrix/scripts/wideband/make_ks.py $alp $bc $rad $Np $Nm $T $Nt ${fft_tol} /home/nh287/scattering/tmatrix/kfiles/
done
sbatch --job_name=Tmatrix_maker /home/nh287/scattering/tmatrix/scripts/wideband/setup.slurm
