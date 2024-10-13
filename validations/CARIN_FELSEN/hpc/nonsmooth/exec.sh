Nm=64
for Np in 64 128 256
do
for bc in d n
do
	export bc Np Nm
	sbatch --job-name=cf_${bc}_${Np} /home/nh287/validations/CARIN_FELSEN/hpc/nonsmooth/nonsmooth.slurm
done
done

