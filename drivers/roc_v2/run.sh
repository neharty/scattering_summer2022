for VF in 0.1 0.15 0.2 0.25 0.3
do
for WIDTH in 0.25 0.5 1.0
do
	CFG=/home/nh287/scattering/drivers/rates_of_convergence_circle/circle_config.cfg
	export CFG VF WIDTH
	sbatch --job-name=vf${VF}_width${WIDTH}_roc_circ setup.slurm
done
done
