for CTYPE in donut filled_circ
do
for VF in 0.2 0.25 0.3
do
	CFG=/home/nh287/scattering/tests/cond_num_tests/${CTYPE}.cfg
	export CTYPE CFG VF
	sbatch --job-name=vf${VF}_${CTYPE}_cnum setup.slurm
done
done
