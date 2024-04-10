for CFG_TYPE in rect
do
        CFG=/home/nh287/scattering/drivers/small_tests/${CFG_TYPE}_config.cfg
        export CFG
        sbatch --job-name=small_test_${CFG_TYPE} setup.slurm
done
